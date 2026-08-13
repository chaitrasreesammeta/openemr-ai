"""Direct LLM prompting: show the note and the candidate codes, ask for codes.

The prompt asks for a verbatim quote alongside every code. That is not decorum.
The quote is what the evidence metric scores, and it is what a human coder needs
in order to accept or reject a suggestion, so an approach that cannot produce it
is not a candidate for deployment however good its F1 is.
"""

from __future__ import annotations

import json
import re
import time

from coding_bench.approaches.base import (
    Candidate,
    Completion,
    LLMClient,
    Note,
    Prediction,
    Truncated,
    locate_span,
)

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are a certified professional medical coder. You assign \
billing codes to clinical documentation, and you justify every code with text \
from the note itself.

Rules:
1. Assign a code only when the note documents it explicitly. Do not infer codes \
from what is clinically likely.
2. Do not code conditions that are negated, hypothetical, historical, or \
attributed to a family member.
3. For every code you assign, quote the shortest span of the note that supports \
it, copied verbatim, character for character.
4. Choose codes only from the candidate list you are given.
5. If the note supports no codes from the list, return an empty list. An empty \
answer is correct when nothing applies.

Answer with JSON only, in exactly this shape, and nothing else:
{"codes": [{"code": "<code>", "quote": "<verbatim text from the note>"}]}"""

USER_TEMPLATE = """Code system: {code_system}

Candidate codes:
{candidates}

Clinical note:
<<<NOTE
{text}
NOTE

Assign the applicable codes from the candidate list. JSON only."""


class LLMPredictor:
    """Ask a model directly, once per note."""

    name = "llm"
    version = PROMPT_VERSION

    def __init__(
        self,
        client: LLMClient,
        code_system: str = "ICD-10-CM",
        max_tokens: int = 16384,
        max_note_chars: int | None = None,
        restrict_to_candidates: bool = False,
    ):
        self.client = client
        self.code_system = code_system
        self.max_tokens = max_tokens
        # Notes run long. Truncating the note is a deliberate, recorded choice,
        # not something to do silently, so it defaults to off.
        self.max_note_chars = max_note_chars
        self.restrict_to_candidates = restrict_to_candidates

    @property
    def model_id(self) -> str:
        return self.client.model_id

    def build_prompt(self, note: Note, candidates: list[Candidate] | None) -> tuple[str, str]:
        text = note.text
        if self.max_note_chars and len(text) > self.max_note_chars:
            text = text[: self.max_note_chars]
        return SYSTEM_PROMPT, USER_TEMPLATE.format(
            code_system=self.code_system,
            candidates=format_candidates(candidates or []),
            text=text,
        )

    def predict(self, note: Note, candidates: list[Candidate] | None = None) -> Prediction:
        system, user = self.build_prompt(note, candidates)
        start = time.perf_counter()
        completion: Completion = self.client.complete(system, user, self.max_tokens)
        elapsed = time.perf_counter() - start

        offered = {c.code for c in candidates} if candidates else None
        restrict = offered if self.restrict_to_candidates else None
        codes = parse_codes(completion.text, note.text, restrict)

        # Strict parsing first, always. The salvage only runs where the strict
        # path found nothing, so a response that parsed is untouched by it and
        # this cannot change an answer that already worked.
        salvaged = False
        if not codes:
            codes = salvage_codes(completion.text, restrict)
            salvaged = bool(codes)

        return Prediction(
            codes=codes,
            truncated=False,
            latency_s=completion.latency_s or elapsed,
            usage=completion.usage,
            salvaged=salvaged,
        )


def format_candidates(candidates: list[Candidate]) -> str:
    """One code per line, with the descriptor when we are allowed to have one."""
    lines = []
    for candidate in candidates:
        if candidate.description:
            lines.append(f"{candidate.code}\t{candidate.description}")
        else:
            lines.append(candidate.code)
    return "\n".join(lines) if lines else "(no candidate list supplied, use your own knowledge)"


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def balanced_objects(text: str) -> list[str]:
    """Every balanced {...} span in the text, in order of appearance.

    Brace counting rather than a regex, because a regex either stops at the
    first closing brace, which truncates any nested object, or spans from the
    first brace to the last, which welds separate objects into nonsense. String
    literals are tracked so that a brace inside a quoted clinical phrase does
    not shift the depth.
    """
    spans: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    spans.append(text[start : index + 1])
                    start = -1

    return spans


def extract_json(text: str) -> dict | None:
    """Pull the answer object out of a response.

    Reasoning models think out loud before answering, and while thinking they
    restate the JSON they are about to produce. So the answer is the *last*
    parseable object, and among those, one that actually carries codes beats a
    passing mention. Picking the first object, or the widest brace span, silently
    returns a draft or nothing at all, which scores as "no codes apply" and is
    indistinguishable from the model genuinely finding nothing.
    """
    if not text:
        return None

    candidates: list[str] = list(_FENCE.findall(text))
    candidates.extend(balanced_objects(text))
    candidates.append(text.strip())

    parsed: list[dict] = []
    for chunk in candidates:
        try:
            value = json.loads(chunk.strip())
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            parsed.append(value)

    if not parsed:
        return None

    with_codes = [value for value in parsed if "codes" in value]
    return with_codes[-1] if with_codes else parsed[-1]


# A code inside a JSON field, e.g. {"code": "M54.50", ...}. Codes never contain
# a quote or a backslash, so this pattern cannot run past the end of its own
# string the way a general value pattern would.
_CODE_FIELD = re.compile(r'"code"\s*:\s*"([^"\\]{1,24})"')


def salvage_codes(response: str, restrict_to: set[str] | None = None) -> dict[str, None]:
    """Recover codes from an answer whose JSON does not parse.

    The prompt asks for a verbatim quote from the note beside every code, and
    clinical notes contain quote marks. Models copy them in without escaping
    them, and a single unescaped quote inside a JSON string flips the parser's
    in-string state: every brace after it is counted in the wrong state, no
    object ever closes at depth zero, and a complete answer is discarded as
    though the model had said nothing. It cost qwen 16 notes in one 578 note
    run, each of them naming the right codes in text nobody could read.

    Only the span after the last `"codes"` is scanned. Reasoning models restate
    their answer while thinking, so an earlier match is a draft the model went
    on to change its mind about, and the final answer is the last one.

    This never repairs the JSON, and it never guesses at evidence. A salvaged
    code carries no span, because the quote is exactly the field that could not
    be read, and inventing an offset would score as a near miss rather than as
    the loss it is. Runs report how many notes were salvaged so that the
    evidence coverage they cost is visible instead of just lower.
    """
    marker = response.rfind('"codes"')
    if marker == -1:
        return {}

    codes: dict[str, None] = {}
    for raw in _CODE_FIELD.findall(response[marker:]):
        code = normalise_code(raw, restrict_to)
        if not code:
            continue
        if restrict_to is not None and code not in restrict_to:
            continue
        codes[code] = None
    return codes


def normalise_code(code: str, offered: set[str] | None) -> str:
    """Tidy a model's code string, the same way for every model.

    Models drop the decimal point in ICD-10 often enough that rejecting those
    answers would measure formatting rather than coding. The repair is only
    attempted against the offered list, so it can never invent a code.
    """
    code = (code or "").strip().upper().rstrip(".")
    code = re.sub(r"\s+", "", code)
    if not offered or code in offered:
        return code
    if len(code) > 3 and "." not in code:
        dotted = f"{code[:3]}.{code[3:]}"
        if dotted in offered:
            return dotted
    return code


def parse_codes(
    response: str, note_text: str, restrict_to: set[str] | None = None
) -> dict[str, tuple[int, int] | None]:
    """Turn a model response into {code: span or None}."""
    payload = extract_json(response)
    if payload is None:
        return {}

    entries = payload.get("codes", payload) if isinstance(payload, dict) else payload
    if isinstance(entries, dict):
        entries = [{"code": code, "quote": quote} for code, quote in entries.items()]
    if not isinstance(entries, list):
        return {}

    codes: dict[str, tuple[int, int] | None] = {}
    for entry in entries:
        if isinstance(entry, str):
            code, quote = entry, None
        elif isinstance(entry, dict):
            code = entry.get("code") or entry.get("Code") or ""
            quote = entry.get("quote") or entry.get("evidence") or entry.get("text")
        else:
            continue

        code = normalise_code(str(code), restrict_to)
        if not code:
            continue
        if restrict_to is not None and code not in restrict_to:
            continue

        span = locate_span(note_text, quote) if quote else None
        # Keep the first span offered for a code, so repeats cannot overwrite a
        # good citation with a worse one.
        if code not in codes or codes[code] is None:
            codes[code] = span

    return codes
