# coding_bench

Benchmark for automated CPT and ICD-10 coding from clinical notes.

Everything here is built so that restricted note text cannot reach git or a
public log by accident. That is not a policy, it is the shape of the code: the
gold sets live on a private Modal volume, evaluation of restricted data runs
inside Modal, and the run record writer raises rather than serialise a note.

**Results: [`results/LEADERBOARD.md`](results/LEADERBOARD.md).** Four ranked
tables, one per task and candidate space, plus frequency bands, paired
significance, quarantined runs and per run provenance. It is generated from the
committed run records by `bench/reporting.py`, along with the four row summary
in the repository [README](../README.md#automated-cpt-and-icd-10-coding), and CI
fails if either is stale. Neither is ever edited by hand.

## Data tiers

| Tier | Content | Where it lives | Who can read it |
|---|---|---|---|
| 0 | 20 synthetic notes with gold CPT and ICD-10 labels and evidence spans | `data/smoke/notes.json`, in git | Anyone |
| 1 | Gold manifests, label spaces, checksums | `data/manifests/`, `data/labels/`, in git | Anyone |
| 2 | MDACE annotations joined to MIMIC-III note text | Modal volume `coding-benchmark-gold` | PhysioNet credentialed members of the Modal workspace |

The **gold manifest** is the contract between the tiers. For each note it records
`{note_id, gold_codes, text_sha256, char_len}`, which is committable because it
is MDACE's public annotations plus a one way hash of the text. It gives a public
definition of the benchmark, an integrity check that every rebuild is identical,
and a version stamp that CI uses to refuse to score against stale data.

## Tasks

| Task | Source | Notes | Codes |
|---|---|---|---|
| `cpt` | MDACE Profee, CPT annotations | 312 | 61 |
| `icd10` | MDACE Profee, ICD-10-CM annotations | 578 | 673 |
| `smoke_cpt`, `smoke_icd10` | Synthetic, Tier 0 | 20 | 17 and 24 |

MDACE is pinned at commit `bf438d0d` of `solventum-oss/MDACE`, and its offsets
index MIMIC-III v1.4 `NOTEEVENTS.csv` by `ROW_ID`. MIMIC-IV-Note cannot be
substituted: the note identifiers are different.

## Building the gold sets

One time, in your own terminal so that no credential stays in a log. The S3
route is the one to use, and it needs AWS keys and nothing else:

```bash
modal secret create physionet \
    AWS_ACCESS_KEY_ID=<key> AWS_SECRET_ACCESS_KEY=<secret>
```

The secret is named `physionet` because that is the name this app declares, not
because it has to hold a PhysioNet login. There is no username or password
anywhere in the build any more.

Then:

```bash
modal run coding_bench/data/build_remote.py                   # build, write Tier 1 back into the repo
modal run coding_bench/data/build_remote.py --force-download  # ignore the cached NOTEEVENTS
modal run coding_bench/data/build_remote.py --verify-only     # check the volume against the repo
modal run coding_bench/data/build_remote.py::inspect          # sizes on the volumes
```

NOTEEVENTS comes from PhysioNet's S3 access point, which grants your AWS
principal directly when you enable cloud access on the project page, so the
transfer is not billed to the caller. The download is cached on the raw volume,
so it is paid for once.

There were two other routes, aria2c with 16 connections and a plain wget, both
pulling over HTTP with the PhysioNet account password. Both are gone. Neither
was ever fast enough to choose, wget measured in hours for this file, and
removing them removes the reason to keep that password in a Modal secret at
all. The AWS keys stay on the `physionet` secret rather than an `aws` secret
beside it, because Modal resolves every secret in an app at startup, so a second
optional secret would break the build for anyone who has not created it.

The build downloads NOTEEVENTS from PhysioNet straight into a private volume,
clones the pinned MDACE, joins them, and writes the parquets to
`coding-benchmark-gold`. Only manifests and label spaces come back over the
wire. Restricted text never touches your machine.

The local path exists too, for anyone who wants to reproduce it from a
credentialed local copy:

```bash
python -m coding_bench.data.build --mdace ~/benchmark-data/MDACE \
    --noteevents /path/to/NOTEEVENTS.csv.gz --out-dir coding_bench/data
python coding_bench/data/publish_volume.py --gold-dir coding_bench/data/gold
```

Either path refuses to emit anything if a rebuilt note hashes differently from
the committed manifest.

## Running an evaluation

Tier 0, anywhere, no credentials:

```bash
python -m pytest coding_bench/tests -q
```

Tier 2, inside Modal:

```bash
modal run coding_bench/eval_remote.py --task icd10 --model muse-glimmer-30b --limit 25
modal run coding_bench/eval_remote.py --task cpt --model muse-glimmer-30b --candidate-space full
```

`--candidate-space` takes `gold`, an integer, or `full`. That is the label space
scaling stress: `gold` is the ceiling condition, `full` is the deployment
condition, and the distractor false positive rate between them is the number
that predicts how a model behaves against a real catalogue.

## Models

Values for `--model`. The leaderboard calls each of these by a short name and
keys on the exact model id; `reporting.MODEL_NAMES` maps between the two, and
the board's `## Models` section prints the mapping.

| `--model` | Board name | Where it runs | Sends note text off Modal |
|---|---|---|---|
| `qwen3.6-27b` | Qwen3.6 27B | Groq, `qwen/qwen3.6-27b` | Yes, to Groq |
| `gpt-oss-120b` | GPT-OSS 120B | Groq, `openai/gpt-oss-120b` | Yes, to Groq |
| `sonnet-5` | Claude Sonnet 5 | Anthropic, `claude-sonnet-5` | Yes, to Anthropic |
| `muse-glimmer-30b-gguf` | Muse Glimmer 30B | Modal RTX PRO 6000, llama.cpp, `Muse-Glimmer-30B-GGUF:kquant-dynamic` | No |
| `muse-glimmer-30b` | (never banked) | Modal H100, transformers, bf16 weights | No |
| `gemma4-26b-a4b-gguf` | Gemma 4 26B-A4B | Modal RTX PRO 6000, llama.cpp, `gemma-4-26B-A4B-it:qat-UD-Q4_K_XL` | No |
| `qwen3.8-27b-fp8` | Qwen3.8 27B FP8 | Modal RTX PRO 6000, vLLM, `Qwen/Qwen3.8-27B-FP8` | No |
| `qwen3.8-27b-bf16` | (never banked) | Modal RTX PRO 6000, vLLM, bf16 weights | No |

`--approach` takes `llm` (the default), `retr_llm`, and the three that need no
model at all: `retrieval` (BGE embeddings), `embed_match` and `entity_match`.
Those three run on CPU and send nothing anywhere.

The three hosted models post note text to somebody else's inference service.
The provider that saw the text is written into every run manifest as
`external_provider`, so the provenance of a number stays answerable without
anyone having to remember which runs went where. Anthropic is a second provider
and a second clearance question; the governance note at the top of
`adapters/api_anthropic.py` is worth reading before running it.

**Zero Data Retention must stay on for the Groq account.** By default Groq
retains inputs and outputs for up to 30 days for reliability and abuse
monitoring. ZDR is enabled per organisation on the Data Controls page in the Groq console,
covers the `/openai/v1/chat/completions` endpoint these adapters use, and
disables features that depend on retention, none of which this benchmark uses.
It is an account setting rather than something a run can assert about itself, so
it belongs in the operational checklist, not in the run manifest.

**The two Muse entries are different models and must never be compared as if
they were one.** `muse-glimmer-30b-gguf` runs Meta's official dynamic K-quant,
roughly 4 bit weights, through llama.cpp with the shipped DFlash speculative
drafter. Its model id carries the quantisation for exactly this reason. Meta
puts the average degradation for that build at 0.2 percent across fifteen
general benchmarks, but that figure should not be assumed to hold here, because
quantisation costs rare label recall first and the long tail is what this
benchmark measures.

The bf16 adapter is kept for the comparison that would settle that question, but
it is not practical at this scale: vLLM 0.27.1 does not register the
`muse_glimmer` architecture, so it falls back to unbatched transformers
generation, which measured at roughly six minutes per note and put a 578 note
run near sixty hours. llama.cpp exists in this repo because of that number, not
out of preference.

**Only the FP8 Qwen3.8 build is on the board.** `qwen3.8-27b-bf16` is registered
and is not queued, for the same reason the bf16 Muse adapter is not: it is the
arm that would settle what FP8 costs, and that is not the question a deployment
board answers. Which build gets the row is argued out in
[Qwen3.8 27B, and why FP8](#qwen38-27b-and-why-fp8) below.

**Gemma 4 is quantised too, and its adapter is a reconstruction.** The model is
a mixture of experts, 26B parameters with about 4B active per token, served as
Google's QAT weights in Unsloth's dynamic 4 bit build. Its id carries that build
for the same reason Muse's does. Four runs of it are banked and all four are
quarantined, every one on truncation, so Gemma has no score on the board and
nothing about it should be quoted as one. The adapter that produced them was
written on another machine and never pushed, so `adapters/modal_gemma4_gguf.py`
reproduces the configuration their manifests record rather than the original
bytes. Their `adapter_sha256` will not match the committed file, and the
docstring says so at the top.

## Qwen3.8 27B, and why FP8

Qwen releases this model twice, `Qwen/Qwen3.8-27B` at bf16 and
`Qwen/Qwen3.8-27B-FP8` at fine grained fp8 with block size 128. Only one of them
gets a row here, and it is the FP8 build.

**CI runs this one.** Unlike Muse and Gemma, the Qwen3.8 chain launches from the
commit that adds it and banks itself when it finishes. The `launch_gpu` job
deploys both apps, smokes the server, spawns the chain, watches it for up to
five hours, then collects, rescores, regenerates the board and commits. It runs
after `evaluate` so the two cannot race each other's commit, and it needs a
green `checks` to start, because the point of that gate is that a red suite must
never buy inference.

It fires on a push only when the push touched `adapters/modal_qwen38_vllm.py` or
`run_chain_qwen38.py`, which are exactly the changes that invalidate this model's
cached notes, and it refuses outright if records are already banked unless the
dispatch sets `confirm_gpu_rerun`. A comment edit in an adapter changes its hash
and therefore its cache key, and that must not silently buy the chain again.

The watch is not what keeps the run alive. `run_now` spawns against the deployed
app, so the chain belongs to Modal: cancel the job, hit the 350 minute ceiling,
lose the runner, and it carries on with every finished step already committed to
the results volume. The collect step runs on `always()` for that reason, so a
timed out watch still banks three conditions out of four, and `action: collect`
picks up the rest whenever it lands.

By hand, which is what a rerun or a bf16 ablation needs:

```bash
modal deploy coding_bench/adapters/modal_qwen38_vllm.py
modal run   coding_bench/adapters/modal_qwen38_vllm.py::smoke
modal deploy coding_bench/run_chain_qwen38.py
modal run   coding_bench/run_chain_qwen38.py::launch   # spawn and walk away
# or: ::run_now, which spawns and then watches
modal run   coding_bench/run_chain.py::fetch
python -m coding_bench.bench.reporting
```

Smoke first either way. A chain that discovers on step one that vLLM will not
serve this architecture has already paid for a GPU to find out, and `smoke` also
prints whether an fp8 kernel actually loaded.

### The build

**This is a deployment board.** `full` candidates exists as a condition because
gold candidates pin precision at 1.000 by construction and predict nothing about
deployment. Every self-hosted row on the board is a deployment build: Muse
Glimmer at roughly 4 bit, Gemma 4 at QAT 4 bit. A bf16 row would be the only
self-hosted number on it measured at a precision nobody would serve for a 27B on
this task.

**A bf16 row would not buy comparability with the hosted rows either.** Neither
Groq nor Anthropic publishes the precision it serves at, and at their prices and
throughput it is unlikely to be bf16. Running bf16 to match them would be a
claim about somebody else's stack that nobody outside it can check. FP8 is at
least a precision this repository can name in the `## Models` key.

**And it is a release rather than a quantisation of one**, with its own
repository and card, which is why its id needs no `:quantisation` tag the way
the GGUF builds do. Qwen puts it at nearly identical to the original, and the
largest study of the format measures W8A8-FP8 as lossless across model scales.
That evidence is about general benchmarks and says nothing about rare label
recall, so it is a reason to expect FP8 to be fine here and not a reason to skip
measuring it. **Read the tail band first**: quantisation costs rare label recall
before it costs anything a headline number can see, which is why this benchmark
reports frequency bands at all.

The bf16 arm stays registered and unqueued so that the ablation is a chain step
rather than a rewrite. `SERVER_ARGS` is shared and `server_command` varies only
the checkpoint, and `tests/test_adapters.py` asserts the two argument lists
differ in exactly those two positions. Two flags are named in that test because
they are the ones somebody will reasonably reach for: `--kv-cache-dtype fp8`,
which vLLM's own recipe for this model suggests and which on one arm alone would
quantise the weights and the cache and report the sum as the weight effect, and
`--max-num-seqs`, which the FP8 arm has the VRAM to raise and which would change
what it decodes as well as how fast.

### The configuration, which is where comparability actually lives

Four things are held to what the rest of the board used, and each is a place a
number could quietly stop being comparable.

**Thinking is on.** `reasoning_strength: "medium"`, the package default, which
on this adapter sets `enable_thinking: true` rather than prepending a line to
the system prompt. This is the equivalence that matters most and the easiest to
lose. The nearest row is Qwen3.6 27B, same family and same size, and Groq serves
it as a reasoning model. An instruct mode Qwen3.8 measured against it would
report a decoder difference as a model difference, and would flatter the newer
model on latency while doing it.

**max_tokens is 16,384**, what the board's ICD-10 and CPT runs used. It is in
the prediction cache key, so a different value is not only incomparable, it
misses every cached note and pays again.

**The card is the RTX PRO 6000**, at four concurrent notes, which is what Muse
Glimmer and Gemma 4 ran on. Latency between self-hosted rows is only a
comparison if the hardware is the same one.

**All four conditions run**, cpt and icd10 by gold and full, because a partial
row cannot be ranked against a full one.

### What will go wrong first

Truncation, not a low score. Greedy decoding in thinking mode is the
configuration Qwen warns about, because a trace can loop until it hits the cap
and no JSON is ever produced. The adapter raises that as `Truncated` rather than
returning an empty answer, the runner counts it separately, and a run over 5
percent failures is quarantined rather than scored. Muse ran the same way and
landed at 4.0 percent on ICD-10 full, just inside the ceiling. If a step
quarantines, raise `max_tokens` for that step and record that it no longer
matches the rest of the board.

Greedy at all is this package's standing deviation from every model card in it,
and it is worth knowing that greedy under vLLM is not bitwise reproducible
anyway: batched matmuls reduce in an order that depends on how many sequences
are in flight, so a rerun can disagree on a handful of notes with nothing having
changed.

## Cost and the prediction cache

Every note's prediction is cached, keyed on everything that can change the
answer: the gold manifest checksum, the note id, the model id, the adapter file
hash, the prompt hash, the exact candidate list, and the run parameters. Rerun
with nothing changed and the run costs nothing. Change the prompt, the adapter,
a parameter, or rebuild the gold set, and the affected entries miss and are
recomputed. Pass `--cache off` to force a genuine re-run.

**Failures are never cached.** A rate limit or a timeout describes the
afternoon, not the model, and caching one would freeze a transient error into
the results permanently. Truncation does cache, because that is the model
genuinely running out of room.

This matters more than it sounds. An interrupted 578 note run used to be a total
loss, and retrying twenty rate limited notes meant paying for the five hundred
that already succeeded.

Other things that drive the bill:

- Groq enforces a tokens per minute ceiling, and a full catalogue prompt is
  around 11k tokens, so sustainable concurrency is about 2 or 3. Going higher
  does not go faster, it just converts requests into 429s and retries.
- Groq's batch API is cheaper but depends on data retention, so zero data
  retention rules it out. Compliance wins that trade.
- GPU containers scale down after 90 seconds idle. A card waiting between runs
  costs the same as one doing work.
- `--limit` exists for iteration. Full runs are for numbers you intend to keep.

## Metrics

Core micro and macro precision, recall and F1, plus exact match, Jaccard and
label cardinality ratio. On top of that, and mandatory in every report:

- **Frequency band table.** Head (top 10 codes), torso, tail (5 or fewer gold
  mentions). A single macro number hides tail collapse; the band table shows it.
- **Bootstrap 95% intervals** over notes, and paired bootstrap for model to
  model deltas. A difference counts as real only when the paired interval
  excludes zero.
- **Evidence overlap** against MDACE's gold spans, which is what a coder
  reviewing a suggestion actually depends on.
- **Operational**: latency, truncation rate, error rate, token usage.

Truncation is never silently an empty answer. Adapters raise a typed `Truncated`
error and the runner records it separately, because "the model ran out of
tokens" and "no codes apply" are opposite findings.

**An empty prediction hides four different things**, and only one of them is a
fact about the model. It may be the model genuinely declining to code. It may be
the answer arriving in a channel the adapter did not read, which cost 124 of 312
notes on Gemma 4 before `answer_text` fixed it. It may be the answer being
unparseable because the model quoted clinical text containing quote marks into
JSON without escaping them, which `salvage_codes` now recovers. Or it may be
nothing at all: re-asking the empty notes across every run moved several scores
by two or three points without anything having changed, so a share of them are
simply the model answering differently on the day.

A note that came back empty having spent thousands of tokens is the signature of
the second or third. `scripts/muse_cpt_probe.py` and
`scripts/groq_silence_probe.py` tell them apart, and either is worth running
before believing a low recall. Rerun with `recompute_empty` to separate the
fourth from the rest, since it regenerates exactly the empty notes and leaves
every priced answer alone.

## Guard rails

```bash
bash coding_bench/scripts/install_hooks.sh                        # local pre-commit
python coding_bench/scripts/check_restricted.py --diff origin/main  # what CI runs
```

The scan rejects any added file over 1 MB, any parquet or pickle whatever its
name, and any file carrying two or more MIMIC de-identification markers of the
`[** ... **]` form. The marker scan is the one that matters, because it catches
note text in any container, not just the shape of the mistake that happened
before.

`data/gold/` and every parquet are gitignored. Public CI touches Tier 0 and
Tier 1 only.

## Copyright

CPT descriptors are AMA copyrighted, so the committed CPT label space is codes
only, with descriptions kept on the restricted volume and loaded at run time.
The Tier 0 smoke set uses CMS style short descriptors. ICD-10-CM is public
domain and ships with descriptions.

## Layout

```
coding_bench/
  data/
    build.py            deterministic MDACE to MIMIC join, manifest verification
    build_remote.py     the same build, run inside Modal against the volumes
    publish_volume.py   upload a local build to the volume
    manifests/          Tier 1, committed
    labels/             Tier 1, committed
    smoke/              Tier 0, committed, generated by build_smoke.py
  bench/
    loaders.py          tier aware loading, checksum verification
    metrics.py          every metric, unit tested against hand computed cases
    runner.py           the evaluation loop and the run record writer
    cache.py            the prediction cache and its key
    rescore.py          recompute metrics from stored predictions, no inference
    reporting.py        LEADERBOARD.md and the root README summary block
  approaches/
    base.py             the Predictor protocol, Truncated, span location
    retrieval.py        BGE embedding baseline, CPU only
    embed_match.py      embedding nearest neighbour over code descriptions
    entity_match.py     entity extraction then string match
    llm.py              direct prompting with quoted evidence
    retr_llm.py         retrieval shortlist, then the model selects
  adapters/
    api_groq.py         Qwen3.6 27B and GPT-OSS 120B over HTTP
    api_anthropic.py    Claude Sonnet 5 over HTTP
    modal_muse.py       Muse Glimmer 30B, bf16 transformers, Modal H100
    modal_muse_gguf.py  Muse Glimmer 30B, dynamic K-quant, llama.cpp
    modal_gemma4_gguf.py  Gemma 4 26B-A4B, QAT 4 bit, llama.cpp
    modal_qwen38_vllm.py  Qwen3.8 27B FP8, vLLM; the bf16 arm is unqueued
  scripts/              guard rails, and probes for why a model went quiet
  tests/                no network, no GPU, no restricted data
  run_chain*.py         detached multi step runs that outlive their client
  eval_remote.py        Modal entrypoint for Tier 2 evaluation
```

The `run_chain*.py` files are one per campaign rather than one parameterised
runner. Each carries the reasoning for its own configuration in its docstring,
which is the part worth keeping once the run is over.

## CI

`.github/workflows/coding-bench.yml` is one workflow with two jobs, and the
second waits on the first:

- **`checks`** runs on every push and pull request that touches this package:
  the Tier 0 suite, `rescore --check`, `reporting --check`, and the restricted
  data scan over the changed files here. It has no Modal token and no provider
  key, so it structurally cannot run inference and cannot spend anything.
- **`evaluate`** runs the hosted leaderboard set against Modal, regenerates the
  board and the README summary, and commits the run records back to the branch.
  Never on a pull request: a fork cannot see the secrets, and paying for a
  stranger's branch is not something this repo should do by default.

They were two files once and ran in parallel, so a commit that broke the tests
still bought inference: the suite went red at minute one while the evaluation
spent twenty dollars discovering the same thing at minute thirty. `needs: checks`
closed that.

GPU models are not in the pushed set. Muse and Gemma take hours and run as
detached chains that outlive any client, so they are launched by hand and banked
afterwards with a dispatch of `action: collect`, which pulls any finished record
off the volume into git. Qwen3.8 is the exception: `launch_gpu` deploys, smokes,
spawns and banks it from the commit that adds it, gated on the change actually
touching its adapter or chain and on no records being banked already. See
[Qwen3.8 27B, and why FP8](#qwen38-27b-and-why-fp8). A change to the prompt or to the gold manifests
invalidates every model's cache, so `evaluate` refuses to do that from a push
and asks for a dispatch with `confirm_full_rerun`.

**Neither job re-runs inference that has already been paid for.** A note is
regenerated only when something that can change its answer changed: the gold
manifest checksum, the model, the adapter, the prompt, the offered candidates or
the run parameters. Everything else is served from the prediction cache and from
the committed run records. Two deliberate exceptions, both narrow and both
recorded in the run: an adapter equivalence, which lets a fix that provably
cannot alter an answer keep the old one, and `recompute_empty`, which pays to
regenerate the notes that came back empty and nothing else.
