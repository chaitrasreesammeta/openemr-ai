# OpenEMR-AI

Open-source ambient clinical documentation, clinical decision support validation, and automated coding for [OpenEMR](https://www.open-emr.org/). This branch (`main`) hosts the deployment code — a SMART on FHIR application that runs inside OpenEMR and writes SOAP notes back to the chart through FHIR R4. The research experiments that benchmark and validate each component live on the [`experiments`](../../tree/experiments) branch.

## Deployment architecture

![OpenEMR-AI SMART Ambient Listening System Architecture](docs/architecture.png)

The app is registered as an OAuth2 client inside OpenEMR and launches from the patient dashboard. The browser captures 16 kHz WebM/Opus audio via the MediaStream Recording API and streams it through a FastAPI gateway to serverless GPU ASR on Modal (production: Voxtral Mini 3B on an L40S). The transcript and the patient's FHIR R4 resources (vitals, labs, conditions, medications) are sent to the summarization service, which uses retrieval-augmented generation: GPT-OSS-20B extracts the primary clinical condition, the condition is matched against 1,000 MIMIC-IV-derived SOAP schemas in ChromaDB (all-MiniLM-L6-v2 embeddings), and the top-2 schemas anchor the final SOAP note generation on GPT-OSS-120B (Groq). The completed note is written back to OpenEMR as a FHIR `DocumentReference` / clinical-note resource. All credential handling is stateless and per-request; the container auto-scales to zero after 120 s of inactivity. Measured per-encounter cost is approximately **$0.02**, a 98–99% reduction versus commercial ambient-scribe alternatives.

## Repository layout

```
smart-ambient-listening/
├── frontend/                 # SMART on FHIR app — launches from OpenEMR, handles OAuth2, captures audio
├── transcription-service/    # Speech-to-text API gateway, calls Modal GPU for ASR inference
├── rag-text-summarization/   # SOAP note generation — ChromaDB schema retrieval, LLM summarization
├── chromadb/                 # MIMIC-IV SOAP-schema vector store bootstrap
└── config/                   # systemd units, Apache reverse-proxy, .env.example
```

Setup and quick-start: [`smart-ambient-listening/README.md`](smart-ambient-listening/README.md).

## Research experiments ([`experiments`](../../tree/experiments) branch)

The `experiments` branch contains the benchmarks, ablations, and statistical analyses that informed model selection for the deployed app. Four experiment tracks:

### 1. ASR word-error-rate benchmark
13 ASR models evaluated on **335 clinical conversations + 380 short-form utterances** across four datasets (institutional IU recordings, Kaggle medical dictation, PriMock57, Fareez OSCE).

- **Voxtral Mini 3B** (Mistral) is the production pick: best on institutional recordings (**5.71% WER**) and on the 272-file Fareez OSCE set (**14.50% WER**).
- **Parakeet TDT 1.1B / 0.6B v2** (NVIDIA) are the low-latency fallbacks (~1–3 s/file) and win on PriMock57 (17.06%).
- **Medical-dictation ASR does not transfer to conversations**: MedASR 44–65% WER across conversational datasets despite being pretrained for medical speech. Production WER will be materially higher once diarization is integrated (PriMock57 mean DER 0.21).

See [`openemr_whisper_wer/`](../../tree/experiments/openemr_whisper_wer) on the experiments branch.

### 2. RAG SOAP-note generation + blinded clinician evaluation
Seven open-weight LLMs (4B–120B) benchmarked for retrieval-augmented SOAP generation, anchored on 1,000 MIMIC-IV disease schemas. Four top models (**GPT-OSS-120B, GPT-OSS-20B, Qwen3-32B, MedGemma-4B**) generated 160 summaries on 40 Fareez OSCE conversations, which three board-eligible physician fellows (GI, IR, EM) rated blindly across six PDQI-9/QNOTE dimensions — **480 ratings** analyzed with Gwet's AC2 and Friedman/Wilcoxon tests.

- **GPT-OSS-120B won the composite** (3.74/5.0), followed by GPT-OSS-20B (3.71), Qwen3-32B (3.61), and MedGemma-4B (2.95). Five of six dimensions differed significantly (p ≤ 0.002).
- **Embedding-based metrics fail to discriminate the models clinicians clearly separate**: BERTScore F1 was 0.800–0.803 for all four models while clinician composites spanned 2.95–3.74. MedCAT entity recall recovers the clinically meaningful separation that embedding similarity misses — with direct implications for ambient-AI procurement and post-market surveillance.
- **Disabling retrieval** produced consistent degradation across all seven LLMs on entity- and structure-sensitive metrics (MedCAT −0.020 to −0.060, ROUGE-L −0.005 to −0.050, BERTScore −0.020 to −0.060).

See [`rag_models/`](../../tree/experiments/rag_models) on the experiments branch.

### 3. ELM / CDS rule validation
16 LLMs evaluated on whether they can validate Expression Logical Model (ELM) JSON — the compiled form of CQL clinical decision rules — against the Clinical Practice Guidelines they are meant to implement. Benchmark: **41 hand-curated, ground-truth-annotated ELM artifacts** (16 valid, 25 invalid: 13 parametric + 12 semantic errors) with 5 trials per model at T=0.1.

- **Qwen3-32B leads overall at 90.7% ± 2.0%** accuracy (F1 = 0.88), the only model balancing high sensitivity (0.88) and specificity (0.93). GPT-OSS-120B (85.9%), Qwen3.5-35B-A3B (85.4%), and Llama-3.3-70B (83.4%) follow.
- The **ELM Simplifier (deterministic key-value extractor) behaves as a quantization-compensation mechanism**: at Q4_K_M deployment precision, dense Gemma 4 31B loses 14.6 pp without it (82.9→68.3), while the 4B-active MoE Gemma 4 26B A4B is unaffected (+2.0 pp). CPG reference is the dominant scaffold — removing it costs 14–33 pp across frontier models.
- Models **≤4B uniformly fall below the 48.4% naive baseline**, establishing a capability threshold around 20B total / 4B active parameters for ELM validation. Medical pretraining (MedGemma) shows no consistent advantage.

See [`cdr_elmjson_validator/`](../../tree/experiments/cdr_elmjson_validator) on the experiments branch.

### 4. Automated CPT and ICD-10 coding
Multi-label coding from clinical notes on MDACE Profee: **312 MIMIC-III notes over 61 CPT codes, and 578 notes over 673 ICD-10-CM codes**. Hosted models (Claude Sonnet 5, GPT-OSS-120B, Qwen3.6-27B) against self-hosted quantised models (Muse Glimmer 30B, Gemma 4 26B-A4B) on the same gold labels, prompt and candidate lists. Every number carries a bootstrap 95% interval, and a run that fails more than 5% of its notes is quarantined rather than scored.

- **The candidate space is most of the result.** Offering a model only the note's correct codes pins precision at 1.000 by construction, so that score is a recall ceiling and not a deployment estimate. Qwen3.6-27B scores **0.815 micro F1 on ICD-10 at gold candidates and 0.528 against the full 673-code catalogue**. Published coding numbers that do not say which condition they measured are not comparable with either.
- **The long tail is where the collapse happens.** Against the full catalogue the best model reaches 0.755 F1 on the ten most frequent ICD-10 codes and 0.344 on codes with five or fewer gold mentions. A single macro F1 hides that gap.
- **Self-hosting costs about ten points.** Muse Glimmer 30B at roughly 4-bit reaches 0.423 on ICD-10 full against 0.528 for the best hosted model, with no note text leaving our own GPU, though at a 4.0% note-failure rate that sits just inside the quarantine ceiling. That is the privacy-versus-accuracy trade in the form a deployment decision actually takes.

Ranked tables, frequency bands, paired significance tests and quarantined runs: [`coding_bench/results/LEADERBOARD.md`](../../blob/experiments/coding_bench/results/LEADERBOARD.md). The board is generated from committed run records and CI fails if it is stale.

An earlier `automated_coding/` package covered the CPT half with traditional NLP baselines. It was retired once `coding_bench` covered the same ground with a gold-set contract, a prediction cache and per-run provenance, and its numbers are not comparable with the board above. It remains in git history on the experiments branch.

## Citation

If you use this framework, please cite:

> Purkayastha S, Pulavarthy LP, Kodali V, Uddin SA, Jones J, Fulton CR. *Open-source ambient clinical documentation with blinded physician evaluation: an OpenEMR-integrated framework.* 2026.

BibTeX:

```bibtex
@article{purkayastha2026openemr,
  title  = {Open-source ambient clinical documentation with blinded physician evaluation: an OpenEMR-integrated framework},
  author = {Purkayastha, Saptarshi and Pulavarthy, Lalitha Pranathi and Kodali, Vishnupriya and Uddin, Sheikh Azam and Jones, Josette and Fulton, Cathy R.},
  year   = {2026},
  url    = {https://github.com/iupui-soic/openemr-ai}
}
```

**Corresponding author:** Saptarshi Purkayastha, PhD — Department of Biomedical Engineering and Informatics, Luddy School of Informatics, Computing, and Engineering, Indiana University, Indianapolis; Regenstrief Institute, Indianapolis. `saptpurk@iu.edu`.

**Funding:** STEM Education Innovation and Research Institute (SEIRI) Seed Grant, Indiana University Indianapolis — *Training Tomorrow's Health AI Leaders: Cross-Disciplinary EMR Innovation for Experiential Learning* (2025–2027, PI: S. Purkayastha).

## License

MPL-2.0 — see [LICENSE](LICENSE).
