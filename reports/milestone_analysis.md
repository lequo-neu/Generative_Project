# IE7615 Generative Project — Milestone Analysis

**Course:** IE7615 Neural Networks/Deep Learning SEC 02, Spring 2026
**Group 8:** Quoc Hung Le, Khoa Tran, Hassan Alfareed

---

## Milestone #1 — Data Pipeline & Environment

**Core objective:** Establish a reproducible data pipeline that extracts visual embeddings via CLIP and tokenizes captions via GPT-2, proving the team can handle multimodal input/output before any model integration.

**Deliverables:** Project proposal (front page, dataset, model, roles), notebook with CLIP embedding + GPT-2 tokenization pipeline, 5 sample test runs, initialized GitHub repo.

**Implicitly tested:** Data engineering, HuggingFace ecosystem fluency, tensor shape reasoning (CLIP 512-d vs GPT-2 768-d), reproducibility discipline (seeds, configs).

**High-score criteria:** Modular code (classes/functions, not monolithic cells), explicit shape assertions, dataset choice justified in writing, sample runs include diverse image types, README with setup instructions.

**M2-M4 prep:** ProjectionHead placeholder defined, embeddings cached to .pt files, Config dataclass extensible for training hyperparameters, consistent logging pattern.

---

## Milestone #2 — Model Integration & Baseline Generation

**Core objective:** Bridge vision-language gap via embedding injection (prefix conditioning or cross-attention), generate first captions, compare decoding strategies.

**Deliverables:** Training log with loss curves, 5-10 images with generated captions, front page + observations summary.

**Implicitly tested:** Prefix conditioning vs cross-attention understanding, freeze/unfreeze discipline, decoding strategy theory (greedy collapse, beam diversity, nucleus stochasticity), gradient flow debugging.

**High-score criteria:** Architectural diagram, side-by-side decoding comparison table, convergence evidence with LR schedule, critical observations (not just "beam is better").

---

## Milestone #3 — Quantitative Evaluation & Ablation

**Core objective:** Rigorously evaluate with NLG metrics and perform controlled ablation studies.

**Deliverables:** Evaluation notebook, tables + plots, draft report (methods & results).

**Implicitly tested:** Correct metric implementation (corpus-level BLEU, not sentence-level), statistical rigor, ablation methodology (one variable at a time), LoRA vs full fine-tuning parameter efficiency understanding.

**High-score criteria:** Metrics on held-out test set, publication-quality plots, quantitative-qualitative connection, LoRA comparison includes param count and wall-clock time.

---

## Milestone #4 — Report, Presentation & Ethics

**Core objective:** Synthesize work into professional research report with ethical reflection.

**Deliverables:** Final PDF (10-15 pages), captions gallery, slides + demo video (15-20 min), complete GitHub repo.

**Implicitly tested:** Academic writing, ethical reasoning (COCO bias, stereotypical descriptions, accessibility), presentation skills, code documentation.

**High-score criteria:** Substantive ethical reflection, gallery with both successes and failures, rehearsed demo within time limit, one-command README setup.
