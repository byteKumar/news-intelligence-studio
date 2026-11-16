# News Intelligence Studio (Team 116)

Single source guide for setup, usage, and system explanation.

---

## 1) What this project does

This is an end-to-end NLP project for news analysis. Given a news article, it can:

- classify whether the article looks `FAKE` or `REAL`
- infer the closest topic/theme
- generate an extractive summary
- run claim-level fact verification using retrieved evidence

Outputs are saved under `results/`.

---

## 2) Project structure

- `demo.py` - standalone terminal runner for training + analysis demo
- `streamlit_app.py` - optional Streamlit UI
- `notebooks/demo.ipynb` - notebook demonstration
- `src/` - data loading, preprocessing, training, and analysis pipeline
- `fact_verify/` - claim extraction, evidence retrieval, and verdict logic
- `data/raw/` - required input datasets
- `data/processed/` - generated intermediate files (cached corpus, previews)
- `models/` - trained model artifacts
- `results/` - metrics, predictions, and figures
- `tests/` - unit tests

---

## 3) Required dataset files

Place these files exactly at the listed paths before retraining:

```text
data/raw/kaggle_fake_real/Fake.csv
data/raw/kaggle_fake_real/True.csv
data/raw/fakenewsnet/BuzzFeed_fake_news_content.csv
data/raw/fakenewsnet/BuzzFeed_real_news_content.csv
data/raw/fakenewsnet/PolitiFact_fake_news_content.csv
data/raw/fakenewsnet/PolitiFact_real_news_content.csv
data/raw/news_category/News_Category_Dataset_v3.json
```

Dataset roles:

- `kaggle_fake_real/*`: primary fake-vs-real signal
- `fakenewsnet/*`: additional fake/real source styles
- `news_category/*`: topic modeling and wider text coverage

---

## 4) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use Conda, create/activate your environment first, then run `pip install -r requirements.txt`.

---

## 5) How to run

### A. Standalone run (recommended quick demo)

```bash
python demo.py
```

What it does:

1. loads data from `data/raw/`
2. trains/selects classifier and topic model
3. saves artifacts to `models/`
4. runs analysis on sample/custom text
5. runs fact verification
6. writes outputs into `results/`

### B. Streamlit app (interactive)

```bash
streamlit run streamlit_app.py
```

Use tabs to train models, analyze article text, run verification, and inspect outputs.

### C. Tests

```bash
cd ~/Downloads/team_116
conda activate py312_ds
pip install -r requirements.txt
pytest -q
```

---

## 6) How article analysis works (FAKE/REAL + topic + summary)

Main flow (from `src/`):

1. `src/data_loader.py` reads and merges datasets.
2. `src/train_pipeline.py` trains and compares multiple models.
   - classifier options include logistic regression, linear SVC, Naive Bayes, SGD
   - topic options include LDA, NMF, KMeans
3. best artifacts are saved in `models/`.
4. `src/pipeline.py` performs inference:
   - fake/real prediction
   - topic inference
   - extractive summary

Classifier selection policy:

- highest validation macro F1
- then weighted F1
- then accuracy

Topic model selection:

- highest quality score (coherence/diversity oriented)

---

## 7) How fact verification works

Fact verification is implemented in `fact_verify/` as a baseline retrieval pipeline:

1. **Claim extraction** (`claim_extractor.py`)
   - splits article into sentences
   - scores claim-like sentences
   - keeps top-k claims

2. **Evidence corpus build/load** (`loaders.py`)
   - builds corpus from raw datasets
   - caches to `data/processed/evidence_corpus_from_raw.csv`
   - each record has `doc_id`, `title`, `passage`, `source`

3. **Evidence retrieval** (`retriever.py`)
   - TF-IDF vectorization (unigrams + bigrams)
   - cosine similarity between claim and corpus documents
   - returns top-k evidence hits

4. **Verdict generation** (`verifier.py`)
   - computes lexical overlap + retrieval score
   - checks negation mismatch signals
   - outputs one label per claim:
     - `SUPPORTED`
     - `REFUTED`
     - `NOT ENOUGH EVIDENCE`
   - includes confidence and rationale

5. **End-to-end orchestration** (`fact_verify/pipeline.py`)
   - returns summary + per-claim verdict + supporting evidence

Important note:

- This is FEVER-style in structure (claim, evidence, verdict), but not a large neural FEVER benchmark model.

---

## 8) Difference: fake/real classification vs fact verification

- Fake/real classification:
  - one verdict for whole article (`FAKE`/`REAL`)
- Fact verification:
  - verdict per claim (`SUPPORTED`/`REFUTED`/`NOT ENOUGH EVIDENCE`)
  - always tied to retrieved evidence snippets

Both modules complement each other:

- classifier gives fast article-level credibility signal
- verifier gives claim-level explainability

---

## 9) Typical outputs

After running, expect files in `results/` such as:

- benchmark tables and metrics
- prediction samples
- figures under `results/figures/`
- custom-article output JSON

`models/` will contain trained artifacts used by analysis and UI.

---

## 10) Common issues and fixes

- Missing raw files -> training/verification corpus build fails.
  - Fix: verify paths in Section 3.
- Artifacts missing in app -> analyze tab may not run.
  - Fix: train first, or run `python demo.py` once.
- Weak verification confidence -> evidence coverage may be low.
  - Fix: ensure complete raw datasets and retry with stronger claims/text.

---
