# Speech: read-aloud ASR (stage 8)

The speech area listens to a child read a page aloud and reports which words they misread or skipped, so the next story can adapt. It is the optional read-aloud mode in `docs/project_scoping.md` (section 5.4, P5), switched on or off by the parent. The same area also holds the text-to-speech narration interface (`SpeechSynthesizer`), which another member owns.

**Owner:** Kush Patel. **Jira:** [KAN-44](https://northeastern-team-12.atlassian.net/browse/KAN-44), with one subtask per step below. **Status:** all code for steps 0 to 6 is written and tested on fakes and synthetic arrays. **Nothing has run on real audio**, so there is no word error rate yet. Nothing is committed.

## What exists today

| Piece | File | State |
|---|---|---|
| `SpeechRecognizer` Protocol: `transcribe(samples, sample_rate, prompt=None)`; `Transcript`, `ReadingResult`, `FakeSpeechRecognizer` | `interface.py` | Done |
| `align(page_index, page_text, transcript)`: labels each page word correct, misread, skipped or inserted | `alignment.py` | Done |
| `WhisperRecognizer`, `Wav2Vec2Recognizer`, `RecognizerConfig`: weights load on first `transcribe` | `model.py` | Written, **not run** |
| Candidate grid: 9 configurations and the selection rule | `configs/speech_baseline.yaml` | Written |
| Speechocean762 loader (fallback for MyST), split by speaker, plus synthetic misread and skip labels | `data.py` | Written, **not run** against Hugging Face |
| WER, misread precision, recall and F1, real-time factor, per-slice metrics, the contract file | `evaluate.py` | Done |
| Age-band and gender slices, the 100-clip floor, gaps, a bias report | `slices.py` | Done |
| Added noise and speed perturbation on the winner | `sensitivity.py` | Done |
| Run every candidate, one MLflow run each, select the best, write the comparison | `train.py` | Done; `--smoke` runs end to end |
| Register the selected configuration, refuse anything unselected, off target or unpinned | `register.py` | Written; MLflow calls **not run** |
| CPU image | `Dockerfile` | Written, not built |
| Model card, with every number marked unmeasured | `CARD.md` | Skeleton |
| Tests (74) | `tests/speech/` | Pass |

## Run the comparison (on your machine)

```bash
pip install -e ".[dev,speech]"
python -m primer.speech.train --smoke --out .local-data/speech/smoke          # no download, checks the path
python -m primer.speech.train --no-mlflow --max-clips 50 --only whisper-small-en --out .local-data/speech/try
python -m primer.speech.train --out .local-data/speech/run1                   # every candidate, 500 clips, MLflow
```

The first real run downloads Speechocean762 and each checkpoint into the Hugging Face cache. `--only NAME` (repeatable) runs a subset; the baseline must be run first or be in the same `--out`. Without `MLFLOW_TRACKING_URI`, MLflow writes to `./mlruns` (git-ignored); `mlflow ui` shows it. Exit code 1 means no candidate met `max_wer` and `max_rtf`.

Output, per candidate: `metrics.json`, `bias_report.md`, `recognizer.json`; overall: `comparison.md`, `comparison.png`, `selection.json`; on the winner: `sensitivity.md`. Copy the numbers into `CARD.md`.

Then register the winner, after pinning `revision` in the config to a commit sha and re-running it:

```bash
python -m primer.speech.register --run-id <run id from selection.json>
```

## How the alignment decides

```python
from primer.speech.alignment import align
from primer.speech.interface import FakeSpeechRecognizer

transcript = FakeSpeechRecognizer("the dog sat").transcribe([])  # samples are ignored by the fake
result = align(0, "The cat sat.", transcript)
result.misread_words  # ("cat",)
```

- Case and punctuation are ignored; apostrophes inside words are kept.
- Two words with a difflib similarity of at least `NEAR_MISS = 0.8` count as the same word, so "dragon" heard as "dragin" is correct. This is deliberately lenient so accents and young speech are not marked wrong (scoping section 9). **0.8 is a guess** until it is measured on real children's speech.
- A word on the page the child did not say is `skipped` and counts as a misread word.
- A word the child added is `inserted` and is never held against them.

## The plan, one subtask per step

The steps come from `docs/engineering/model-development.md`. ASR is a pretrained stage, so it skips training (step 5 there) and does not fine-tune unless the baseline is beaten and time allows.

| Jira | Step | What |
|---|---|---|
| KAN-95 | S0 | Interface, types, fake, alignment. **Done** (commit `a39935c`, branch `feat/KAN-95-asr-interface`) |
| KAN-96 | S1 | Whisper baseline as a named frozen configuration (`whisper-small-en`), word error rate on a hold-out. Code written; **needs a real run** |
| KAN-97 | S2 | `evaluate.py` writes the shared `metrics.json` shape. **Done** on fakes, not committed |
| KAN-98 | S3 | Bias slices (age band, gender; accent needs other data) and a report. Code done |
| KAN-99 | S4 | Sensitivity analysis (added noise, speed perturbation). Code done |
| KAN-100 | S5 | MLflow tracking, registry, container, complete `CARD.md`. Code written; card numbers wait on the run |
| KAN-101 | S6 | Get MyST access and a fallback evaluation set. Fallback: Speechocean762 in `data.py`. MyST request still to send |

## Known gaps and risks

- **Fallback data only.** MyST (children's speech) is access-by-request from Boulder Learning and has no ingestion pipeline. Until it arrives, Speechocean762 stands in: about half children, but every speaker is Mandarin-L1 and the text is short prompts. It is read from Hugging Face directly, not through `primer-data`, which is a gap against the repo's data rules.
- **Synthetic misreads.** Misread labels come from editing the page, not from real misreading; partial and mispronounced words are not covered.
- **No ASR metric in the shared contract.** `primer.evaluation.contract` has no word error rate key, so this stage's numbers go in `overall` of `metrics.json` as `asr_wer`, `asr_misread_precision`, `asr_misread_recall`, `asr_misread_f1`, `asr_rtf`, `asr_clips` and `asr_reference_words`, and in `CARD.md`. Precision or recall is `null` when undefined (nothing flagged, nothing labelled). Adding a gated key means changing `contract.py` and `.claude/skills/golden-gate/thresholds.json`, agreed with the team.
- **Whisper tends to auto-correct** what a child misreads, so a transcript alone can hide errors. Selection ranks by misread F1, not WER, for this reason, and the `-prompted` candidates measure how much the page prompt makes it worse. KAN-44's phoneme-level check is not built.
- **Real runs need a person.** Whisper weights, GPUs and the shared MLflow and registry services are not available to an agent session. Unit tests use the fake and never load weights.
- This branch is stacked on `docs/model-development-blueprint` (KAN-93, not yet merged), which is where the speech skeleton comes from.

## Rules for working here

- Branch `<type>/KAN-<n>-<slug>`; commit `type(scope): summary (KAN-<n>)`; open draft PRs; never merge your own PR. See the root `AGENTS.md`.
- **No child audio, names or recordings** in code, fixtures, logs or commits (COPPA). Audio is processed in memory and discarded; nothing here stores a waveform. Tests use synthetic samples.
- Importing `interface.py` must stay cheap: no weights, no network, no GCS. Weights load lazily in `model.py`.
- A model area never imports `primer.serving`; serving imports the area.
- Split any evaluation data by speaker, never by clip.
- Never write to the shared bucket from an agent session. Use `--data-uri .local-data`.

## Running the checks

```bash
pip install -e ".[dev]"
pytest tests/speech
ruff check .
mypy src
```

## For an AI assistant picking this up

Read, in order: the root `AGENTS.md`, `docs/engineering/model-development.md` (the seven steps and the metrics file shape), `docs/project_scoping.md` sections 5.4, 9 and 11, then this file. All the code for steps 0 to 6 exists and is tested on fakes. What is left needs a person: run `train.py` for real, pin the revisions, fill `CARD.md` from the output, send the MyST request. Do not invent word error rate numbers, and say plainly what was not run. Ask before creating or moving Jira issues.
