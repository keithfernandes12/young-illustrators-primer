# Model card: read-aloud ASR (speech, stage 8)

**Proposed, pending team approval.** Numbers are from 14 Nov 2026 runs on a MacBook (Apple MPS)
against real Speechocean762 audio, 500 test clips (280 children aged 6-8, 220 adults). MLflow run ids
are in `.local-data/speech/final/selection.json` on the owner's machine. **No candidate meets the
0.25 WER target**, so `train.py` selected none; `distil-small-en-guarded` is the team's choice as the
best available, recorded here and not as a pass.

| Field | Value |
|---|---|
| Stage | speech: read-aloud ASR and misread detection |
| Purpose | Listen to a child read a page and report the words they misread or skipped, so the next story can adapt. Optional, switched on by the parent (scoping §5.4) |
| Owner | Kush Patel (KAN-44) |
| Candidates | `configs/speech_baseline.yaml`: Whisper tiny/base/small (English), distil-small, large-v3-turbo, each with decoding variants; wav2vec2-base-960h (CTC) |
| Baseline (frozen configuration) | `whisper-small-en`: `openai/whisper-small.en`, greedy, no prompt. Revision: _pin a commit sha before registration_ |
| Selected | `distil-small-en-guarded`: `distil-whisper/distil-small.en`, greedy, no prompt, at most 6 tokens per second of audio, no repeated 3-token runs. Revision: _pin before registration_ |
| Evaluation data | Speechocean762 test split (`hf:mispeech/speechocean762`), CC BY 4.0, read in full only (completeness = 10), with synthetic misreads. Revision: _pin before registration_ |
| What is registered | The configuration (`recognizer.json`), not the weights; weights come from Hugging Face at the pinned revision |

## Metrics

The shared contract has no ASR key, so nothing here is gated by `golden-gate`; `register.py` checks
the WER target itself.

| Metric | Meaning | Target | Baseline (`whisper-small-en`) | Chosen (`distil-small-en-guarded`) |
|---|---|---|---|---|
| `asr_wer` | Word error rate against what was actually said | ≤ 0.25 (scoping §7) | 0.476 | 0.335 |
| `asr_misread_precision` | Of the words flagged as misread, the share that were | — | 0.346 | 0.342 |
| `asr_misread_recall` | Of the words misread, the share flagged | — | 0.997 | 0.997 |
| `asr_misread_f1` | What selection ranks by | — | 0.514 | 0.509 |
| `asr_runaway_rate` | Share of clips where the model invented far more words than were said | 0 | 0.002 | 0.000 |
| `asr_rtf` | Transcription time over audio time, on the machine it ran on | ≤ 0.5 | 0.073 | 0.058 |

wav2vec2-base-960h on the same clips: WER 0.572, misread F1 0.386; ruled out.

**Why distil-small-en-guarded although its F1 ties the baseline.** Misread F1 is level (0.509 vs
0.514, inside what 500 clips can separate). The choice rests on the other numbers: WER on children
0.540 against 0.892, no runaway loops, and 20% faster.

**Why selection ranks by misread F1 and not by WER.** Whisper's language model tends to output the
word that *should* have been read, which lowers WER and hides exactly the errors this stage exists
to find. Prompting Whisper with the page text makes this stronger. The `-prompted` candidates
measure that trade-off directly.

**Whisper hallucinates on young children.** On one 4 s clip of a 6-year-old reading "Lilly likes
biscuit", unguarded Whisper small wrote "ni" 443 times, which alone took the 50-clip WER from about
0.29 to 1.93. The guards (`max_tokens_per_second`, `no_repeat_ngram_size`) cut it to 0.34;
distil-small-en never looped. Runaways are reported as `asr_runaway_rate` so they cannot hide in WER.

**Earlier 500-clip run of the decoding variants, unguarded** (WER / misread F1): small 0.476 / 0.514, small beam 5
0.470 / 0.532, small prompted 0.347 / 0.440, base 0.384 / 0.479, tiny 0.515 / 0.438. Prompting lowers
WER and lowers F1: it hides misreads. These WERs include runaways.

**How misreads are labelled.** A correctly read clip becomes a labelled misread by changing the page
text, not the audio (`data.with_synthetic_misreads`). This measures whether the pipeline catches a
reader saying a different word than the page. It does **not** measure partial or mispronounced
words, which real children produce and synthetic labels do not. Because the substituted page word
is never in the audio, recall is near 1.0 for every model; precision (false alarms) is what
separates them.

## Slices

`age_band` (3-5, 6-8, 9-12, 13-17, adult) and `gender` from Speechocean762's speaker metadata. A
slice under 100 clips is reported but not judged (`primer.data.bias.SMALL_SLICE`).

| Slice | n | Baseline `asr_wer` | Chosen `asr_wer` |
|---|---|---|---|
| age 6-8 | 280 | 0.892 | 0.540 |
| adult | 220 | 0.152 | 0.175 |
| female | 300 | 0.240 | 0.265 |
| male | 200 | 0.926 | 0.469 |

**Age is the disparity.** The chosen model is 3x worse on 6-8-year-olds than on adults (gap 0.365),
down from 6x for the baseline (gap 0.741). Mitigations: the runaway guards (in place); lenient
alignment so near-misses are not marked wrong (in place); evaluate and, if needed, fine-tune on
children's speech once MyST arrives (not done). Read-aloud stays optional and parent-enabled.

**The gender gap is mostly age.** Of the 200 male clips, 160 are children aged 6-8 (80%); of the 300
female clips, 120 are (40%). Boys' higher WER follows from that, so it is not reported as a gender
effect. A fair gender comparison needs WER by gender within each age band, which was not run.

**Accent cannot be measured on this data**: every Speechocean762 speaker's first language is
Mandarin. That is the slice scoping §9 worries about most; it needs MyST or Common Voice.

## Sensitivity

On the chosen candidate (`train.py --sensitivity-on distil-small-en-guarded`): white noise at 20, 10 and 5 dB SNR, and speed ×0.9 and ×1.1
(`sensitivity.py`). Decoding sensitivity (beam 1 vs 5, page prompt on vs off) comes from the
candidate grid.

| Perturbation | `asr_wer` | change vs clean |
|---|---|---|
| _not yet run: no candidate was selected, so the run skipped it_ | | |

## Mitigations and trade-offs

- **Lenient alignment.** Case, punctuation and near-misses (difflib ratio ≥ 0.8) count as correct, so
  accents and young speech are not marked wrong. 0.8 is a guess until measured on children.
- **Inserted words are never counted against the child.**
- **Runaway guards.** Output capped at 6 tokens per second of audio and no repeated 3-token runs,
  so an unclear clip cannot flag a whole page.
- **Privacy.** Audio is processed in memory and discarded; no waveform, transcript or child
  identifier is logged or sent to MLflow. Only aggregate metrics and settings are.

## Known limitations

- Not children reading stories: short prompts, Mandarin-L1 speakers. MyST access is KAN-101.
- Synthetic misreads only; real misreading (partial words, sounding out) is not covered.
- No phoneme-level check yet (KAN-44's proposal); wav2vec2 is the closest candidate to it.
- Speechocean762 is read from Hugging Face directly, not through `primer-data`.
