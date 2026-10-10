# TL;DR - Young Illustrator's Primer

**Team:** Rutvij Surti · Kush Patel · Keith Fernandes · Rishi Parekh · Diya Kaswa · Adit Vakil

---

**What it is:** A child draws characters and scenes. The system turns those drawings into a personalized, illustrated, narrated storybook written at the child's reading level, built around words the parent wants them to learn. Inspired by Y Combinator's *"The Primer"* request for an adaptive AI reading tutor for kids.

**Repo:** https://github.com/keithfernandes12/young-illustrators-primer

**Who it's for**
- **Ages 3–5, *Learn to Read*:** sight words, short pages, narration with word highlighting.
- **Ages 5–8, *Grow Vocabulary*:** richer words, longer stories, glossary, comprehension questions.
- **Parents:** a dashboard to set goals, target words, book length, and turn read-aloud mode on or off.
- **Dev team:** an internal dashboard for model health, usage, cost, and KPIs.

**Problem → Solution**

| Problem | Our solution |
|---|---|
| Generic books don't hold kids' interest | Stories star the child's own drawings |
| Parents can't tell which words their child knows | Target-word selection + tracking of words mastered |
| Practice needs an adult present | TTS narration; read-aloud speech recognition that flags misread words (stretch) |

**MVP (due Nov 15):** drawing → story at the right level with target words → readability + safety gate → illustration (pretrained SD + ControlNet) → narration → web book, both age modes, plus the full MLOps pipeline. **Stretch goals, in order:** read-aloud ASR · learner adaptation · custom illustration style · age-mode extras · full dev dashboard.

**Data (open-licensed only):** StoryWeaver and African Storybook (illustrated books with reading levels), TinyStories, Project Gutenberg, Quick, Draw! and Amateur Drawings (child-style art), CLEAR (readability), and MyST (children's speech). No copyrighted media. Child data follows **COPPA**: parent-owned accounts, consent, minimal storage, deletion on request.

**Models (fine-tuned from Hugging Face):** drawing recognizer · story LLM (LoRA) · readability scorer · illustration style LoRA + ControlNet · safety filter. Pretrained models handle TTS narration and Whisper speech recognition.

**MLOps stack:** Airflow · DVC · MLflow · GitHub Actions CI/CD with an evaluation gate · FastAPI serving · Evidently + Prometheus + Grafana monitoring, on **GCP or AWS** (decision in Week 1).

**Key targets:** ≥ 95% of target words included · reading level within ±1 of target for ≥ 85% of books · zero safety violations on the golden test set · full book in ≤ 2 min · ≤ $0.25 per book · parent satisfaction ≥ 4/5.

**Top risks → mitigations**
- Unsafe content for kids → safety filters at every stage, plus deployment blocked if golden-set safety tests fail.
- Unclear drawings → the child or parent labels each drawing.
- GPU cost and latency → asynchronous page generation, LoRA, spot instances, budget alerts.
- Model drift → drift monitoring triggers retraining.
- Scope creep → MVP first, stretch goals only after; ~1.5 weeks of buffer.
- Child pilot needs IRB → asked in Week 1; fallback is evaluation with adults only (no children).

**Timeline**

| When | What |
|---|---|
| Oct 5 – Oct 25 | Repo, cloud setup, IRB question, data pipelines, dataset v1, baselines |
| Oct 26 – Nov 15 | Fine-tune models, end-to-end pipeline deployed; **MVP complete Nov 15** |
| Nov 16 – Nov 30 | **Buffer week**, then stretch goals and hardening; **code freeze Nov 30** |
| Dec 1 – Dec 14 | QA, safety red-teaming, pilot/evaluation; Dec 12–14 buffer for report + demo |

**Done means (MVP):** a drawing goes in, a complete narrated book comes out on the deployed cloud system; both age modes work; 3 team-trained models are versioned and reproducible; pipelines and monitoring are automated; privacy controls are in place.

*Full details: [project_scoping.md](project_scoping.md)*
