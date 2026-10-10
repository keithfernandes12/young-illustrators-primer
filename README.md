# Young Illustrator's Primer

A child draws characters and scenes. The Primer turns those drawings into a personalized, illustrated, narrated storybook written at the child's reading level and built around words their parent wants them to learn.

This is our MLOps course team project, inspired by Y Combinator's *"The Primer"* request for an adaptive AI reading tutor for kids.

> **Status:** Week 1, scoping and setup. Nothing is runnable yet.

## Docs

- **[Project summary (1 page)](docs/project_scoping_tldr.md):** start here.
- **[Full scoping document](docs/project_scoping.md):** data, models, architecture, MLOps pipeline, monitoring, timeline, and risks.
- **[Data pipeline](src/primer/data/README.md):** how to run ingestion and where the data lives in GCS.
- **[Tracker dashboard](dashboards/tracker/README.md):** project progress and data pipeline status.

## Repository layout

```
app/          Web app (child reader + parent dashboard)
configs/      Dataset registry and pipeline configs
dags/         Airflow DAGs
dashboards/   Project tracker and developer monitoring dashboards
data/         DVC-tracked data (never committed to git)
docs/         Scoping and design docs
infra/        Terraform, Docker, deployment configs
notebooks/    EDA and experiments
src/          Model, pipeline, and serving code
tests/        Unit and integration tests
```

## Contributing

- Branch off `main` and open a pull request. At least one teammate reviews each PR before it merges.
- Never commit data, model weights, credentials, or `.env` files.
- Use only open-licensed data, and record its license and attribution.

## Team

Rutvij Surti · Kush Patel · Keith Fernandes · Rishi Parekh · Diya Kaswa · Adit Vakil
