# The illustration area. One image runs locally and on Vertex AI, which is what makes a registered
# run reproducible (docs/adrs/where-model-training-happens.md).
#
# Stub: pick a CUDA base that matches the driver on the member machines, install the area's extras,
# and keep the source mounted at /work so the same image serves local iteration.
FROM python:3.11-slim

WORKDIR /work
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]" || true
COPY src ./src

ENV HF_HOME=/hf
ENTRYPOINT ["python", "-m", "primer.illustration.train"]
