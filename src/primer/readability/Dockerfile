# The readability area. One image runs locally and on Vertex AI, which is what makes a registered
# run reproducible (docs/adrs/where-model-training-happens.md).
#
# Stub: pick a CUDA base that matches the driver on the member machines and add the area's own
# dependencies. The layout below is not a stub -- it is the part that has to be right.
FROM python:3.11-slim

WORKDIR /work

# Everything the editable install reads must be present BEFORE pip runs: pyproject.toml names
# README.md as the project readme, and the package lives under src/. Installing first and copying
# afterwards produced an image with no `primer` package in it.
COPY pyproject.toml README.md ./
COPY src ./src

# No `|| true`. A failed install must stop the build rather than leave an image that looks fine
# and has nothing installed.
RUN pip install --no-cache-dir -e ".[dev]"

# Keep the model cache on a mount: these downloads are gigabytes, and on Windows hosts a long
# default path hits the 260-character limit (docs/engineering/model-containers-on-windows.md).
ENV HF_HOME=/hf

ENTRYPOINT ["python", "-m", "primer.readability.train"]
