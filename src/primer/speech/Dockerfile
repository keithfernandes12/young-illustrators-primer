# The speech area. One image runs locally and on Vertex AI, which is what makes a registered
# run reproducible (docs/adrs/bounded-local-training-for-agents.md).
#
# CPU image: the selected candidates are small enough to evaluate on CPU. For the large-v3-turbo
# candidates, swap in a CUDA base that matches the driver and drop the CPU torch index below.
FROM python:3.11-slim

WORKDIR /work

# Everything the editable install reads must be present BEFORE pip runs: pyproject.toml names
# README.md as the project readme, and the package lives under src/. Installing first and copying
# afterwards produced an image with no `primer` package in it.
COPY pyproject.toml README.md ./
COPY src ./src

# No `|| true`. A failed install must stop the build rather than leave an image that looks fine
# and has nothing installed.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -e ".[dev,speech]"

# Keep the model cache on a mount: these downloads are gigabytes, and on Windows hosts a long
# default path hits the 260-character limit (docs/engineering/model-containers-on-windows.md).
ENV HF_HOME=/hf

ENTRYPOINT ["python", "-m", "primer.speech.train"]
