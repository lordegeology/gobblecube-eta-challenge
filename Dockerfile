# ETA Challenge Submission Dockerfile
#
# Build:
#   docker build -t my-eta .
# Test locally:
#   docker run --rm -v $(pwd)/data:/work my-eta /work/dev.parquet /work/preds.csv

FROM python:3.11-slim

WORKDIR /app

# libgomp1 required by LightGBM at runtime on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Inference-only requirements (no geopandas — only needed at training time)
COPY requirements.inference.txt .
RUN pip install --no-cache-dir -r requirements.inference.txt

# features.py must be present — predict.py imports it at load time
COPY features.py predict.py grade.py ./

# model.pkl contains: LightGBM booster + lookup tables + centroids + weather
# Everything needed for inference is baked in — no network calls at runtime
COPY model.pkl ./

# Grader invokes: python grade.py <input.parquet> <output.csv>
ENTRYPOINT ["python", "grade.py"]
