FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY photonics.py optics.py pipeline.py ./
COPY models/ ./models/

ENTRYPOINT ["python", "pipeline.py"]
CMD ["--help"]
