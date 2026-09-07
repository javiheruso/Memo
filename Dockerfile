FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy the project metadata and source, then install the package (its
# dependencies are declared in pyproject.toml; templates ship in the wheel).
COPY pyproject.toml README.md alembic.ini ./
COPY app ./app
COPY migrations ./migrations

RUN pip install --no-cache-dir .

EXPOSE 8000

# Serve the API/UI. Bind to $PORT when present (Railway injects it), falling
# back to 8000 for local Docker Compose. Apply migrations as a deploy step:
#   docker compose run --rm app alembic upgrade head
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
