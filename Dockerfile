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

# Serve the API/UI. Apply migrations as a deploy step, e.g.:
#   docker compose run --rm app alembic upgrade head
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
