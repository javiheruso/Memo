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

# Repository-owned startup wrapper: when MEMO_AUTO_MIGRATE is enabled it
# applies pending migrations, disables the in-app lifespan migration, and
# replaces itself with Uvicorn. Explicit commands (e.g. `docker compose run
# --rm app alembic upgrade head`) bypass that default.
COPY docker/entrypoint.sh /usr/local/bin/memo-entrypoint
RUN chmod 0755 /usr/local/bin/memo-entrypoint

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/memo-entrypoint"]
# "serve" is handled by the wrapper; it reads $PORT (Render) or falls back to
# 8000 for local Docker Compose. Keep Render's Docker Command empty.
CMD ["serve"]
