#!/bin/sh
# Repository-owned container entrypoint.
#
# The default command ("serve") applies pending Alembic migrations when
# MEMO_AUTO_MIGRATE is enabled and then replaces this shell with Uvicorn. The
# spawned application process inherits MEMO_AUTO_MIGRATE=false, so the FastAPI
# lifespan does not apply the same migrations a second time.
#
# Any other explicit command (for example `docker compose run --rm app alembic
# upgrade head`) is executed as-is and bypasses the migration step.
set -eu

run_migrations() {
    echo "memo: applying database migrations" >&2
    if ! alembic upgrade head; then
        echo "memo: database migrations failed; refusing to start" >&2
        exit 1
    fi
    MEMO_AUTO_MIGRATE=false
    export MEMO_AUTO_MIGRATE
}

if [ "$#" -eq 0 ] || { [ "$#" -eq 1 ] && [ "$1" = "serve" ]; }; then
    case "$(printf '%s' "${MEMO_AUTO_MIGRATE:-false}" | tr '[:upper:]' '[:lower:]')" in
        1 | true | yes | on)
            run_migrations
            ;;
    esac
    # Render injects $PORT. Local Docker Compose falls back to 8000.
    exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
fi

exec "$@"
