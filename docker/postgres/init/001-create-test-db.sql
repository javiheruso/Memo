-- Local development databases for memo.
--
-- Runs once on first container start via docker-entrypoint-initdb.d, connected
-- to the POSTGRES_DB database (memo). It provisions the dedicated test
-- database and enables pgvector in both, so:
--   - `alembic upgrade head` works against `memo` (the baseline migration also
--     enables the extension, this just makes the dependency explicit);
--   - the postgres test profile can run against `memo_test` without waiting for
--     migrations, because conftest creates the schema directly via SQLAlchemy
--     `Base.metadata.create_all`, which does NOT create the vector extension.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE DATABASE memo_test;

\connect memo_test
CREATE EXTENSION IF NOT EXISTS vector;
