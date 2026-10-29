# PostgreSQL operations and Phase 4 migrations

This guide covers the local `postgres` service in [`compose.yaml`](../compose.yaml). Commands below are instructions, **not a record of operations performed**. Work only with a database you are authorized to manage. The database can contain original document bytes, extracted text, and embeddings; protect its credentials and backups accordingly.

## Passwords and the persistent volume

`POSTGRES_USER`, `POSTGRES_DB`, and `POSTGRES_PASSWORD` belong in the local, Git-ignored `.env` file copied from [`.env.example`](../.env.example). Compose passes them to the `postgres` container. `DATABASE_URL` in the same file supplies the Python API and `python -m rag_ai.storage` CLI. Keep the username, database name, host/port, and password in that URL consistent with the actual PostgreSQL role and database. Never commit `.env` or print it with `docker compose config`.

The Compose service mounts `postgres_data` at `/var/lib/postgresql/data`. The PostgreSQL image uses `POSTGRES_PASSWORD` to **initialize** the role when that data directory is empty. Once the volume contains a database, changing `.env` or recreating the container does **not** change the password stored inside PostgreSQL. The container's local Unix-socket connection can use trust authentication, whereas the API connects over TCP and needs the current role password. See the [PostgreSQL image's initialization rules](https://github.com/docker-library/docs/blob/master/postgres/content.md). Never use `docker compose down -v` for password rotation: it removes the named data volume and can destroy the database.

For an authorized local password change, first confirm the service is running with the read-only diagnostic `docker compose ps postgres`. Open an **interactive** console in the existing service (the inner shell expands the container's `POSTGRES_USER` and `POSTGRES_DB`):

```text
docker compose exec postgres sh -c 'exec psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
```

At the `psql` prompt, enter `\password` to change the connected role's password. Type the new value only at its hidden prompts, then enter `\q`. Do not use `ALTER ROLE ... PASSWORD '...'`, a password-bearing URI, `PGPASSWORD=...`, or a shell argument for rotation: those can expose a secret in history, logs, or process inspection. PostgreSQL documents that [`\password` avoids cleartext command history and server logs](https://www.postgresql.org/docs/17/app-psql.html). If local authentication has been customized, `psql` may first request the current password interactively.

After `\password` succeeds, edit the private `.env` file locally: update `POSTGRES_PASSWORD` for the container's initialization configuration **and** update the password part of `DATABASE_URL` for new TCP connections. These are separate settings; changing one does not synchronize the other. In a URI, percent-encode password characters with URL meaning, such as `@` → `%40`, `:` → `%3A`, `/` → `%2F`, `#` → `%23`, and `%` → `%25`; encode the password component, not the entire URL. Also follow [Compose `.env` quoting rules](https://docs.docker.com/compose/how-tos/environment-variables/variable-interpolation/) for characters such as `$`. Avoid online encoders and do not paste the credential into a command. See [PostgreSQL connection URI rules](https://www.postgresql.org/docs/17/libpq-connect.html).

Verify a **new** connection with the read-only RAG-AI CLI command `python -m rag_ai.storage status`. It should report the schema state rather than a connection error; `status` is not a migration. Restart or reconnect application processes that cached `DATABASE_URL`. Existing established sessions may continue until they disconnect, so a password change does not prove they were terminated.

## Back up the local database

Make a backup **before** adopting or upgrading an existing schema. A `pg_dump` custom archive covers this database's objects and data, including sensitive documents; it is not a backup of server-wide roles, Docker configuration, or every database. Store it outside this repository in a private directory, restrict access, and keep a separate retention and recovery plan. The examples use the existing service's local socket and container environment, so no password appears in command arguments. `-T` disables Compose's TTY for binary output. [PostgreSQL documents `-Fc` custom archives](https://www.postgresql.org/docs/17/app-pgdump.html) and [`pg_restore -l` archive listing](https://www.postgresql.org/docs/17/app-pgrestore.html).

**Linux/macOS shell example** (run from the repository root with the local service ready):

```bash
(
  set -euo pipefail
  umask 077
  backup_dir="$HOME/rag-ai-backups"
  mkdir -p "$backup_dir"
  archive="$backup_dir/rag-ai-$(date -u +%Y%m%dT%H%M%SZ).dump"
  test ! -e "$archive"
  docker compose exec -T postgres sh -c 'exec pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' > "$archive"
  test -s "$archive"
  wc -c < "$archive"
  docker compose exec -T postgres pg_restore -l < "$archive" > /dev/null
)
echo "Backup checks exit status: $?"
```

The subshell stops on a failed command; require exit status `0`, a nonzero byte count, and successful archive listing. A failed dump can leave a partial file: do not treat it as a backup. Do not redirect error output into the archive. `pg_restore -l` only reads its contents; it does not prove that restoration will work.

**Windows PowerShell 7.4+ differences:** Use a private directory outside the repository, verify its access-control list, and use `$LASTEXITCODE` after native Docker commands. PowerShell 7.4+ preserves binary stdout when redirecting a native command to a file; older PowerShell versions can corrupt a custom archive with `>`, so use a suitable POSIX shell or a binary-safe PostgreSQL client instead. The following is a PowerShell 7.4+ example; it was not run on Windows in this documentation session:

```powershell
$backupDir = Join-Path $HOME 'rag-ai-backups'
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
(Get-Acl -LiteralPath $backupDir).Access  # Stop if unexpected users can read this directory.
$archive = Join-Path $backupDir ("rag-ai-{0}.dump" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
if (Test-Path -LiteralPath $archive) { throw 'Archive path already exists.' }
docker compose exec -T postgres sh -c 'exec pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' > $archive
if ($LASTEXITCODE -ne 0) { throw 'pg_dump failed; do not use the partial archive.' }
$size = (Get-Item -LiteralPath $archive).Length
if ($size -le 0) { throw 'The archive is empty.' }
Get-Content -LiteralPath $archive -AsByteStream -ReadCount 65536 -ErrorAction Stop | docker compose exec -T postgres pg_restore -l > $null
if ($LASTEXITCODE -ne 0) { throw 'pg_restore could not list the archive.' }
```

The PowerShell byte-stream behavior is documented by [Microsoft](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_redirection?view=powershell-7.4). Keep the directory readable only by authorized users. Do not rely on a listable archive as the sole recovery check: with separate authorization, **restore into an isolated disposable database**, confirm the expected schema and representative synthetic or authorized records, and leave the development database untouched. Never run a test restore against the active development database.

## Versioned Phase 4 schema

Versioned migrations make schema changes repeatable while protecting data from an application version that expects a different structure. The runner reads ordered `src/rag_ai/storage/migrations/NNNN_name.sql` files; the current baseline is `0001_initial.sql`. `rag_ai_schema_migrations` records each applied version, its SQL SHA-256 checksum, and application time. Checksums detect edits to a migration after it was applied; unknown, missing, or mismatched versions are rejected. Never edit applied migration SQL or manually change this table. Indexing and deletion require the current supported version. Migrations never run automatically at API startup or during requests.

| Command | Effect | Typical result |
| --- | --- | --- |
| `python -m rag_ai.storage status` | Read-only schema/version diagnostic. | `empty`, `unversioned Phase 4 schema`, `partial unversioned schema`, or `version 1` for supported local states. |
| `python -m rag_ai.storage upgrade` | **Writes schema state:** transactionally applies pending numbered migrations and records checksums. Safe to rerun on a compatible versioned schema. | `Schema at version 1.` with the current files. Refuses an existing unversioned Phase 4 schema. |
| `python -m rag_ai.storage adopt-legacy` | **Writes migration history:** verifies the known unversioned Phase 4 shape, then records version 1 without rewriting document tables. | `Adopted legacy schema as version 1.`; refuses partial/incompatible or already-versioned schemas. |

The last user-reported status of the existing local database was **`unversioned Phase 4 schema`**. That status was not checked again for this document; do not assume adoption has happened. For an **empty, new** database, use `status`, then—after authorization—`upgrade`, then `status`. For the reported **unversioned** database, obtain operational authorization and a verified backup first, then use this sequence:

```text
python -m rag_ai.storage status
python -m rag_ai.storage adopt-legacy
python -m rag_ai.storage upgrade
python -m rag_ai.storage status
```

The first `status` must say `unversioned Phase 4 schema` before considering adoption. The runner's adoption check compares table columns/types/nullability, essential primary/foreign/unique constraints, dimension checks, and the collection/document index to its supported Phase 4 baseline. Successful `adopt-legacy` records version 1; the following `upgrade` is currently a compatible recheck; final `status` should print `version 1`. **Do not run `adopt-legacy` or `upgrade` without explicit operational authorization.** `status` alone neither fixes nor proves the physical schema is fully compatible. If adoption rejects a partial or changed schema, stop and inspect it against the backup; do not force adoption or modify migration history manually.

## Troubleshooting without accidental writes

Start with read-only checks: `docker compose ps postgres` for container state, then `python -m rag_ai.storage status` for the configured database. A container that is starting or unhealthy may not yet accept connections; wait for readiness before retrying. `DATABASE_URL is required for schema commands.` means it is absent from the process environment and optional local `.env`. Authentication failures can mean the URL password or username differs from the PostgreSQL role; after rotation, check the private file without printing it. A connection failure can also mean the service is stopped, the host/port is wrong, or PostgreSQL is not ready.

`unversioned Phase 4 schema` calls for backup-first explicit adoption, **not** immediate `upgrade`. `partial unversioned schema`, an incompatible legacy-shape error, a migration checksum mismatch, or an unknown newer version require investigation and recovery planning; the runner refuses to guess. `Database already has migration history; use upgrade.` means `adopt-legacy` is the wrong command; use `status` first. `upgrade`, `adopt-legacy`, password rotation, and any restore **modify state** and require authorization. Avoid `docker compose down -v`, manual history edits, and restores into the active database as troubleshooting shortcuts.

## Retention and deletion

Indexed original bytes, chunks, and vectors remain in PostgreSQL until explicitly deleted; **there is no automatic retention expiry**. `PostgresIndexStore.delete_document(collection_id=..., document_id=...)` deletes a document only in the specified collection. Foreign-key cascades remove its chunks and embeddings in the same transaction. Other documents and shared embedding-model metadata remain. The method returns no existence signal, and there is no HTTP deletion endpoint or collection authorization yet. Use only synthetic or otherwise authorized documents in this development environment. Backups and database snapshots may retain older copies and require their own retention policy.
