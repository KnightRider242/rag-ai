"""Run explicit local PostgreSQL schema commands."""

import argparse

import psycopg

from rag_ai.core.config import Settings
from rag_ai.storage.migrate import MigrationError, MigrationRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the local RAG-AI database schema.")
    parser.add_argument("command", choices=("status", "upgrade", "adopt-legacy"))
    command = parser.parse_args().command
    settings = Settings()
    if settings.database_url is None:
        raise SystemExit("DATABASE_URL is required for schema commands.")
    try:
        runner = MigrationRunner(settings.database_url.get_secret_value())
        if command == "status":
            print(runner.status())
        elif command == "upgrade":
            print(f"Schema at version {runner.upgrade()}.")
        else:
            print(f"Adopted legacy schema as version {runner.adopt_legacy()}.")
    except MigrationError as exc:
        raise SystemExit(str(exc)) from exc
    except psycopg.Error as exc:
        raise SystemExit("Schema command failed; inspect the database connection.") from exc


if __name__ == "__main__":
    main()
