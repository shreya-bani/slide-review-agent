"""
Alembic migration environment configuration.

Configured to work with PostgreSQL schema and async SQLAlchemy.
"""
from logging.config import fileConfig
import sys
from pathlib import Path

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from sqlalchemy import text

from alembic import context

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Import settings and models
from backend.config.settings import settings
from backend.database.database import Base
from backend.database import models  # Import models to register them

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Set the database URL from settings (use sync URL for migrations)
if settings.database_url_sync:
    config.set_main_option("sqlalchemy.url", settings.database_url_sync)
else:
    # Fallback: convert asyncpg to psycopg2
    sync_url = settings.database_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
    config.set_main_option("sqlalchemy.url", sync_url)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target metadata from our models
target_metadata = Base.metadata

# Get schema from settings
SCHEMA = settings.database_schema if settings.is_postgres() else None


def include_name(name, type_, parent_names):
    """
    Filter function to control what gets included in migrations.

    This ensures we only migrate tables in our target schema.
    """
    if type_ == "schema":
        # Include our target schema
        return name == SCHEMA or name is None
    else:
        return True


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table_schema=SCHEMA,  # Store alembic_version in our schema
        include_schemas=True,
        include_name=include_name,
    )

    with context.begin_transaction():
        # Set search_path to our schema so ENUMs are created in the right place
        if SCHEMA:
            context.execute(text(f"SET search_path TO {SCHEMA}, public"))
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table_schema=SCHEMA,  # Store alembic_version in our schema
            include_schemas=True,
            include_name=include_name,
        )

        with context.begin_transaction():
            # Set search_path to our schema so ENUMs are created in the right place
            if SCHEMA:
                connection.execute(text(f"SET search_path TO {SCHEMA}, public"))
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
