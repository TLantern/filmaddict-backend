import asyncio
import os
import ssl
from logging.config import fileConfig
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

from database import Base
from db.models import Video, Transcript, Highlight, Clip

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def get_url():
    database_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/dbname")
    
    # Parse URL and handle SSL for asyncpg (same as database.py)
    parsed = urlparse(database_url)
    query_params = parse_qs(parsed.query)
    
    # Remove sslmode from query params
    query_params.pop("sslmode", None)
    
    # Reconstruct URL without sslmode
    new_query = urlencode(query_params, doseq=True)
    clean_url = urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        new_query,
        parsed.fragment
    ))
    
    return clean_url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    database_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/dbname")
    
    # Parse URL and handle SSL for asyncpg (same as database.py)
    parsed = urlparse(database_url)
    query_params = parse_qs(parsed.query)
    
    # Remove sslmode from query params and configure SSL for asyncpg
    ssl_mode = query_params.pop("sslmode", None)
    connect_args = {}
    if ssl_mode and ssl_mode[0] == "require":
        # Create SSL context that doesn't verify certificates (for Neon/cloud databases)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        connect_args["ssl"] = ssl_context
    
    # Reconstruct URL without sslmode
    new_query = urlencode(query_params, doseq=True)
    clean_url = urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        new_query,
        parsed.fragment
    ))
    
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = clean_url
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args=connect_args,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""

    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

