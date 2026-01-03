import os
import ssl
import logging
from typing import AsyncGenerator
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import event
from sqlalchemy.pool import Pool

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

# Parse URL and handle SSL for asyncpg
parsed = urlparse(DATABASE_URL)
query_params = parse_qs(parsed.query)

# Remove sslmode and channel_binding from query params (asyncpg doesn't support channel_binding)
ssl_mode = query_params.pop("sslmode", None)
query_params.pop("channel_binding", None)  # Remove channel_binding as asyncpg doesn't support it
connect_args = {
    "command_timeout": 60,  # 60 seconds timeout for commands
    "timeout": 60,  # 60 seconds connection timeout
}
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

engine = create_async_engine(
    clean_url,
    echo=False,
    connect_args=connect_args,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Ping connections before using to detect stale ones
    pool_recycle=60,  # Recycle connections every minute (Neon has aggressive idle timeouts)
    pool_timeout=30,
)


# Handle connection invalidation gracefully
@event.listens_for(Pool, "checkout")
def check_connection(dbapi_conn, connection_rec, connection_proxy):
    """Verify connection is still valid on checkout from pool."""
    pass  # pool_pre_ping handles this


@event.listens_for(Pool, "invalidate")
def on_invalidate(dbapi_conn, connection_rec, exception):
    """Log when a connection is invalidated."""
    if exception:
        logger.warning(f"Connection invalidated due to: {exception}")


async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions with closed connection handling."""
    session = async_session_maker()
    try:
        yield session
    except Exception:
        # Try to rollback on error, but ignore if connection is closed
        try:
            await session.rollback()
        except Exception as rollback_err:
            logger.debug(f"Rollback failed (connection may be closed): {rollback_err}")
        raise
    finally:
        # Try to close, but ignore if connection is already closed
        try:
            await session.close()
        except Exception as close_err:
            logger.debug(f"Session close failed (connection may be closed): {close_err}")

