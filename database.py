import os
import ssl
from typing import AsyncGenerator
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

# Parse URL and handle SSL for asyncpg
parsed = urlparse(DATABASE_URL)
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

engine = create_async_engine(
    clean_url,
    echo=False,
    connect_args=connect_args,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
)
async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session

