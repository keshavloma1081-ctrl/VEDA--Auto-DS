"""
Advanced Database Configuration with PostgreSQL Support
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import os

# Database URLs
SQLITE_URL = "sqlite:///./veda.db"
POSTGRES_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:password@localhost:5432/veda"
)

# Use PostgreSQL if available, fallback to SQLite
USE_POSTGRES = os.getenv("USE_POSTGRES", "false").lower() == "true"

if USE_POSTGRES:
    DATABASE_URL = POSTGRES_URL
    print("🐘 Using PostgreSQL database")
else:
    DATABASE_URL = SQLITE_URL.replace("sqlite:///", "sqlite+aiosqlite:///")
    print("💾 Using SQLite database")

# Async engine
async_engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

# Async session factory
AsyncSessionLocal = sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

async def get_async_db():
    """Async database session dependency"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()