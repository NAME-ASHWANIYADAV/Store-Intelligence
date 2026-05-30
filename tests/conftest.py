"""
Store Intelligence System - Test Configuration
Shared fixtures for async database testing.
"""

import asyncio
import os
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# Override DATABASE_URL for tests (use SQLite for isolation)
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_store_intel.db"
os.environ["DATABASE_URL_SYNC"] = "sqlite:///./test_store_intel.db"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"
os.environ["DEBUG"] = "true"

from app.database import Base, get_db
from app.main import app


# Test database engine
test_engine = create_async_engine(
    "sqlite+aiosqlite:///./test_store_intel.db",
    echo=False,
)

test_session_factory = async_sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@pytest_asyncio.fixture(autouse=True)
async def setup_database():
    """Create tables before each test, drop after."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def override_get_db():
    """Override DB dependency for tests."""
    async with test_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# Override the dependency
app.dependency_overrides[get_db] = override_get_db


@pytest_asyncio.fixture
async def client():
    """Async test client for FastAPI."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def db_session():
    """Direct database session for test setup."""
    async with test_session_factory() as session:
        yield session
