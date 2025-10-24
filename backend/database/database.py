"""
Database connection and session management with async SQLAlchemy support.

Supports both SQLite (sync) and PostgreSQL (async) connections.
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, declarative_base
from typing import AsyncGenerator
import logging

from ..config.settings import settings

logger = logging.getLogger(__name__)

# Base class for database models
Base = declarative_base()

# Async Engine and Session (for PostgreSQL with asyncpg)
async_engine = None
AsyncSessionLocal = None

# Sync Engine and Session (for SQLite and migrations)
sync_engine = None
SyncSessionLocal = None


def init_engines():
    """Initialize database engines based on configuration."""
    global async_engine, AsyncSessionLocal, sync_engine, SyncSessionLocal

    is_postgres = settings.is_postgres()

    if is_postgres:
        # PostgreSQL with async support
        logger.info("Initializing async PostgreSQL engine")

        async_engine = create_async_engine(
            settings.database_url,
            echo=settings.debug,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
            pool_pre_ping=True,  # Verify connections before using
        )

        AsyncSessionLocal = async_sessionmaker(
            async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )

        # Sync connection for Alembic migrations
        if settings.database_url_sync:
            logger.info("Initializing sync PostgreSQL engine for migrations")
            sync_engine = create_engine(
                settings.database_url_sync,
                echo=settings.debug,
                pool_pre_ping=True,
            )
            SyncSessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=sync_engine
            )
    else:
        # SQLite with sync support (fallback)
        logger.info("Initializing sync SQLite engine")

        sync_engine = create_engine(
            settings.database_url,
            connect_args={"check_same_thread": False},
            echo=settings.debug
        )

        SyncSessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=sync_engine
        )


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Async dependency to get database session (for PostgreSQL).

    Usage in FastAPI:
        @app.get("/example")
        async def example(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(User))
            users = result.scalars().all()
    """
    if AsyncSessionLocal is None:
        init_engines()

    if AsyncSessionLocal is None:
        raise RuntimeError("Async database engine not initialized. Check database configuration.")

    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_db_sync():
    """
    Sync dependency to get database session (for SQLite or migrations).

    Usage in FastAPI:
        @app.get("/example")
        def example(db: Session = Depends(get_db_sync)):
            users = db.query(User).all()
    """
    if SyncSessionLocal is None:
        init_engines()

    if SyncSessionLocal is None:
        raise RuntimeError("Sync database engine not initialized. Check database configuration.")

    db = SyncSessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """
    Initialize database - create all tables.
    Call this on application startup.
    """
    from . import models  # Import models to register them

    init_engines()

    is_postgres = settings.is_postgres()

    if is_postgres:
        logger.info("Skipping auto table creation for PostgreSQL - use Alembic migrations")
        logger.info("Run: alembic upgrade head")
    else:
        logger.info("Creating SQLite tables automatically")
        if sync_engine:
            Base.metadata.create_all(bind=sync_engine)
        else:
            raise RuntimeError("Sync engine not initialized for SQLite")


async def init_db_async():
    """
    Async version of init_db for PostgreSQL.
    Creates all tables using async engine.

    NOTE: In production, use Alembic migrations instead.
    """
    from . import models  # Import models to register them

    if async_engine is None:
        init_engines()

    if async_engine:
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully (async)")
    else:
        raise RuntimeError("Async engine not initialized")


async def dispose_engines():
    """
    Dispose database engines on application shutdown.
    Call this in FastAPI shutdown event.
    """
    if async_engine:
        await async_engine.dispose()
        logger.info("Async database engine disposed")

    if sync_engine:
        sync_engine.dispose()
        logger.info("Sync database engine disposed")
