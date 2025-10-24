"""
Database module for authentication, document tracking, and analysis.
"""
from .database import (
    Base,
    async_engine,
    sync_engine,
    AsyncSessionLocal,
    SyncSessionLocal,
    get_db,
    get_db_sync,
    init_db,
    init_db_async,
    dispose_engines,
)

from .models import (
    User,
    Session,
    UserRole,
    Document,
    DocumentType,
    AnalysisResult,
    StyleIssue,
    IssueSeverity,
    IssueCategory,
    AuditLog,
)

__all__ = [
    # Database
    "Base",
    "async_engine",
    "sync_engine",
    "AsyncSessionLocal",
    "SyncSessionLocal",
    "get_db",
    "get_db_sync",
    "init_db",
    "init_db_async",
    "dispose_engines",
    # Models
    "User",
    "Session",
    "UserRole",
    "Document",
    "DocumentType",
    "AnalysisResult",
    "StyleIssue",
    "IssueSeverity",
    "IssueCategory",
    "AuditLog",
]
