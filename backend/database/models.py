"""
Database models for authentication, document tracking, and analysis results.
"""
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, Integer, Float, Text, JSON, Enum as SQLEnum
from sqlalchemy.orm import relationship
import enum
import uuid

from ..config.settings import settings
from .database import Base


# Determine schema based on configuration
SCHEMA = settings.database_schema if settings.is_postgres() else None


class UserRole(str, enum.Enum):
    """User roles enum."""
    USER = "USER"
    ADMIN = "ADMIN"


class DocumentType(str, enum.Enum):
    """Document type enum."""
    PPTX = "pptx"
    PDF = "pdf"


class IssueSeverity(str, enum.Enum):
    """Issue severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueCategory(str, enum.Enum):
    """Issue categories."""
    GRAMMAR = "grammar"
    TONE = "tone"
    STYLE = "style"
    FORMATTING = "formatting"
    CONTENT = "content"


# Helper function to create schema-aware enum columns
def _enum_column(enum_class, **kwargs):
    """Create an ENUM column with schema support for PostgreSQL."""
    return Column(SQLEnum(enum_class, schema=SCHEMA, name=enum_class.__name__.lower()), **kwargs)


class User(Base):
    """
    User model - stores authenticated users from Azure AD.
    """
    __tablename__ = "users"
    __table_args__ = {"schema": SCHEMA} if SCHEMA else {}

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    azure_oid = Column(String(255), unique=True, nullable=False, index=True)  # Azure Object ID
    email = Column(String(255), unique=True, nullable=False, index=True)
    display_name = Column(String(255), nullable=False)
    role = Column(SQLEnum(UserRole, schema=SCHEMA, name='userrole'), nullable=False, default=UserRole.USER)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    # Relationships
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="uploaded_by_user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, role={self.role})>"

    def to_dict(self):
        """Convert user to dictionary."""
        return {
            "id": self.id,
            "azure_oid": self.azure_oid,
            "email": self.email,
            "display_name": self.display_name,
            "role": self.role.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "is_active": self.is_active
        }


class Session(Base):
    """
    Session model - stores active user sessions.
    """
    __tablename__ = "sessions"
    __table_args__ = {"schema": SCHEMA} if SCHEMA else {}

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey(f"{SCHEMA}.users.id" if SCHEMA else "users.id", ondelete="CASCADE"), nullable=False, index=True)
    token = Column(String(512), unique=True, nullable=False, index=True)  # JWT token
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ip_address = Column(String(45), nullable=True)  # IPv6 max length
    user_agent = Column(String(512), nullable=True)

    # Relationships
    user = relationship("User", back_populates="sessions")

    def __repr__(self):
        return f"<Session(id={self.id}, user_id={self.user_id}, expires_at={self.expires_at})>"

    def is_expired(self):
        """Check if session is expired."""
        return datetime.utcnow() > self.expires_at

    def to_dict(self):
        """Convert session to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "ip_address": self.ip_address,
            "is_expired": self.is_expired()
        }


class Document(Base):
    """
    Document model - stores uploaded documents metadata.
    """
    __tablename__ = "documents"
    __table_args__ = {"schema": SCHEMA} if SCHEMA else {}

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(String(10), unique=True, nullable=False, index=True)  # e.g., "001", "002"
    original_filename = Column(String(255), nullable=False)
    clean_filename = Column(String(255), nullable=False)  # Stored filename
    document_type = Column(SQLEnum(DocumentType, schema=SCHEMA, name='documenttype'), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)

    # User tracking
    user_id = Column(String(36), ForeignKey(f"{SCHEMA}.users.id" if SCHEMA else "users.id", ondelete="SET NULL"), nullable=True, index=True)
    user_info = Column(String(255), nullable=True)  # Legacy: uploader name/email from form

    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    processed_at = Column(DateTime, nullable=True)

    # File paths (relative to upload/output dirs)
    upload_path = Column(String(512), nullable=False)  # Relative path in uploads dir
    output_path = Column(String(512), nullable=True)  # Relative path in outputs dir

    # Relationships
    uploaded_by_user = relationship("User", back_populates="documents")
    analysis_results = relationship("AnalysisResult", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Document(id={self.id}, file_id={self.file_id}, filename={self.original_filename})>"

    def to_dict(self):
        """Convert document to dictionary."""
        return {
            "id": self.id,
            "file_id": self.file_id,
            "original_filename": self.original_filename,
            "clean_filename": self.clean_filename,
            "document_type": self.document_type.value,
            "file_size_bytes": self.file_size_bytes,
            "file_size_mb": round(self.file_size_bytes / (1024 * 1024), 2) if self.file_size_bytes else 0,
            "user_id": self.user_id,
            "user_info": self.user_info,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
        }


class AnalysisResult(Base):
    """
    Analysis result model - stores analysis metadata and summary.
    """
    __tablename__ = "analysis_results"
    __table_args__ = {"schema": SCHEMA} if SCHEMA else {}

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey(f"{SCHEMA}.documents.id" if SCHEMA else "documents.id", ondelete="CASCADE"), nullable=False, index=True)

    # Analysis summary
    total_issues = Column(Integer, default=0)
    grammar_issues = Column(Integer, default=0)
    tone_issues = Column(Integer, default=0)
    issues_with_suggestions = Column(Integer, default=0)

    # Breakdown (stored as JSON)
    severity_breakdown = Column(JSON, default=dict)  # {"critical": 0, "high": 2, ...}
    category_breakdown = Column(JSON, default=dict)  # {"grammar": 5, "tone": 3, ...}
    rule_breakdown = Column(JSON, default=dict)  # {"numeral_formatting": 2, ...}

    # Metadata
    total_pages = Column(Integer, nullable=True)
    total_elements = Column(Integer, nullable=True)
    processing_time_seconds = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    document = relationship("Document", back_populates="analysis_results")
    issues = relationship("StyleIssue", back_populates="analysis", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<AnalysisResult(id={self.id}, document_id={self.document_id}, total_issues={self.total_issues})>"

    def to_dict(self):
        """Convert analysis result to dictionary."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "total_issues": self.total_issues,
            "grammar_issues": self.grammar_issues,
            "tone_issues": self.tone_issues,
            "issues_with_suggestions": self.issues_with_suggestions,
            "severity_breakdown": self.severity_breakdown,
            "category_breakdown": self.category_breakdown,
            "rule_breakdown": self.rule_breakdown,
            "total_pages": self.total_pages,
            "total_elements": self.total_elements,
            "processing_time_seconds": self.processing_time_seconds,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class StyleIssue(Base):
    """
    Style issue model - stores individual issues found during analysis.
    """
    __tablename__ = "style_issues"
    __table_args__ = {"schema": SCHEMA} if SCHEMA else {}

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(Integer, ForeignKey(f"{SCHEMA}.analysis_results.id" if SCHEMA else "analysis_results.id", ondelete="CASCADE"), nullable=False, index=True)

    # Issue classification
    category = Column(SQLEnum(IssueCategory, schema=SCHEMA, name='issuecategory'), nullable=False, index=True)
    severity = Column(SQLEnum(IssueSeverity, schema=SCHEMA, name='issueseverity'), nullable=False, index=True)
    rule_name = Column(String(100), nullable=False, index=True)

    # Location
    slide_number = Column(Integer, nullable=True, index=True)  # or page_number
    element_type = Column(String(50), nullable=True)  # e.g., "title", "body", "bullet"
    element_index = Column(Integer, nullable=True)

    # Content
    original_text = Column(Text, nullable=False)
    suggested_text = Column(Text, nullable=True)
    explanation = Column(Text, nullable=True)

    # Metadata
    detected_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    analysis = relationship("AnalysisResult", back_populates="issues")

    def __repr__(self):
        return f"<StyleIssue(id={self.id}, rule={self.rule_name}, severity={self.severity})>"

    def to_dict(self):
        """Convert style issue to dictionary."""
        return {
            "id": self.id,
            "analysis_id": self.analysis_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "rule_name": self.rule_name,
            "slide_number": self.slide_number,
            "element_type": self.element_type,
            "element_index": self.element_index,
            "original_text": self.original_text,
            "suggested_text": self.suggested_text,
            "explanation": self.explanation,
            "detected_at": self.detected_at.isoformat() if self.detected_at else None,
        }


class AuditLog(Base):
    """
    Audit log model - tracks user actions for compliance.
    """
    __tablename__ = "audit_logs"
    __table_args__ = {"schema": SCHEMA} if SCHEMA else {}

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey(f"{SCHEMA}.users.id" if SCHEMA else "users.id", ondelete="SET NULL"), nullable=True, index=True)

    # Action details
    action = Column(String(100), nullable=False, index=True)  # e.g., "upload_document", "view_analysis"
    resource_type = Column(String(50), nullable=True)  # e.g., "document", "analysis"
    resource_id = Column(String(50), nullable=True)

    # Metadata
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(512), nullable=True)
    details = Column(JSON, default=dict)  # Additional context

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    user = relationship("User", back_populates="audit_logs")

    def __repr__(self):
        return f"<AuditLog(id={self.id}, action={self.action}, user_id={self.user_id})>"

    def to_dict(self):
        """Convert audit log to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "ip_address": self.ip_address,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "details": self.details,
        }
