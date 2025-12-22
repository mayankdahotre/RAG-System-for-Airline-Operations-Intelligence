"""
Conversation Memory Store
Maintains session context for multi-turn interactions
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
import structlog

logger = structlog.get_logger()


@dataclass
class Message:
    """A single message in a conversation"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ConversationSession:
    """A conversation session with history"""
    session_id: str
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Query-specific context
    active_fleet: Optional[str] = None
    active_airport: Optional[str] = None
    last_query_type: Optional[str] = None
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the conversation."""
        self.messages.append(Message(
            role=role,
            content=content,
            metadata=metadata
        ))
        self.last_activity = datetime.now()
    
    def get_history(self, max_messages: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation history formatted for LLM."""
        recent = self.messages[-max_messages:]
        return [{"role": m.role, "content": m.content} for m in recent]
    
    def get_context_summary(self) -> str:
        """Generate a summary of conversation context."""
        summary_parts = []
        
        if self.active_fleet:
            summary_parts.append(f"Fleet: {self.active_fleet}")
        if self.active_airport:
            summary_parts.append(f"Airport: {self.active_airport}")
        if self.last_query_type:
            summary_parts.append(f"Topic: {self.last_query_type}")
        
        # Add recent query topics
        recent_queries = [
            m.content[:50] for m in self.messages[-3:] if m.role == "user"
        ]
        if recent_queries:
            summary_parts.append(f"Recent queries: {'; '.join(recent_queries)}")
        
        return " | ".join(summary_parts) if summary_parts else "New conversation"
    
    def update_context(self, key: str, value: Any):
        """Update session context."""
        self.context[key] = value
        self.last_activity = datetime.now()
    
    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """Check if session has expired."""
        return datetime.now() - self.last_activity > timedelta(minutes=timeout_minutes)


class ConversationStore:
    """
    In-memory conversation store with LRU eviction.
    In production, use Redis or a proper database.
    """
    
    def __init__(
        self,
        max_sessions: int = 1000,
        session_timeout_minutes: int = 30
    ):
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout_minutes
        self.sessions: OrderedDict[str, ConversationSession] = OrderedDict()
    
    def get_or_create(self, session_id: str) -> ConversationSession:
        """Get existing session or create new one."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            # Move to end (most recently used)
            self.sessions.move_to_end(session_id)
            return session
        
        # Create new session
        session = ConversationSession(session_id=session_id)
        self.sessions[session_id] = session
        
        # Evict old sessions if necessary
        self._evict_if_needed()
        
        logger.info("session_created", session_id=session_id)
        return session
    
    def get(self, session_id: str) -> Optional[ConversationSession]:
        """Get a session by ID."""
        return self.sessions.get(session_id)
    
    def add_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict] = None
    ):
        """Add a user-assistant exchange to the session."""
        session = self.get_or_create(session_id)
        session.add_message("user", user_message, metadata)
        session.add_message("assistant", assistant_message, metadata)
    
    def update_session_context(
        self,
        session_id: str,
        fleet: Optional[str] = None,
        airport: Optional[str] = None,
        query_type: Optional[str] = None
    ):
        """Update session context with query information."""
        session = self.get_or_create(session_id)
        
        if fleet:
            session.active_fleet = fleet
        if airport:
            session.active_airport = airport
        if query_type:
            session.last_query_type = query_type
    
    def get_formatted_history(
        self,
        session_id: str,
        max_messages: int = 6
    ) -> List[Dict[str, str]]:
        """Get formatted conversation history for LLM context."""
        session = self.get(session_id)
        if not session:
            return []
        return session.get_history(max_messages)
    
    def delete(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info("session_deleted", session_id=session_id)
            return True
        return False
    
    def cleanup_expired(self) -> int:
        """Remove expired sessions."""
        expired = [
            sid for sid, session in self.sessions.items()
            if session.is_expired(self.session_timeout)
        ]
        for sid in expired:
            del self.sessions[sid]
        
        if expired:
            logger.info("sessions_cleaned", count=len(expired))
        
        return len(expired)
    
    def _evict_if_needed(self):
        """Evict oldest sessions if at capacity."""
        while len(self.sessions) > self.max_sessions:
            oldest_id, _ = self.sessions.popitem(last=False)
            logger.info("session_evicted", session_id=oldest_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            "total_sessions": len(self.sessions),
            "max_sessions": self.max_sessions,
            "oldest_session": min(
                (s.created_at for s in self.sessions.values()),
                default=None
            ),
            "newest_session": max(
                (s.created_at for s in self.sessions.values()),
                default=None
            )
        }


# Global store instance
conversation_store = ConversationStore()

