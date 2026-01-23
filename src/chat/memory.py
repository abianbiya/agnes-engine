"""
Conversation memory management for the RAG chatbot.

This module provides session-based conversation history management with support for
multiple concurrent sessions, configurable history windows, and automatic session cleanup.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from src.utils.logging import LoggerMixin


class ConversationMemory(LoggerMixin):
    """
    Manages conversation history for multiple chat sessions.
    
    Provides session-based memory storage with configurable window size and
    automatic cleanup of old sessions.
    
    Attributes:
        window_size: Maximum number of message pairs to retain per session
        session_timeout: Time in minutes before inactive sessions are cleaned up
        sessions: Dictionary mapping session IDs to message histories
        last_accessed: Dictionary tracking last access time for each session
    """
    
    def __init__(
        self,
        window_size: int = 10,
        session_timeout: int = 60,
    ):
        """
        Initialize conversation memory manager.
        
        Args:
            window_size: Maximum number of message pairs (human + AI) to keep.
                Default is 10 pairs (20 messages total).
            session_timeout: Minutes of inactivity before session is cleaned up.
                Default is 60 minutes. Set to 0 to disable cleanup.
        """
        super().__init__()
        
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        
        if session_timeout < 0:
            raise ValueError("session_timeout must be non-negative")
        
        self.window_size = window_size
        self.session_timeout = session_timeout
        self.sessions: Dict[str, ChatMessageHistory] = {}
        self.last_accessed: Dict[str, datetime] = {}
        
        self.logger.info(
            "initialized_conversation_memory",
            window_size=window_size,
            session_timeout=session_timeout,
        )
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new conversation session.
        
        Args:
            session_id: Optional custom session ID. If not provided, a UUID is generated.
            
        Returns:
            The session ID (either provided or generated)
            
        Raises:
            ValueError: If the provided session_id already exists
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        if session_id in self.sessions:
            raise ValueError(f"Session {session_id} already exists")
        
        self.sessions[session_id] = ChatMessageHistory()
        self.last_accessed[session_id] = datetime.now()
        
        self.logger.info("created_session", session_id=session_id)
        
        return session_id
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Get or create chat message history for a session.
        
        This method is compatible with LangChain's RunnableWithMessageHistory.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Chat message history for the session
        """
        # Clean up old sessions periodically
        self._cleanup_old_sessions()
        
        # Create session if it doesn't exist
        if session_id not in self.sessions:
            self.logger.info("auto_creating_session", session_id=session_id)
            self.create_session(session_id)
        
        # Update last access time
        self.last_accessed[session_id] = datetime.now()
        
        return self.sessions[session_id]
    
    def get_messages(self, session_id: str) -> List[BaseMessage]:
        """
        Get all messages for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of messages in chronological order
        """
        if session_id not in self.sessions:
            # Auto-create session if it doesn't exist
            self.create_session(session_id)
            return []
        
        self.last_accessed[session_id] = datetime.now()
        history = self.sessions[session_id]
        
        # Apply window size limit
        messages = history.messages
        max_messages = self.window_size * 2  # human + AI pairs
        
        if len(messages) > max_messages:
            # Keep only the most recent messages within window
            messages = messages[-max_messages:]
            self.logger.debug(
                "applied_window_limit",
                session_id=session_id,
                total_messages=len(history.messages),
                kept_messages=len(messages),
            )
        
        return messages
    
    def add_messages(
        self,
        session_id: str,
        messages: List[BaseMessage],
    ) -> None:
        """
        Add messages to a session.
        
        Args:
            session_id: Session identifier
            messages: List of messages to add
            
        Raises:
            KeyError: If session doesn't exist
        """
        if session_id not in self.sessions:
            raise KeyError(f"Session {session_id} not found")
        
        self.sessions[session_id].add_messages(messages)
        self.last_accessed[session_id] = datetime.now()
        
        self.logger.debug(
            "added_messages",
            session_id=session_id,
            message_count=len(messages),
        )
    
    def add_exchange(
        self,
        session_id: str,
        human_message: str,
        ai_message: str,
    ) -> None:
        """
        Add a human-AI message exchange to the session.
        
        Convenience method for adding a question-answer pair.
        
        Args:
            session_id: Session identifier
            human_message: User's message
            ai_message: AI's response
            
        Raises:
            KeyError: If session doesn't exist
        """
        messages = [
            HumanMessage(content=human_message),
            AIMessage(content=ai_message),
        ]
        self.add_messages(session_id, messages)
        
        self.logger.debug(
            "added_exchange",
            session_id=session_id,
            human_message_length=len(human_message),
            ai_message_length=len(ai_message),
        )
    
    def clear_session(self, session_id: str) -> None:
        """
        Clear all messages from a session.
        
        Args:
            session_id: Session identifier
            
        Raises:
            KeyError: If session doesn't exist
        """
        if session_id not in self.sessions:
            raise KeyError(f"Session {session_id} not found")
        
        message_count = len(self.sessions[session_id].messages)
        self.sessions[session_id].clear()
        self.last_accessed[session_id] = datetime.now()
        
        self.logger.info(
            "cleared_session",
            session_id=session_id,
            cleared_messages=message_count,
        )
    
    def delete_session(self, session_id: str) -> None:
        """
        Delete a session completely.
        
        Args:
            session_id: Session identifier
            
        Raises:
            KeyError: If session doesn't exist
        """
        if session_id not in self.sessions:
            raise KeyError(f"Session {session_id} not found")
        
        del self.sessions[session_id]
        del self.last_accessed[session_id]
        
        self.logger.info("deleted_session", session_id=session_id)
    
    def list_sessions(self) -> List[str]:
        """
        List all active session IDs.
        
        Returns:
            List of session IDs
        """
        return list(self.sessions.keys())
    
    def get_session_info(self, session_id: str) -> Dict[str, any]:
        """
        Get information about a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with session information
            
        Raises:
            KeyError: If session doesn't exist
        """
        if session_id not in self.sessions:
            raise KeyError(f"Session {session_id} not found")
        
        messages = self.sessions[session_id].messages
        last_access = self.last_accessed[session_id]
        
        return {
            "session_id": session_id,
            "message_count": len(messages),
            "last_accessed": last_access.isoformat(),
            "inactive_minutes": (datetime.now() - last_access).total_seconds() / 60,
        }
    
    def _cleanup_old_sessions(self) -> None:
        """
        Remove sessions that have been inactive for longer than session_timeout.
        
        Called automatically during get_session_history.
        """
        if self.session_timeout <= 0:
            return  # Cleanup disabled
        
        now = datetime.now()
        timeout_delta = timedelta(minutes=self.session_timeout)
        sessions_to_delete = []
        
        for session_id, last_access in self.last_accessed.items():
            if now - last_access > timeout_delta:
                sessions_to_delete.append(session_id)
        
        for session_id in sessions_to_delete:
            self.logger.info(
                "cleaning_up_session",
                session_id=session_id,
                inactive_minutes=(now - self.last_accessed[session_id]).total_seconds() / 60,
            )
            del self.sessions[session_id]
            del self.last_accessed[session_id]
        
        if sessions_to_delete:
            self.logger.info(
                "cleanup_completed",
                cleaned_sessions=len(sessions_to_delete),
                active_sessions=len(self.sessions),
            )
    
    def cleanup_all(self) -> int:
        """
        Manually clean up all inactive sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        initial_count = len(self.sessions)
        self._cleanup_old_sessions()
        cleaned = initial_count - len(self.sessions)
        
        self.logger.info(
            "manual_cleanup",
            cleaned_sessions=cleaned,
            remaining_sessions=len(self.sessions),
        )
        
        return cleaned


class ConversationManager(ConversationMemory):
    """
    Alias for ConversationMemory for backward compatibility with design docs.
    """
    pass


def create_runnable_with_history(
    runnable,
    memory: ConversationMemory,
) -> RunnableWithMessageHistory:
    """
    Wrap a LangChain runnable with message history support.
    
    Args:
        runnable: LangChain runnable (chain, LLM, etc.)
        memory: ConversationMemory instance
        
    Returns:
        RunnableWithMessageHistory that automatically manages conversation history
        
    Example:
        >>> chain = create_rag_chain(...)
        >>> memory = ConversationMemory()
        >>> chain_with_history = create_runnable_with_history(chain, memory)
        >>> response = chain_with_history.invoke(
        ...     {"question": "What is RAG?"},
        ...     config={"configurable": {"session_id": "user123"}}
        ... )
    """
    return RunnableWithMessageHistory(
        runnable,
        memory.get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )


__all__ = [
    "ConversationMemory",
    "ConversationManager",
    "create_runnable_with_history",
]
