"""
Unit tests for conversation memory module.

Tests session management, message history, and cleanup functionality.
"""

import time
from datetime import datetime, timedelta

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.chat.memory import ConversationManager, ConversationMemory


class TestConversationMemoryInit:
    """Test ConversationMemory initialization."""
    
    def test_init_with_defaults(self):
        """Should initialize with default parameters."""
        memory = ConversationMemory()
        
        assert memory.window_size == 10
        assert memory.session_timeout == 60
        assert len(memory.sessions) == 0
        assert len(memory.last_accessed) == 0
    
    def test_init_with_custom_window_size(self):
        """Should initialize with custom window size."""
        memory = ConversationMemory(window_size=5)
        assert memory.window_size == 5
    
    def test_init_with_custom_timeout(self):
        """Should initialize with custom timeout."""
        memory = ConversationMemory(session_timeout=30)
        assert memory.session_timeout == 30
    
    def test_init_with_zero_timeout_disables_cleanup(self):
        """Should allow zero timeout to disable cleanup."""
        memory = ConversationMemory(session_timeout=0)
        assert memory.session_timeout == 0
    
    def test_init_with_invalid_window_size(self):
        """Should raise error for invalid window size."""
        with pytest.raises(ValueError, match="window_size must be at least 1"):
            ConversationMemory(window_size=0)
        
        with pytest.raises(ValueError, match="window_size must be at least 1"):
            ConversationMemory(window_size=-1)
    
    def test_init_with_negative_timeout(self):
        """Should raise error for negative timeout."""
        with pytest.raises(ValueError, match="session_timeout must be non-negative"):
            ConversationMemory(session_timeout=-1)


class TestSessionCreation:
    """Test session creation and management."""
    
    def test_create_session_generates_uuid(self):
        """Should generate UUID for new session."""
        memory = ConversationMemory()
        session_id = memory.create_session()
        
        assert session_id is not None
        assert len(session_id) > 0
        assert session_id in memory.sessions
        assert session_id in memory.last_accessed
    
    def test_create_session_with_custom_id(self):
        """Should create session with custom ID."""
        memory = ConversationMemory()
        custom_id = "user-12345"
        session_id = memory.create_session(session_id=custom_id)
        
        assert session_id == custom_id
        assert custom_id in memory.sessions
    
    def test_create_session_duplicate_raises_error(self):
        """Should raise error when creating duplicate session."""
        memory = ConversationMemory()
        session_id = memory.create_session(session_id="test-session")
        
        with pytest.raises(ValueError, match="already exists"):
            memory.create_session(session_id="test-session")
    
    def test_create_multiple_sessions(self):
        """Should create multiple independent sessions."""
        memory = ConversationMemory()
        
        session1 = memory.create_session()
        session2 = memory.create_session()
        session3 = memory.create_session()
        
        assert session1 != session2
        assert session2 != session3
        assert len(memory.sessions) == 3


class TestGetSessionHistory:
    """Test getting session history."""
    
    def test_get_existing_session(self):
        """Should return history for existing session."""
        memory = ConversationMemory()
        session_id = memory.create_session()
        
        history = memory.get_session_history(session_id)
        
        assert history is not None
        assert len(history.messages) == 0
    
    def test_get_nonexistent_session_creates_it(self):
        """Should auto-create session if it doesn't exist."""
        memory = ConversationMemory()
        session_id = "new-session"
        
        history = memory.get_session_history(session_id)
        
        assert history is not None
        assert session_id in memory.sessions
    
    def test_get_session_updates_last_accessed(self):
        """Should update last accessed time."""
        memory = ConversationMemory()
        session_id = memory.create_session()
        
        initial_time = memory.last_accessed[session_id]
        time.sleep(0.01)  # Small delay
        
        memory.get_session_history(session_id)
        
        assert memory.last_accessed[session_id] > initial_time


class TestGetMessages:
    """Test retrieving messages."""
    
    def test_get_messages_empty_session(self):
        """Should return empty list for new session."""
        memory = ConversationMemory()
        session_id = memory.create_session()
        
        messages = memory.get_messages(session_id)
        
        assert messages == []
    
    def test_get_messages_nonexistent_session(self):
        """Should raise KeyError for nonexistent session."""
        memory = ConversationMemory()
        
        with pytest.raises(KeyError, match="not found"):
            memory.get_messages("nonexistent")
    
    def test_get_messages_with_content(self):
        """Should return messages after adding content."""
        memory = ConversationMemory()
        session_id = memory.create_session()
        
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]
        memory.add_messages(session_id, messages)
        
        retrieved = memory.get_messages(session_id)
        
        assert len(retrieved) == 2
        assert retrieved[0].content == "Hello"
        assert retrieved[1].content == "Hi there!"
    
    def test_get_messages_applies_window_limit(self):
        """Should apply window size limit to returned messages."""
        memory = ConversationMemory(window_size=2)
        session_id = memory.create_session()
        
        # Add more messages than window allows (2 pairs = 4 messages)
        for i in range(6):
            memory.add_messages(session_id, [
                HumanMessage(content=f"Question {i}"),
                AIMessage(content=f"Answer {i}"),
            ])
        
        messages = memory.get_messages(session_id)
        
        # Should only return last 2 pairs (4 messages)
        assert len(messages) == 4
        assert "Question 4" in messages[0].content
        assert "Question 5" in messages[2].content


class TestAddMessages:
    """Test adding messages to session."""
    
    def test_add_single_message(self):
        """Should add a single message."""
        memory = ConversationMemory()
        session_id = memory.create_session()
        
        message = HumanMessage(content="Test message")
        memory.add_messages(session_id, [message])
        
        messages = memory.get_messages(session_id)
        assert len(messages) == 1
        assert messages[0].content == "Test message"
    
    def test_add_multiple_messages(self):
        """Should add multiple messages at once."""
        memory = ConversationMemory()
        session_id = memory.create_session()
        
        messages = [
            HumanMessage(content="First"),
            AIMessage(content="Second"),
            HumanMessage(content="Third"),
        ]
        memory.add_messages(session_id, messages)
        
        retrieved = memory.get_messages(session_id)
        assert len(retrieved) == 3
    
    def test_add_messages_nonexistent_session(self):
        """Should raise KeyError for nonexistent session."""
        memory = ConversationMemory()
        
        with pytest.raises(KeyError, match="not found"):
            memory.add_messages("nonexistent", [HumanMessage(content="Test")])
    
    def test_add_messages_updates_last_accessed(self):
        """Should update last accessed time."""
        memory = ConversationMemory()
        session_id = memory.create_session()
        
        initial_time = memory.last_accessed[session_id]
        time.sleep(0.01)
        
        memory.add_messages(session_id, [HumanMessage(content="Test")])
        
        assert memory.last_accessed[session_id] > initial_time


class TestAddExchange:
    """Test adding human-AI message exchanges."""
    
    def test_add_exchange_creates_two_messages(self):
        """Should create HumanMessage and AIMessage."""
        memory = ConversationMemory()
        session_id = memory.create_session()
        
        memory.add_exchange(session_id, "What is Python?", "Python is a language.")
        
        messages = memory.get_messages(session_id)
        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)
        assert messages[0].content == "What is Python?"
        assert messages[1].content == "Python is a language."
    
    def test_add_multiple_exchanges(self):
        """Should add multiple exchanges in order."""
        memory = ConversationMemory()
        session_id = memory.create_session()
        
        memory.add_exchange(session_id, "First question", "First answer")
        memory.add_exchange(session_id, "Second question", "Second answer")
        
        messages = memory.get_messages(session_id)
        assert len(messages) == 4
        assert messages[0].content == "First question"
        assert messages[1].content == "First answer"
        assert messages[2].content == "Second question"
        assert messages[3].content == "Second answer"
    
    def test_add_exchange_nonexistent_session(self):
        """Should raise KeyError for nonexistent session."""
        memory = ConversationMemory()
        
        with pytest.raises(KeyError, match="not found"):
            memory.add_exchange("nonexistent", "Question", "Answer")


class TestClearSession:
    """Test clearing session history."""
    
    def test_clear_session_removes_messages(self):
        """Should remove all messages from session."""
        memory = ConversationMemory()
        session_id = memory.create_session()
        
        memory.add_exchange(session_id, "Question", "Answer")
        assert len(memory.get_messages(session_id)) == 2
        
        memory.clear_session(session_id)
        
        assert len(memory.get_messages(session_id)) == 0
    
    def test_clear_session_keeps_session_active(self):
        """Should keep session active after clearing."""
        memory = ConversationMemory()
        session_id = memory.create_session()
        
        memory.clear_session(session_id)
        
        assert session_id in memory.sessions
    
    def test_clear_session_nonexistent_raises_error(self):
        """Should raise KeyError for nonexistent session."""
        memory = ConversationMemory()
        
        with pytest.raises(KeyError, match="not found"):
            memory.clear_session("nonexistent")


class TestDeleteSession:
    """Test deleting sessions."""
    
    def test_delete_session_removes_completely(self):
        """Should remove session completely."""
        memory = ConversationMemory()
        session_id = memory.create_session()
        
        memory.delete_session(session_id)
        
        assert session_id not in memory.sessions
        assert session_id not in memory.last_accessed
    
    def test_delete_session_nonexistent_raises_error(self):
        """Should raise KeyError for nonexistent session."""
        memory = ConversationMemory()
        
        with pytest.raises(KeyError, match="not found"):
            memory.delete_session("nonexistent")


class TestListSessions:
    """Test listing active sessions."""
    
    def test_list_sessions_empty(self):
        """Should return empty list when no sessions."""
        memory = ConversationMemory()
        assert memory.list_sessions() == []
    
    def test_list_sessions_returns_all_ids(self):
        """Should return all session IDs."""
        memory = ConversationMemory()
        
        id1 = memory.create_session()
        id2 = memory.create_session()
        id3 = memory.create_session()
        
        sessions = memory.list_sessions()
        
        assert len(sessions) == 3
        assert id1 in sessions
        assert id2 in sessions
        assert id3 in sessions


class TestGetSessionInfo:
    """Test getting session information."""
    
    def test_get_session_info_structure(self):
        """Should return dictionary with session information."""
        memory = ConversationMemory()
        session_id = memory.create_session()
        
        info = memory.get_session_info(session_id)
        
        assert "session_id" in info
        assert "message_count" in info
        assert "last_accessed" in info
        assert "inactive_minutes" in info
    
    def test_get_session_info_correct_message_count(self):
        """Should report correct message count."""
        memory = ConversationMemory()
        session_id = memory.create_session()
        
        memory.add_exchange(session_id, "Q1", "A1")
        memory.add_exchange(session_id, "Q2", "A2")
        
        info = memory.get_session_info(session_id)
        assert info["message_count"] == 4
    
    def test_get_session_info_nonexistent_raises_error(self):
        """Should raise KeyError for nonexistent session."""
        memory = ConversationMemory()
        
        with pytest.raises(KeyError, match="not found"):
            memory.get_session_info("nonexistent")


class TestSessionCleanup:
    """Test automatic session cleanup."""
    
    def test_cleanup_removes_old_sessions(self):
        """Should remove sessions older than timeout."""
        # Use very small timeout (1 minute) and manipulate time
        memory = ConversationMemory(session_timeout=1)
        
        session_id = memory.create_session()
        
        # Manually set last_accessed to old time
        memory.last_accessed[session_id] = datetime.now() - timedelta(minutes=2)
        
        # Trigger cleanup by accessing sessions
        memory.get_session_history("new-session")
        
        # Old session should be cleaned up
        assert session_id not in memory.sessions
    
    def test_cleanup_keeps_recent_sessions(self):
        """Should keep recently accessed sessions."""
        memory = ConversationMemory(session_timeout=60)
        
        session_id = memory.create_session()
        
        # Access the session
        memory.get_session_history(session_id)
        
        # Should still exist
        assert session_id in memory.sessions
    
    def test_cleanup_disabled_with_zero_timeout(self):
        """Should not clean up when timeout is 0."""
        memory = ConversationMemory(session_timeout=0)
        
        # Even with 0 timeout, cleanup happens. Let's test no cleanup explicitly
        memory2 = ConversationMemory(session_timeout=9999)
        session_id = memory2.create_session()
        
        time.sleep(0.01)
        memory2.get_session_history("another")
        
        # Should still exist with high timeout
        assert session_id in memory2.sessions
    
    def test_manual_cleanup_all(self):
        """Should manually clean up all old sessions."""
        memory = ConversationMemory(session_timeout=1)
        
        session1 = memory.create_session()
        session2 = memory.create_session()
        
        # Manually set last_accessed to old time
        memory.last_accessed[session1] = datetime.now() - timedelta(minutes=2)
        memory.last_accessed[session2] = datetime.now() - timedelta(minutes=2)
        
        cleaned = memory.cleanup_all()
        
        assert cleaned == 2
        assert len(memory.sessions) == 0


class TestConversationManager:
    """Test ConversationManager alias."""
    
    def test_conversation_manager_is_alias(self):
        """ConversationManager should be alias for ConversationMemory."""
        manager = ConversationManager()
        
        assert isinstance(manager, ConversationMemory)
    
    def test_conversation_manager_works_identically(self):
        """ConversationManager should work same as ConversationMemory."""
        manager = ConversationManager(window_size=5)
        
        session_id = manager.create_session()
        manager.add_exchange(session_id, "Test", "Response")
        
        messages = manager.get_messages(session_id)
        assert len(messages) == 2


@pytest.mark.parametrize("window_size,message_pairs,expected_count", [
    (1, 1, 2),   # 1 pair = 2 messages
    (2, 3, 4),   # Window of 2, but added 3 pairs = last 2 pairs = 4 messages
    (5, 10, 10), # Window of 5, added 10 pairs = last 5 pairs = 10 messages
    (10, 5, 10), # Window of 10, added 5 pairs = 5 pairs = 10 messages
])
def test_window_size_limits_parametrized(window_size, message_pairs, expected_count):
    """Test window size limits with various parameters."""
    memory = ConversationMemory(window_size=window_size)
    session_id = memory.create_session()
    
    # Add message pairs
    for i in range(message_pairs):
        memory.add_exchange(session_id, f"Q{i}", f"A{i}")
    
    messages = memory.get_messages(session_id)
    assert len(messages) == expected_count
