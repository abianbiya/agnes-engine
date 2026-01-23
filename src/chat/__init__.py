"""
Chat module for conversational RAG.

This module provides conversational AI capabilities with memory management,
prompt templates, and RAG chains for question-answering with source citations.
"""

from src.chat.chain import (
    ChatResponse,
    RAGChatChain,
    SimpleRAGChain,
    SourceDocument,
)
from src.chat.memory import (
    ConversationManager,
    ConversationMemory,
    create_runnable_with_history,
)
from src.chat.prompts import (
    CHAT_PROMPT_TEMPLATE,
    CONDENSE_QUESTION_PROMPT,
    QA_PROMPT,
    RAG_CHAT_PROMPT,
    RAG_PROMPT,
    SYSTEM_PROMPT,
    format_chat_history,
    format_docs_for_context,
)

__all__ = [
    # Chain components
    "RAGChatChain",
    "SimpleRAGChain",
    "ChatResponse",
    "SourceDocument",
    # Memory components
    "ConversationMemory",
    "ConversationManager",
    "create_runnable_with_history",
    # Prompt components
    "SYSTEM_PROMPT",
    "CONDENSE_QUESTION_PROMPT",
    "QA_PROMPT",
    "CHAT_PROMPT_TEMPLATE",
    "RAG_PROMPT",
    "RAG_CHAT_PROMPT",
    "format_chat_history",
    "format_docs_for_context",
]
