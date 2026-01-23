"""
Unit tests for prompt templates module.

Tests all prompt templates, formatting functions, and template validation.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate

from src.chat.prompts import (
    CHAT_PROMPT_TEMPLATE,
    CITATION_EXTRACTION_PROMPT,
    CONDENSE_QUESTION_PROMPT,
    QA_PROMPT,
    RAG_CHAT_PROMPT,
    RAG_PROMPT,
    SYSTEM_PROMPT,
    format_chat_history,
    format_docs_for_context,
)


class TestPromptTemplates:
    """Test prompt template definitions."""
    
    def test_system_prompt_exists(self):
        """System prompt should be defined."""
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 0
        assert "context" in SYSTEM_PROMPT.lower()
        assert "cite" in SYSTEM_PROMPT.lower() or "source" in SYSTEM_PROMPT.lower()
    
    def test_condense_question_prompt_is_prompt_template(self):
        """Condense question prompt should be a PromptTemplate."""
        assert isinstance(CONDENSE_QUESTION_PROMPT, PromptTemplate)
    
    def test_condense_question_prompt_has_required_variables(self):
        """Condense question prompt should have required input variables."""
        required_vars = {"chat_history", "question"}
        assert required_vars.issubset(set(CONDENSE_QUESTION_PROMPT.input_variables))
    
    def test_qa_prompt_is_prompt_template(self):
        """QA prompt should be a PromptTemplate."""
        assert isinstance(QA_PROMPT, PromptTemplate)
    
    def test_qa_prompt_has_required_variables(self):
        """QA prompt should have required input variables."""
        required_vars = {"context", "question"}
        assert required_vars.issubset(set(QA_PROMPT.input_variables))
    
    def test_rag_prompt_is_prompt_template(self):
        """RAG prompt should be a PromptTemplate."""
        assert isinstance(RAG_PROMPT, PromptTemplate)
    
    def test_rag_prompt_has_required_variables(self):
        """RAG prompt should have context and question variables."""
        required_vars = {"context", "question"}
        assert required_vars.issubset(set(RAG_PROMPT.input_variables))
    
    def test_rag_chat_prompt_is_prompt_template(self):
        """RAG chat prompt should be a PromptTemplate."""
        assert isinstance(RAG_CHAT_PROMPT, PromptTemplate)
    
    def test_rag_chat_prompt_has_required_variables(self):
        """RAG chat prompt should have all required variables."""
        required_vars = {"context", "chat_history", "question"}
        assert required_vars.issubset(set(RAG_CHAT_PROMPT.input_variables))
    
    def test_citation_extraction_prompt_is_prompt_template(self):
        """Citation extraction prompt should be a PromptTemplate."""
        assert isinstance(CITATION_EXTRACTION_PROMPT, PromptTemplate)
    
    def test_citation_extraction_prompt_has_required_variables(self):
        """Citation extraction prompt should have response and documents variables."""
        required_vars = {"response", "documents"}
        assert required_vars.issubset(set(CITATION_EXTRACTION_PROMPT.input_variables))


class TestCondenseQuestionPrompt:
    """Test condense question prompt formatting."""
    
    def test_format_with_simple_history(self):
        """Should format prompt with simple chat history."""
        formatted = CONDENSE_QUESTION_PROMPT.format(
            chat_history="Human: What is Python?\nAI: Python is a programming language.",
            question="Tell me more",
        )
        
        assert "What is Python?" in formatted
        assert "Python is a programming language" in formatted
        assert "Tell me more" in formatted
        assert "Standalone Question:" in formatted
    
    def test_format_with_empty_history(self):
        """Should handle empty chat history."""
        formatted = CONDENSE_QUESTION_PROMPT.format(
            chat_history="",
            question="What is RAG?",
        )
        
        assert "What is RAG?" in formatted
        assert "Standalone Question:" in formatted
    
    def test_format_with_multiline_history(self):
        """Should handle multiline chat history."""
        history = """Human: What is machine learning?
AI: Machine learning is a subset of AI.
Human: How does it work?
AI: It learns patterns from data."""
        
        formatted = CONDENSE_QUESTION_PROMPT.format(
            chat_history=history,
            question="Can you give examples?",
        )
        
        assert "machine learning" in formatted.lower()
        assert "Can you give examples?" in formatted


class TestQAPrompt:
    """Test QA prompt formatting."""
    
    def test_format_with_context_and_question(self):
        """Should format QA prompt with context and question."""
        context = "Python is a high-level programming language."
        question = "What is Python?"
        
        formatted = QA_PROMPT.format(context=context, question=question)
        
        assert context in formatted
        assert question in formatted
        assert "Answer:" in formatted
    
    def test_format_with_multiline_context(self):
        """Should handle multiline context."""
        context = """Document 1: Python basics
Python is versatile.

Document 2: Python uses
Used for web development."""
        
        formatted = QA_PROMPT.format(
            context=context,
            question="What is Python used for?",
        )
        
        assert "Document 1" in formatted
        assert "Document 2" in formatted
        assert "Python is versatile" in formatted


class TestRAGChatPrompt:
    """Test RAG chat prompt formatting."""
    
    def test_format_with_all_variables(self):
        """Should format with context, history, and question."""
        formatted = RAG_CHAT_PROMPT.format(
            context="Python is a programming language.",
            chat_history="Human: Hello\nAI: Hi there!",
            question="What is Python?",
        )
        
        assert "Python is a programming language" in formatted
        assert "Hello" in formatted
        assert "What is Python?" in formatted
    
    def test_format_with_empty_history(self):
        """Should handle empty chat history."""
        formatted = RAG_CHAT_PROMPT.format(
            context="Python is great.",
            chat_history="",
            question="Why?",
        )
        
        assert "Python is great" in formatted
        assert "Why?" in formatted


class TestFormatDocsForContext:
    """Test document formatting for context."""
    
    def test_format_single_document(self):
        """Should format a single document."""
        docs = [
            Document(
                page_content="Python is a programming language.",
                metadata={"source": "python_guide.txt"},
            )
        ]
        
        formatted = format_docs_for_context(docs)
        
        assert "Document 1" in formatted
        assert "python_guide.txt" in formatted
        assert "Python is a programming language" in formatted
    
    def test_format_multiple_documents(self):
        """Should format multiple documents with separators."""
        docs = [
            Document(
                page_content="Python basics.",
                metadata={"source": "basics.txt"},
            ),
            Document(
                page_content="Advanced Python.",
                metadata={"source": "advanced.txt"},
            ),
        ]
        
        formatted = format_docs_for_context(docs)
        
        assert "Document 1" in formatted
        assert "Document 2" in formatted
        assert "basics.txt" in formatted
        assert "advanced.txt" in formatted
        assert "---" in formatted  # Separator
    
    def test_format_with_page_metadata(self):
        """Should include page number in header."""
        docs = [
            Document(
                page_content="Content here.",
                metadata={"source": "book.pdf", "page": 42},
            )
        ]
        
        formatted = format_docs_for_context(docs)
        
        assert "book.pdf" in formatted
        assert "Page 42" in formatted
    
    def test_format_with_section_metadata(self):
        """Should include section in header."""
        docs = [
            Document(
                page_content="Section content.",
                metadata={"source": "doc.md", "section": "Introduction"},
            )
        ]
        
        formatted = format_docs_for_context(docs)
        
        assert "doc.md" in formatted
        assert "Introduction" in formatted
    
    def test_format_with_all_metadata(self):
        """Should include all metadata fields."""
        docs = [
            Document(
                page_content="Complete metadata example.",
                metadata={
                    "source": "complete.pdf",
                    "page": 10,
                    "section": "Chapter 1",
                },
            )
        ]
        
        formatted = format_docs_for_context(docs)
        
        assert "complete.pdf" in formatted
        assert "Page 10" in formatted
        assert "Chapter 1" in formatted
    
    def test_format_with_missing_source(self):
        """Should handle missing source metadata."""
        docs = [
            Document(
                page_content="No source metadata.",
                metadata={},
            )
        ]
        
        formatted = format_docs_for_context(docs)
        
        assert "Unknown" in formatted
        assert "No source metadata" in formatted
    
    def test_format_empty_list(self):
        """Should handle empty document list."""
        formatted = format_docs_for_context([])
        assert formatted == ""


class TestFormatChatHistory:
    """Test chat history formatting."""
    
    def test_format_single_exchange(self):
        """Should format a single human-AI exchange."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]
        
        formatted = format_chat_history(messages)
        
        assert "Human: Hello" in formatted
        assert "AI: Hi there!" in formatted
    
    def test_format_multiple_exchanges(self):
        """Should format multiple exchanges."""
        messages = [
            HumanMessage(content="What is Python?"),
            AIMessage(content="Python is a programming language."),
            HumanMessage(content="Is it easy to learn?"),
            AIMessage(content="Yes, it's beginner-friendly."),
        ]
        
        formatted = format_chat_history(messages)
        
        assert "Human: What is Python?" in formatted
        assert "AI: Python is a programming language" in formatted
        assert "Human: Is it easy to learn?" in formatted
        assert "AI: Yes, it's beginner-friendly" in formatted
    
    def test_format_preserves_order(self):
        """Should preserve message order."""
        messages = [
            HumanMessage(content="First"),
            AIMessage(content="Second"),
            HumanMessage(content="Third"),
        ]
        
        formatted = format_chat_history(messages)
        lines = formatted.split("\n")
        
        assert "First" in lines[0]
        assert "Second" in lines[1]
        assert "Third" in lines[2]
    
    def test_format_empty_list(self):
        """Should handle empty message list."""
        formatted = format_chat_history([])
        assert formatted == ""
    
    def test_format_with_multiline_content(self):
        """Should handle messages with multiline content."""
        messages = [
            HumanMessage(content="Line 1\nLine 2"),
            AIMessage(content="Response line 1\nResponse line 2"),
        ]
        
        formatted = format_chat_history(messages)
        
        assert "Human: Line 1\nLine 2" in formatted
        assert "AI: Response line 1\nResponse line 2" in formatted


class TestChatPromptTemplate:
    """Test chat prompt template with message placeholders."""
    
    def test_chat_prompt_template_exists(self):
        """Chat prompt template should be defined."""
        assert CHAT_PROMPT_TEMPLATE is not None
    
    def test_chat_prompt_template_has_system_message(self):
        """Chat prompt template should include system message."""
        messages = CHAT_PROMPT_TEMPLATE.messages
        assert len(messages) > 0
        # First message should be system message
        assert messages[0].prompt.template == SYSTEM_PROMPT


@pytest.mark.parametrize(
    "context,question,expected_in_result",
    [
        (
            "Python is easy.",
            "What is Python?",
            ["Python is easy", "What is Python?"],
        ),
        (
            "RAG combines retrieval and generation.",
            "Explain RAG",
            ["RAG combines retrieval", "Explain RAG"],
        ),
        (
            "LangChain is a framework.",
            "What is LangChain?",
            ["LangChain is a framework", "What is LangChain?"],
        ),
    ],
)
def test_rag_prompt_formatting_parametrized(context, question, expected_in_result):
    """Test RAG prompt formatting with various inputs."""
    formatted = RAG_PROMPT.format(context=context, question=question)
    
    for expected in expected_in_result:
        assert expected in formatted
