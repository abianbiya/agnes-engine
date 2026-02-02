"""
Prompt templates for the RAG chatbot.

This module defines all prompt templates used in the conversational RAG chain,
including system prompts, question condensing prompts, and QA prompts with
citation instructions.
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)


# System prompt for seamless document-based answers
SYSTEM_PROMPT = """You are Agnes (Artificial Guide of UNNES), customer service AI for Universitas Negeri Semarang.

Rules:
- Answer using ONLY the provided context
- Never mention sources, documents, or citations
- Answer in Bahasa Indonesia (or English if asked in English)
- Be friendly, professional, and concise
- If context is insufficient, say "Saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan ini"

Present answers naturally as your own knowledge."""


# Condense question prompt for follow-up questions
CONDENSE_QUESTION_TEMPLATE = """Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that captures all necessary context from the conversation.

The standalone question should:
1. Include relevant context from previous messages
2. Be self-contained and understandable without the conversation history
3. Preserve the original intent and specificity of the follow-up question
4. Not introduce new information not present in the original question or history

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)


# QA prompt for document-based seamless answers
QA_TEMPLATE = """Context:
{context}

Question: {question}

Answer using ONLY the context above. No citations. Bahasa Indonesia default (English if asked in English).

Answer:"""

QA_PROMPT = PromptTemplate.from_template(QA_TEMPLATE)


# Chat prompt template with system message and history
CHAT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


# RAG prompt for document-based seamless answers
RAG_TEMPLATE = """Context:
{context}

Question: {question}

Answer using ONLY the context. No sources mentioned. Bahasa Indonesia default (English if asked in English).

Answer:"""

RAG_PROMPT = PromptTemplate.from_template(RAG_TEMPLATE)


# RAG chat prompt for document-based conversational seamless answers
RAG_CHAT_TEMPLATE = """Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer using ONLY the context. No sources mentioned. Bahasa Indonesia default (English if asked in English).

Answer:"""

RAG_CHAT_PROMPT = PromptTemplate.from_template(RAG_CHAT_TEMPLATE)


# Prompt for extracting citations from generated responses
CITATION_EXTRACTION_TEMPLATE = """Extract source citations from the following AI response.

Response:
{response}

Context Documents:
{documents}

Extract all document references, quotes, or citations mentioned in the response. For each citation, provide:
1. Document name
2. Page number (if mentioned)
3. Section (if mentioned)
4. Quoted text or referenced content

Return the citations in JSON format as a list:
[
  {{
    "document": "filename",
    "page": 1,
    "section": "Introduction",
    "content": "quoted or referenced text"
  }}
]

If no citations are found, return an empty list: []

Citations:"""

CITATION_EXTRACTION_PROMPT = PromptTemplate.from_template(CITATION_EXTRACTION_TEMPLATE)


# Query expansion prompt for better semantic search
QUERY_EXPANSION_TEMPLATE = """Expand this search query to improve semantic search. Keep it focused and relevant.

Query: {query}

Rules:
1. Keep original terms
2. Add synonyms in the same language
3. Add English equivalents for non-English terms
4. Do NOT add unrelated topics
5. Keep expansion short (max 50 words)

Examples:
- "siapa rektor unnes" → "siapa rektor unnes dipimpin pemimpin rector UNNES Universitas Negeri Semarang pimpinan"
- "biaya kuliah" → "biaya kuliah UKT uang kuliah tunggal SPP tuition fee"

Expanded Query:"""

QUERY_EXPANSION_PROMPT = PromptTemplate.from_template(QUERY_EXPANSION_TEMPLATE)


# Helper function to format documents for context (simplified)
def format_docs_for_context(docs: list) -> str:
    """
    Format retrieved documents into a context string for the prompt.
    
    Args:
        docs: List of documents with content and metadata
        
    Returns:
        Formatted context string without source references
    """
    formatted_parts = []
    
    for doc in docs:
        # Just use the content without metadata references
        formatted_parts.append(doc.page_content)
    
    return "\n\n---\n\n".join(formatted_parts)


# Helper function to format chat history
def format_chat_history(messages: list) -> str:
    """
    Format chat history messages into a string for prompts.
    
    Args:
        messages: List of BaseMessage objects (HumanMessage, AIMessage)
        
    Returns:
        Formatted chat history string
    """
    formatted = []
    
    for msg in messages:
        role = "Human" if msg.type == "human" else "AI"
        formatted.append(f"{role}: {msg.content}")
    
    return "\n".join(formatted)


__all__ = [
    "SYSTEM_PROMPT",
    "CONDENSE_QUESTION_TEMPLATE",
    "CONDENSE_QUESTION_PROMPT",
    "QA_TEMPLATE",
    "QA_PROMPT",
    "CHAT_PROMPT_TEMPLATE",
    "RAG_TEMPLATE",
    "RAG_PROMPT",
    "RAG_CHAT_TEMPLATE",
    "RAG_CHAT_PROMPT",
    "CITATION_EXTRACTION_TEMPLATE",
    "CITATION_EXTRACTION_PROMPT",
    "QUERY_EXPANSION_TEMPLATE",
    "QUERY_EXPANSION_PROMPT",
    "format_docs_for_context",
    "format_chat_history",
]
