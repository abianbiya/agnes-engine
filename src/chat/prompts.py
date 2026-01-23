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


# System prompt with citation instructions
SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context from documents.

Your primary responsibilities:
1. Answer questions accurately using ONLY the information from the provided context
2. Always cite your sources by referencing the document name and page/section when available
3. If the context doesn't contain relevant information, acknowledge this clearly and don't make up answers
4. Be concise but thorough in your responses
5. Use proper formatting (markdown) when appropriate

When citing sources, use this format:
- For specific facts: "According to [document_name], ..."
- For page references: "As mentioned in [document_name] (page X), ..."
- For multiple sources: "This is supported by [doc1] and [doc2]..."

Remember: Accuracy and honesty are more important than providing an answer. If you're unsure or the context is insufficient, say so."""


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


# QA prompt with context and citation instructions
QA_TEMPLATE = """Answer the question based on the following context from documents. Include citations to source documents where applicable.

Context:
{context}

Question: {question}

Instructions:
- Use ONLY the information provided in the context above
- Cite specific documents when making claims
- If the context is insufficient, acknowledge this
- Format your answer clearly and concisely
- Include relevant quotes when they strengthen your answer

Answer:"""

QA_PROMPT = PromptTemplate.from_template(QA_TEMPLATE)


# Chat prompt template with system message and history
CHAT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


# RAG prompt with context for non-conversational use
RAG_TEMPLATE = """You are a helpful AI assistant that answers questions based on provided document context.

Context from documents:
{context}

User Question: {question}

Instructions:
- Answer based solely on the context provided
- Cite source documents when making claims (format: [document_name])
- If the context doesn't contain the answer, say "I don't have enough information in the provided documents to answer this question."
- Be accurate, concise, and helpful

Answer:"""

RAG_PROMPT = PromptTemplate.from_template(RAG_TEMPLATE)


# RAG chat prompt combining system message, context, history, and question
RAG_CHAT_TEMPLATE = """You are a helpful AI assistant that answers questions based on provided document context.

Context from documents:
{context}

Instructions:
- Answer the question using ONLY the provided context
- Cite sources by document name when making claims
- Acknowledge when information is not available in the context
- Be accurate and concise

Chat History:
{chat_history}

Question: {question}

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


# Helper function to format documents for context
def format_docs_for_context(docs: list) -> str:
    """
    Format retrieved documents into a context string for the prompt.
    
    Args:
        docs: List of documents with content and metadata
        
    Returns:
        Formatted context string
    """
    formatted_parts = []
    
    for i, doc in enumerate(docs, 1):
        # Extract metadata
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")
        section = doc.metadata.get("section")
        
        # Build document header
        header_parts = [f"[Document {i}: {source}"]
        if page is not None:
            header_parts.append(f"Page {page}")
        if section:
            header_parts.append(f"Section: {section}")
        header = ", ".join(header_parts) + "]"
        
        # Combine header and content
        formatted_parts.append(f"{header}\n{doc.page_content}")
    
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
