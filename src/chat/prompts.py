"""
Prompt templates for the RAG chatbot.

This module defines all prompt templates used in the conversational RAG chain,
including system prompts, question condensing prompts, and QA prompts.

Design note (ponytail): the prompts deliberately never spell out the shape we
do not want ("[Document 1]", "(Sumber: ...)"). Naming a forbidden format primes
the model to produce it. Instead the context is framed as Agnes' own knowledge,
and strip_source_mentions() is the deterministic safety net.
"""

import re

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)


# System prompt for seamless answers
SYSTEM_PROMPT = """You are Agnes (Artificial Guide of UNNES), customer service AI for Universitas Negeri Semarang.

The reference information you receive is your own knowledge. Answer from it and nothing else.
Write the answer itself: no preamble, no meta-commentary, no bracketed markers, no explanation of where the knowledge came from.
Reply in Bahasa Indonesia (English only if the question is in English), friendly and concrete, at most 5 sentences unless details are requested.
If the reference information does not cover the question, reply exactly: Saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan ini."""


# Condense question prompt for follow-up questions
CONDENSE_QUESTION_TEMPLATE = """Rewrite the follow-up question as a standalone question using the conversation. Keep the original language and intent, add no new facts, output the question only.

{chat_history}

Follow-up Question: {question}
Standalone Question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)


# QA prompt for seamless answers
QA_TEMPLATE = """Reference information:
{context}

Question: {question}

Answer directly, in the language of the question, using only the reference information:"""

QA_PROMPT = PromptTemplate.from_template(QA_TEMPLATE)


# Chat prompt template with system message and history
CHAT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


# ponytail: the grounding rule is repeated after the context on purpose.
# Instructions placed only before a long context get out-weighted, and gemma3
# then fills gaps from pretraining (it named a rector who left in 2022).
GROUNDING_RULE = """Answer using only the reference information above. Whatever you remember about UNNES from elsewhere is out of date and must not be used. If the reference information above does not answer the question, reply exactly: Saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan ini."""


# RAG prompt (one-off questions)
RAG_TEMPLATE = SYSTEM_PROMPT + """

Reference information:
{context}

Question: {question}

""" + GROUNDING_RULE + """
Answer:"""

RAG_PROMPT = PromptTemplate.from_template(RAG_TEMPLATE)


# RAG chat prompt (conversational)
RAG_CHAT_TEMPLATE = SYSTEM_PROMPT + """

Reference information:
{context}

{chat_history}

Question: {question}

""" + GROUNDING_RULE + """
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


# --- Source-mention scrubbing ---------------------------------------------
# Narrow on purpose: real FAQ answers legitimately talk about "dokumen"
# ("upload dokumen persyaratan"), so only unmistakable citation shapes are cut.

_REF_WORD = r"(?:dokumen|document|sumber|source|referensi|reference|konteks|context|kutipan|citation)"

_SCRUB_PATTERNS = [
    # [Document 1], [Dokumen 2: file.pdf], [Sumber: FAQ.txt], [Source 3]
    re.compile(rf"\[[^\]\n]*{_REF_WORD}[^\]\n]*\]", re.IGNORECASE),
    # bare footnote markers: [1], [2,3]
    re.compile(r"\[\s*\d+(?:\s*[,;-]\s*\d+)*\s*\]"),
    # (Sumber: ...), (Document 2), (lihat dokumen X), (hal. 3), (baris 12)
    re.compile(rf"\(\s*(?:{_REF_WORD}|lihat|see)[^)\n]*\)", re.IGNORECASE),
    re.compile(
        r"\(\s*(?:hal(?:aman)?\.?|page|baris|line|p\.)\s*\d+[^)\n]*\)", re.IGNORECASE
    ),
    # "Dokumen 1", "Document 2:" used as a label
    re.compile(rf"\b{_REF_WORD}\s*#?\s*\d+\s*:?", re.IGNORECASE),
    # "pada baris 12", "di halaman 3", "on line 45"
    re.compile(
        r"\b(?:pada|di|dari|on|in|at)\s+(?:baris|line|hal(?:aman)?\.?|page)\s*\d+\b",
        re.IGNORECASE,
    ),
    # leading attribution clause: "Berdasarkan dokumen tersebut, ..."
    re.compile(
        rf"^\s*(?:berdasarkan|menurut|sesuai(?:\s+dengan)?|sebagaimana|dari|based on|according to)\s+{_REF_WORD}\b[^,.\n]{{0,60}}[,.]\s*",
        re.IGNORECASE,
    ),
    # whole line that is nothing but a citation: "Sumber: FAQ.txt"
    re.compile(rf"^\s*{_REF_WORD}\s*:.*$", re.IGNORECASE | re.MULTILINE),
    # "Informasi ini tersedia di dokumen ...", "informasi tersebut terdapat pada konteks ..."
    re.compile(
        rf"\b(?:informasi|jawaban|hal)\s+(?:ini|tersebut|itu)\s+(?:tersedia|terdapat|dijelaskan|disebutkan|tercantum|diambil|berasal)[^.\n]*\b{_REF_WORD}\b[^.\n]*\.?",
        re.IGNORECASE,
    ),
]

_LEFTOVER_PUNCT = re.compile(r"[ \t]*([,.;:])(?=[ \t]*[,.;:])")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_BLANK = re.compile(r"\n{3,}")


def strip_source_mentions(text: str) -> str:
    """
    Remove citation/source markers the model may still leak into an answer.

    Args:
        text: Raw model output.

    Returns:
        Answer text without document, source, page or line references.
    """
    if not text:
        return text

    for pattern in _SCRUB_PATTERNS:
        text = pattern.sub("", text)

    text = _LEFTOVER_PUNCT.sub("", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    text = _MULTI_BLANK.sub("\n\n", text)

    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


# --- Context formatting ----------------------------------------------------

# Boilerplate that lives inside the ingested corpus itself and teaches the model
# to talk about "this document". Dropped before the context reaches the LLM.
_META_LINE = re.compile(
    r"^.*(?:this document contains|retrieval-augmented generation|rag chatbot|"
    r"dokumen ini berisi|information provided in this document).*$",
    re.IGNORECASE | re.MULTILINE,
)


def format_docs_for_context(docs: list) -> str:
    """
    Format retrieved documents into a context string for the prompt.

    Drops corpus meta-boilerplate and duplicate chunks (the UNNES FAQ repeats
    whole answers verbatim), which shortens the prompt and speeds up prefill.

    Args:
        docs: List of documents with content and metadata

    Returns:
        Formatted context string without source references
    """
    parts: list[str] = []
    seen: set[str] = set()

    for doc in docs:
        content = _META_LINE.sub("", doc.page_content).strip()
        if not content:
            continue

        key = " ".join(content.lower().split())
        if key in seen:
            continue
        seen.add(key)

        parts.append(content)

    return "\n\n---\n\n".join(parts)


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
    "GROUNDING_RULE",
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
    "strip_source_mentions",
    "format_docs_for_context",
    "format_chat_history",
]
