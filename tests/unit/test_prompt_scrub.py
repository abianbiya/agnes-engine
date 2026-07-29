"""Checks for source-mention scrubbing and context cleanup."""

from langchain_core.documents import Document

from src.chat.prompts import format_docs_for_context, strip_source_mentions


def test_strips_citation_markers():
    leaked = (
        "Berdasarkan dokumen tersebut, UKT dibayar tiap semester [Document 2]. "
        "Pendaftaran dibuka Mei (Sumber: FAQ Unnes Apr 30). "
        "Detailnya ada pada baris 12.\n"
        "Sumber: FAQ Database Unnes Jun 06.txt"
    )
    clean = strip_source_mentions(leaked)

    for banned in ["Document", "Sumber", "dokumen tersebut", "baris 12", "["]:
        assert banned not in clean, f"{banned!r} survived: {clean!r}"
    assert "UKT dibayar tiap semester" in clean
    assert "Pendaftaran dibuka Mei" in clean


def test_keeps_legitimate_document_talk():
    real = "Upload dokumen persyaratan satu per satu agar tidak gagal."
    assert strip_source_mentions(real) == real


def test_context_drops_meta_and_duplicates():
    docs = [
        Document(page_content="This document contains a collection of FAQs.\nUKT dibayar tiap semester."),
        Document(page_content="UKT dibayar tiap semester."),
        Document(page_content="Beasiswa KIP dibuka Mei."),
    ]
    ctx = format_docs_for_context(docs)

    assert "This document contains" not in ctx
    assert ctx.count("UKT dibayar tiap semester") == 1
    assert "Beasiswa KIP dibuka Mei." in ctx


if __name__ == "__main__":
    test_strips_citation_markers()
    test_keeps_legitimate_document_talk()
    test_context_drops_meta_and_duplicates()
    test_grounding_rule_comes_after_context()
    print("ok")


def test_grounding_rule_comes_after_context():
    """Gemma3 follows the last instruction; before the context it gets outvoted."""
    from src.chat.prompts import GROUNDING_RULE, RAG_CHAT_TEMPLATE, RAG_TEMPLATE

    for tmpl in (RAG_TEMPLATE, RAG_CHAT_TEMPLATE):
        assert tmpl.index(GROUNDING_RULE) > tmpl.index("{context}")
        assert tmpl.index(GROUNDING_RULE) > tmpl.index("{question}")
