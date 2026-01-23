# Hybrid Search Implementation for Indonesian Language RAG Systems

## Improving Retrieval Accuracy through BM25 and Semantic Search Fusion

**Authors:** RAG Chatbot Research Team  
**Date:** December 25, 2025  
**Version:** 1.1

---

## Abstract

This document presents our findings on implementing hybrid search to solve retrieval failures in Retrieval-Augmented Generation (RAG) systems for Indonesian language queries. We demonstrate that combining BM25 keyword matching with semantic embedding search using Reciprocal Rank Fusion (RRF) significantly improves retrieval accuracy, particularly for queries where exact keyword matches are critical but semantic similarity alone fails to capture the relationship.

**Update (v1.1):** We also evaluate the impact of switching from `nomic-embed-text` to `bge-m3` embedding model, demonstrating significant improvements in semantic search accuracy for Indonesian language queries.

---

## 1. Introduction

### 1.1 Problem Statement

In our RAG chatbot implementation for UNNES (Universitas Negeri Semarang), we encountered a critical retrieval failure. The query **"siapa rektor unnes?"** (Who is the UNNES rector?) consistently failed to retrieve the document containing the answer, despite the document existing in the ChromaDB vector store.

**Document content (Tentang UNNES 1.txt):**
```
Siapa pimpinan UNNES saat ini?
Answer: Rektor UNNES saat ini adalah Prof. Dr. S. Martono, M.Si., 
seorang profesor di bidang manajemen.
```

### 1.2 Technical Environment

| Component | Technology | Version |
|-----------|------------|---------|
| Vector Store | ChromaDB | 1.4.0 |
| Embedding Model | nomic-embed-text → **bge-m3** (Ollama) | Latest |
| LLM | llama3.2 (Ollama) | Latest |
| Framework | LangChain | 1.2.0 |
| API | FastAPI | 0.127.0 |
| Distance Metric | Cosine | - |

### 1.3 Dataset Composition

| Document | Chunks | Description |
|----------|--------|-------------|
| FAQ Database Unnes Apr 30.txt | 153 | FAQ Q&A format |
| FAQ Database Unnes Jun 06.txt | 6 | FAQ Q&A format |
| Tentang UNNES 1.txt | 14 | General information (contains rector info) |
| Tentang UNNES 2.txt | 3 | General information |
| **Total** | **176** | - |

---

## 2. Problem Analysis

### 2.1 Semantic Search Failure

When using pure semantic search with `nomic-embed-text` embeddings, the query "siapa rektor unnes?" returned irrelevant documents:

**Table 1: Semantic Search Results for "siapa rektor unnes?"**

| Rank | Semantic Score | Source | Content Preview |
|------|----------------|--------|-----------------|
| 1 | 0.4684 | FAQ chunk 47 | "berapa biaya kuliah di unnes?" |
| 2 | 0.4671 | FAQ chunk 45 | "bagaimana cara mendapatkan LoA..." |
| 3 | 0.4666 | FAQ chunk 99 | "Bagaimana cara mendapatkan legalisir..." |
| 4 | 0.4642 | FAQ chunk 98 | "Apa akreditasi prodi..." |
| 5 | 0.4587 | FAQ chunk 29 | "Bagaimana cara mengirim surat?..." |

**Key Observation:** The document containing "Rektor UNNES saat ini adalah Prof. Dr. S. Martono" did not appear in the top 12 semantic search results.

### 2.2 Root Cause Analysis

The embedding model `nomic-embed-text` failed to capture the semantic relationship between:

- **Query terms:** "siapa" (who), "rektor" (rector)
- **Document terms:** "pimpinan" (leader), "Rektor" (Rector), "dipimpin" (led by)

This is a known limitation of embedding models, particularly for:
1. **Low-resource languages** like Indonesian
2. **Domain-specific terminology** (academic/institutional)
3. **Indirect semantic relationships** (siapa → pimpinan → rektor)

### 2.3 Failed Approaches

Before implementing hybrid search, we attempted several solutions:

| Approach | Result | Reason for Failure |
|----------|--------|-------------------|
| LLM Query Expansion | Failed | LLM responded conversationally instead of expanding |
| Keyword-based Query Expansion | Worse | Diluted semantic similarity with irrelevant terms |
| Increased k (15 results) | Failed | Rector document still not in results |
| Disabled MMR | No improvement | Problem is in initial retrieval, not reranking |
| Re-ingested documents | Failed | Embedding quality unchanged |

---

## 3. Solution: Hybrid Search with RRF

### 3.1 Architecture Overview

```
                    ┌─────────────────┐
                    │   User Query    │
                    │ "siapa rektor   │
                    │    unnes?"      │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │  BM25 Retriever │           │Semantic Retriever│
    │  (Keyword Match)│           │  (Embeddings)   │
    └────────┬────────┘           └────────┬────────┘
             │                             │
             │ Top-K Results               │ Top-K Results
             │                             │
             └──────────────┬──────────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │ Reciprocal Rank │
                  │ Fusion (RRF)    │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Final Ranked   │
                  │    Results      │
                  └─────────────────┘
```

### 3.2 BM25 Algorithm

BM25 (Best Matching 25) is a probabilistic retrieval function that ranks documents based on term frequency and inverse document frequency.

**Scoring Formula:**
```
score(D,Q) = Σ IDF(qi) · (f(qi,D) · (k1 + 1)) / (f(qi,D) + k1 · (1 - b + b · |D|/avgdl))
```

Where:
- `f(qi,D)` = term frequency of qi in document D
- `|D|` = document length
- `avgdl` = average document length
- `k1`, `b` = tuning parameters (typically k1=1.5, b=0.75)

### 3.3 Reciprocal Rank Fusion (RRF)

RRF combines results from multiple retrieval systems without requiring score normalization:

**Formula:**
```
RRF_score(d) = Σ (weight_i / (k + rank_i(d)))
```

Where:
- `k` = constant (default: 60) for smoothing
- `rank_i(d)` = rank of document d in retriever i
- `weight_i` = weight for retriever i

**Our Configuration:**
```python
HybridSearchConfig(
    semantic_weight=0.5,
    bm25_weight=0.5,
    k=5,              # Final results to return
    semantic_k=15,    # Candidates from semantic search
    bm25_k=15,        # Candidates from BM25
    rrf_k=60          # RRF smoothing constant
)
```

---

## 4. Implementation

### 4.1 BM25 Retriever

```python
# src/retrieval/bm25.py
from rank_bm25 import BM25Okapi

class BM25Retriever(LoggerMixin):
    def __init__(self, documents: list[Document], k: int = 4):
        self.k = k
        self._documents = documents
        self._tokenized_corpus = [
            self._tokenize(doc.page_content) for doc in documents
        ]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenization with lowercasing."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [token for token in text.split() if token]
    
    def retrieve(self, query: str) -> list[tuple[Document, float]]:
        query_tokens = self._tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        doc_scores = list(zip(self._documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[:self.k]
```

### 4.2 Hybrid Retriever with RRF

```python
# src/retrieval/hybrid.py
class HybridRetriever(LoggerMixin):
    async def retrieve(self, query: str) -> list[RetrievedDocument]:
        # Get results from both retrievers
        semantic_results = await self._semantic_search(query)
        bm25_results = self._bm25_search(query)
        
        # Combine using RRF
        combined = self._reciprocal_rank_fusion(semantic_results, bm25_results)
        return combined[:self.config.k]
    
    def _reciprocal_rank_fusion(
        self,
        semantic_results: list[tuple[Document, float]],
        bm25_results: list[tuple[Document, float]],
    ) -> list[tuple[Document, float]]:
        k = self.config.rrf_k  # 60
        doc_scores: dict[str, tuple[Document, float]] = {}
        
        # Process semantic results
        for rank, (doc, _) in enumerate(semantic_results, start=1):
            content_key = doc.page_content
            rrf_score = self.config.semantic_weight * (1.0 / (k + rank))
            if content_key in doc_scores:
                doc_scores[content_key] = (
                    doc_scores[content_key][0],
                    doc_scores[content_key][1] + rrf_score
                )
            else:
                doc_scores[content_key] = (doc, rrf_score)
        
        # Process BM25 results
        for rank, (doc, _) in enumerate(bm25_results, start=1):
            content_key = doc.page_content
            rrf_score = self.config.bm25_weight * (1.0 / (k + rank))
            if content_key in doc_scores:
                doc_scores[content_key] = (
                    doc_scores[content_key][0],
                    doc_scores[content_key][1] + rrf_score
                )
            else:
                doc_scores[content_key] = (doc, rrf_score)
        
        # Sort by combined score
        return sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
```

---

## 5. Results

### 5.1 BM25 Search Results

**Table 2: BM25 Results for "siapa rektor unnes?"**

| Rank | BM25 Score | Source | Content Preview |
|------|------------|--------|-----------------|
| **1** | **7.9628** | Tentang UNNES (chunk 0) | "Siapa pimpinan UNNES saat ini? **Rektor UNNES saat ini adalah Prof. Dr. S. Martono, M.Si.**" |
| **2** | **5.6885** | Tentang UNNES (chunk 9) | "**Rektor Universitas Negeri Semarang saat ini adalah Prof. Dr. S Martono M.Si.**" |
| 3 | 5.0015 | FAQ (chunk 47) | "berapa biaya kuliah di **unnes**?" |
| 4 | 4.8076 | FAQ (chunk 1) | "Peraturan **Rektor** Nomor 11 Tahun 2023..." |
| 5 | 4.8076 | FAQ (chunk 3) | "Peraturan **Rektor** Nomor 11 Tahun 2023..." |

**Key Finding:** BM25 successfully ranked the correct documents at positions #1 and #2 by matching the keyword "rektor" directly.

### 5.2 Comparison: Semantic vs BM25 vs Hybrid

**Table 3: Retrieval Method Comparison**

| Method | Correct Doc in Top-5? | Correct Doc Rank | Answer Retrieved? |
|--------|----------------------|------------------|-------------------|
| Semantic Only | ❌ No | Not found | ❌ No |
| BM25 Only | ✅ Yes | #1 | ✅ Yes |
| **Hybrid (RRF)** | ✅ Yes | #3 | ✅ Yes |

### 5.3 Final System Response

**Query:** "siapa rektor unnes?"

**Response:**
```json
{
    "answer": "Rektor UNNES saat ini adalah Prof. Dr. S. Martono, M.Si., 
               seorang profesor di bidang manajemen",
    "sources": [
        {
            "filename": "Tentang UNNES 1.txt",
            "relevance_score": 0.00819,
            "content_preview": "FAQ – Informasi Umum Tentang UNNES...
                               Siapa pimpinan UNNES saat ini?
                               Answer: Rektor UNNES saat ini adalah 
                               Prof. Dr. S. Martono, M.Si...."
        }
    ]
}
```

---

## 6. Embedding Model Comparison: nomic-embed-text vs bge-m3

### 6.1 Motivation for Model Change

Our initial hypothesis was that the semantic search failure was due to limited Indonesian language support in `nomic-embed-text`. To validate this, we tested `bge-m3` (BAAI General Embedding - Multilingual), a state-of-the-art multilingual embedding model with explicit support for 100+ languages including Indonesian.

### 6.2 Model Specifications

| Feature | nomic-embed-text | bge-m3 |
|---------|------------------|--------|
| **Dimensions** | 768 | 1024 |
| **Max Tokens** | 8192 | 8192 |
| **Languages** | English-focused | 100+ languages |
| **Indonesian Support** | Limited | Explicit |
| **Model Size** | 274 MB | ~2.3 GB |
| **Architecture** | BERT-based | RetroMAE + UniLM |

### 6.3 Semantic Search Results Comparison

**Query: "siapa rektor unnes?"**

**Table 4: nomic-embed-text Results (FAILED)**

| Rank | Score | Content | Correct? |
|------|-------|---------|----------|
| 1 | 0.4684 | "berapa biaya kuliah di unnes?" | ❌ |
| 2 | 0.4671 | "bagaimana cara mendapatkan LoA..." | ❌ |
| 3 | 0.4666 | "Bagaimana cara mendapatkan legalisir..." | ❌ |
| 4 | 0.4642 | "Apa akreditasi prodi..." | ❌ |
| 5 | 0.4587 | "Bagaimana cara mengirim surat?" | ❌ |

**Table 5: bge-m3 Results (SUCCESS)**

| Rank | Score | Content | Correct? |
|------|-------|---------|----------|
| **1** | **0.4889** | **"Rektor Universitas Negeri Semarang saat ini adalah Prof. Dr. S Martono M.Si."** | ✅ |
| **2** | **0.4837** | **"FAQ – Informasi Umum Tentang UNNES... Siapa pimpinan UNNES?"** | ✅ |
| **3** | **0.4737** | **"dipimpin oleh Rektor Prof. Dr. S. Martono"** | ✅ |
| 4 | 0.4186 | "Apa status hukum UNNES?" | ❌ |
| 5 | 0.4052 | "Bagaimana cara mengirim surat? (rektor)" | Partial |

### 6.4 Key Findings

1. **bge-m3 correctly identifies the rector document at Rank #1** with a semantic score of 0.4889, while nomic-embed-text completely missed it.

2. **Top 3 results are all relevant** with bge-m3:
   - Rank #1: Direct answer about the rector
   - Rank #2: FAQ containing "Siapa pimpinan UNNES?"
   - Rank #3: Document mentioning "dipimpin oleh Rektor"

3. **Semantic understanding improved:**
   - bge-m3 correctly associates "siapa rektor" with "Rektor saat ini adalah..."
   - bge-m3 understands the synonym relationship: "rektor" ↔ "pimpinan"

4. **Score distribution is similar** but the ranking is dramatically different, indicating better semantic matching rather than just higher confidence.

### 6.5 Hybrid Search with bge-m3

With bge-m3, both semantic search AND BM25 now retrieve the correct document:

| Retriever | Rank | Score | Content |
|-----------|------|-------|---------|
| Semantic | #1 | 0.4889 | "Rektor UNNES saat ini adalah Prof. Dr. S Martono" |
| BM25 | #1 | 7.9628 | "Siapa pimpinan UNNES?... Rektor UNNES saat ini adalah..." |
| BM25 | #2 | 5.6885 | "Rektor Universitas Negeri Semarang saat ini adalah..." |

**RRF Fusion Result:** The correct document appears in **both** top results, receiving boosted scores from RRF fusion.

### 6.6 Recommendation

**For Indonesian language RAG systems, we strongly recommend using `bge-m3` over `nomic-embed-text`.**

| Aspect | Recommendation |
|--------|----------------|
| **Embedding Model** | bge-m3 (or similar multilingual model) |
| **Hybrid Search** | Still recommended for robustness |
| **BM25 Weight** | Can potentially reduce from 0.5 to 0.3 with bge-m3 |

---

## 7. Performance Analysis

### 6.1 Latency Breakdown

| Component | Time (ms) | Notes |
|-----------|-----------|-------|
| BM25 Index Loading | ~100 | Load 176 documents from ChromaDB |
| BM25 Search | ~10 | In-memory search |
| Semantic Search | ~700 | Embedding generation + vector search |
| RRF Fusion | ~1 | Simple score combination |
| **Total Retrieval** | **~800** | Acceptable for interactive use |

### 6.2 Memory Usage

| Component | Memory |
|-----------|--------|
| BM25 Index (176 docs) | ~2 MB |
| ChromaDB Connection | Shared |
| Per-request overhead | Minimal |

---

## 8. Discussion

### 8.1 Why Hybrid Search Works

1. **Complementary Strengths:**
   - BM25 excels at exact keyword matching ("rektor" → "Rektor")
   - Semantic search captures meaning even with different words

2. **RRF Benefits:**
   - Documents appearing in both result sets get boosted
   - No need for score normalization between different systems
   - Robust to outlier scores

3. **Indonesian Language Considerations:**
   - Indonesian has relatively consistent morphology
   - Keyword matching is effective for formal terms (rektor, universitas)
   - Semantic models have limited Indonesian training data

### 8.2 When to Use Hybrid Search

| Scenario | Recommended Approach |
|----------|---------------------|
| Domain-specific terminology | Hybrid (BM25 + Semantic) |
| Low-resource languages | Hybrid (BM25 + Semantic) |
| General English queries | Semantic may suffice |
| Exact phrase matching | BM25 primary |
| Conceptual similarity | Semantic primary |

### 8.3 Limitations

1. **BM25 Index Size:** Scales linearly with document count
2. **Index Freshness:** BM25 index must be rebuilt when documents change
3. **Tokenization:** Simple whitespace tokenization may miss compound terms
4. **Weight Tuning:** Optimal weights may vary by domain

---

## 9. Recommendations

### 9.1 For Production Deployment

1. **Implement BM25 caching** with invalidation on document ingestion
2. **Monitor retrieval quality** with test queries
3. **Consider weight tuning** based on query patterns
4. **Add stemming/lemmatization** for Indonesian if needed

### 9.2 Future Improvements

1. ~~**Multilingual Embeddings:** Try `multilingual-e5-large` or `paraphrase-multilingual-MiniLM`~~ **DONE: Switched to bge-m3**
2. **Cross-encoder Reranking:** Add a second-stage reranker for precision
3. **Query Classification:** Route queries to optimal retriever based on type
4. **Learned Sparse Retrieval:** Consider SPLADE for better term matching
5. **Weight Tuning:** With bge-m3, consider reducing BM25 weight since semantic search is now effective

---

## 10. Conclusion

This research demonstrates that hybrid search combining BM25 keyword matching with semantic embedding search significantly improves retrieval accuracy for Indonesian language RAG systems. The implementation successfully resolved the retrieval failure for the query "siapa rektor unnes?" by leveraging BM25's ability to match the keyword "rektor" directly in documents.

**Update (v1.1):** Switching from `nomic-embed-text` to `bge-m3` dramatically improved semantic search accuracy. With bge-m3, the correct document now ranks #1 in semantic search results, validating our hypothesis that the original failure was due to limited Indonesian language support in the embedding model.

**Key Takeaways:**
1. Pure semantic search can fail for domain-specific terminology in low-resource languages
2. BM25 provides a robust fallback for keyword-critical queries
3. RRF effectively combines results without complex score normalization
4. The hybrid approach adds minimal latency (~100ms) for significant accuracy gains
5. **Multilingual embedding models (bge-m3) significantly outperform English-focused models for Indonesian queries**
6. **Even with improved embeddings, hybrid search remains valuable for robustness**

---

## 11. References

1. Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333-389.

2. Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods. *SIGIR '09*.

3. Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP 2020*.

4. Nogueira, R., & Cho, K. (2019). Passage Re-ranking with BERT. *arXiv preprint arXiv:1901.04085*.

5. LangChain Documentation. (2024). Retrievers - Hybrid Search. https://python.langchain.com/docs/modules/data_connection/retrievers/

6. Chen, J., et al. (2024). BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. *arXiv preprint arXiv:2402.03216*.

---

## Appendix A: File Structure

```
src/retrieval/
├── __init__.py          # Module exports
├── retriever.py         # Base RAGRetriever (semantic only)
├── bm25.py              # BM25Retriever implementation
├── hybrid.py            # HybridRetriever with RRF
└── reranker.py          # Optional cross-encoder reranker
```

## Appendix B: Configuration

```python
# src/api/dependencies.py
def get_retriever(settings, vectorstore) -> HybridRAGRetriever:
    return HybridRAGRetriever(
        vectorstore=vectorstore,
        k=settings.retrieval.retrieval_k,
        semantic_weight=0.5,
        bm25_weight=0.5,
    )
```

## Appendix C: Dependencies

```toml
# pyproject.toml
dependencies = [
    "langchain>=0.3.0",
    "langchain-chroma>=0.1.0",
    "chromadb>=0.5.0",
    "rank-bm25>=0.2.2",  # BM25 implementation
    # ... other dependencies
]
```

---

**Document Version History:**

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-25 | Initial release |
| 1.1 | 2025-12-25 | Added bge-m3 embedding model comparison (Section 6) |
