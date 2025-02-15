# Simple/Local Code RAG Example

## Getting Started

1. git clone git@github.com:cyber-nic/code-rag-spike.git
2. cd code-rag-spike
3. Configure embedding provider: Ollama or VoyageAI. Note: Ollama performs faster. See below.
4. Run

```
# default to current path; prompt for user query
go run .

# run in specified path; prompt for user query
go run . /some/path

# run in specified path; specified user query
go run . /some/path "my custom user query"
```

5. Reset
   To fully reset, simply delete the local.db file.

### Ollama

- Install Ollama.
- Pull the model:

```
ollama pull unclemusclez/jina-embeddings-v2-base-code
```

### Voyage AI

- Get VoyageAI API key from www.voyageai.com
- Store API key in a file eg. `/path/to/.voyageai-api.key`
- Export file path as environment variable

```
export VOYAGE_API_KEY_FILE=/path/to/.voyageai-api.key
```

- Update the code (comment/uncomment) to select VoyageAI

```
q, _, err := emb.Get(ctx, query)        # comment these
// q, _, err := emb.Voyage(vKey, query) # uncomment these
```

## Overview

This project implements a lightweight CLI that efficiently stores and searches file embeddings. The system:

1. **Extracts embeddings** from text content of files.
2. **Stores metadata and embeddings** in **DuckDB**.
3. **Performs nearest neighbor searches** using **hnsw** (a Go-native vector search engine).
4. **Minimizes redundant computations** by tracking file hashes to recompute embeddings only when files change.

## Tools & Libraries Considered

We explored various options for storing and searching embeddings, comparing their trade-offs:

### **1️⃣ Vector Search Libraries**voyageAPIKeyPath

| Library        | Go-native                     | Standalone       | Scalability                           | Ease of Use     | Notes                                                              |
| -------------- | ----------------------------- | ---------------- | ------------------------------------- | --------------- | ------------------------------------------------------------------ |
| **FAISS**      | ❌ (CGO)                      | ✅ (local index) | ✅ (handles millions of vectors)      | ⚠️ Requires CGO | Fast, but lacks Go support. No metadata storage.                   |
| **Vald**       | ✅ Yes                        | ✅ Yes           | ⚠️ Limited                            | ✅ Simple API   | Best choice for Go CLI. No external DB needed.                     |
| **coder/hnsw** | ✅ Yes                        | ✅ Yes           | ⚠️ Limited                            | ✅ Simple       | Best choice for Go CLI. No external DB needed.                     |
| **Milvus**     | ❌ No                         | ❌ Needs server  | ✅ Distributed                        | ❌ Heavy infra  | Python-first, better for large-scale ML applications.              |
| **Weaviate**   | ⚠️ Partial (GraphQL/REST API) | ❌ Needs server  | ✅ Scalable                           | ⚠️ Extra setup  | Better suited for structured metadata + vector search.             |
| **Annoy**      | ✅ Yes                        | ✅ Yes           | ❌ Static trees (not update-friendly) | ✅ Simple       | Good for read-heavy workloads, but not ideal for frequent updates. |

### **2️⃣ Storage & Metadata Databases**

| Database                  | Standalone | Lightweight | SQL Support | Notes                                                      |
| ------------------------- | ---------- | ----------- | ----------- | ---------------------------------------------------------- |
| **DuckDB**                | ✅ Yes     | ✅ Yes      | ✅ Yes      | Best choice for our metadata (file path, hash, embedding). |
| **PostgreSQL + pgvector** | ❌ No      | ❌ Heavy    | ✅ Yes      | Great for enterprise, but too heavyweight for our needs.   |
| **Featureform**           | ❌ No      | ❌ No       | ✅ Yes      | Built for ML pipelines, not CLI-based search.              |

---

## Final Design: DuckDB + hnsw

### **Workflow**

1. **Start the CLI**

   - Load **existing embeddings** from **DuckDB**.
   - Load **vector index** from **hnsw**.

2. **Read all files in the current directory**

   - Compute the **MD5/SHA256 hash** of each file.
   - Compare with stored hashes in **DuckDB**.
   - **Only compute embeddings for new/changed files**.

3. **Update DuckDB**

   - Insert new embeddings for changed files.
   - Delete outdated entries.

4. **Update hnsw**

   - Add/replace embeddings in the **hnsw index**.

5. **Perform nearest neighbor search**
   - Compute query embedding.
   - Search **hnsw** for nearest neighbors.
   - Retrieve file paths from **DuckDB**.

### **Architecture**

```plaintext
+------------------------+
|        CLI App        |
+------------------------+
            |
            v
+------------------------+
|   Read Files & Hashes  |
+------------------------+
            |
            v
+------------------------+    +-----------------------+
| Compute Embeddings     |<-->|     Ollama | VoyageAI |
+------------------------+    +-----------------------+
            |
            v
+------------------------+
|       DuckDB          |
|  (Metadata Storage)   |
+------------------------+
            |
            v
+------------------------+
| Nearest Neighbor hnsw  |
|        Search          |
+------------------------+
```

---

## Why This Design Works

✅ **Lightweight**: No need for a heavy DB like PostgreSQL.
✅ **Efficient Updates**: Embeddings are recomputed only when files change.
✅ **Fast Vector Search**: hnsw provides ANN without CGO.
✅ **Simple Setup**: Standalone, runs locally, no servers needed.

---

# What I learned / Next steps

Below is a structured approach to building a high-performance Retrieval-Augmented Generation (RAG) system specialized for code. We’ll focus on both _speed_ (throughput/latency) and _consistency_ (quality of results). The general pipeline has these stages:

1. **Ingestion & Preprocessing**
2. **Chunking & Embedding**
3. **Indexing**
4. **Query Processing & Retrieval**
5. **Augmentation & Generation**
6. **System Optimization & Maintenance**

Let’s walk through each step in detail.

---

## 1. Ingestion & Preprocessing

### 1.1 Gather and Organize the Repository

- **Scan your repo** to gather all code files (and optionally documentation, READMEs, etc.).
- **Filter out** files that are not useful for retrieval (e.g., build artifacts, large autogenerated files, or libraries you do not need to answer questions about).

### 1.2 Source Code Parsing

- **Language-specific parsing**: For structured chunking and better embeddings, you may want to parse the code with a language parser (like [tree-sitter](https://tree-sitter.github.io/tree-sitter/) or any other AST-based tool). This allows you to break the code into logical units (functions, classes, methods) rather than arbitrary text splits.

### 1.3 Cleaning & Normalizing

- **Remove excessive comments** that are not relevant or out of scope (for example, large license headers).
- **Convert tab indentation to spaces** consistently so that the embeddings don’t capture irrelevant formatting differences.
- **Retain docstrings** in a structured way if they add value for code queries.

---

## 2. Chunking & Embedding

### 2.1 Chunking Strategy

Chunking is critical for returning high-quality results. If your code chunks are too large, the embeddings become “blurry,” encompassing many functions/classes at once. If they are too small, you may lose context.

- **Logical chunking**: Whenever possible, chunk around function or class boundaries. For example:
  - **Single function** (plus docstrings and relevant comments).
  - **Single class** (if not too large).
  - **File-level** chunk for short files (e.g., small utility scripts).
- **Sliding Window** or **Hierarchical** approach:
  - For very large functions/classes, consider a sliding window approach or hierarchical splitting (function-level with further sub-chunking if a single function is very large).

### 2.2 Selecting an Embedding Model

- **LLM-based text embeddings**: If you’re using OpenAI, `text-embedding-ada-002` is commonly used.
- **Code-specific embeddings**: For more nuanced code understanding, consider specialized models like [CodeBERT](https://github.com/microsoft/CodeBERT) or [UniXcoder](https://arxiv.org/abs/2109.12486). OpenAI’s `text-embedding-ada-002` has also been found to work well on code, especially for multi-lingual code repos.
- **Trade-off**: Larger models might capture deeper semantics but are more expensive and sometimes slower. Weigh your speed/quality requirements.

### 2.3 Generating Embeddings

- For each chunk, generate an **embedding vector**.
- **Metadata attachment**: Tag each embedding with relevant metadata:
  - **File name** or path.
  - **Language** (e.g., Python, JavaScript).
  - **Function/Class name**.
  - **Line numbers** to facilitate reference or code snippet reconstruction.

---

## 3. Indexing

### 3.1 Choosing a Vector Database

High-performance vector databases include:

- **FAISS** (Facebook AI Similarity Search) for local on-prem or smaller scale needs.
- **Pinecone**, **Weaviate**, or **Milvus** for managed or distributed solutions.

### 3.2 Index Configuration

- **Index type**: Typically an **IVF** (Inverted File) or **HNSW** (Hierarchical Navigable Small World) index is common for sub-millisecond retrieval on large vector sets.
- **Dimension**: Must match your embedding’s dimension (e.g., 1536 for `text-embedding-ada-002`).
- **Metadata filtering**: Ensure the vector database supports filtering by metadata so you can restrict results by language, directory, or other tags if needed.

### 3.3 Data Insertion

- **Batch insertion**: Insert embeddings in large batches to minimize overhead.
- **Index building**: Some databases require an indexing step post-insertion. Optimize by segmenting indexes if your dataset is massive (sharding).

---

## 4. Query Processing & Retrieval

### 4.1 Query Understanding

When the user asks a question, you want to produce a query embedding that captures the semantic meaning. Steps:

1. **Parse user query**: Possibly run a quick classification to detect language references or function references.
2. **Rewrite or refine**: Use an LLM to clarify or refine the query if needed (especially if the question is vague).

### 4.2 Searching the Vector Index

- Compute the **query embedding** using the same model you used for code.
- **K-Nearest Neighbors (kNN)** retrieval: By default, you might fetch top-5 or top-10 code chunks.
- **Metadata-based filtering**: If the user specifically asks for “Python code to do X,” filter out non-Python chunks.

### 4.3 Reranking

- Optional step: **Rerank** the top candidates using a secondary scoring approach (e.g., cross-encoder, reranker LLM) for better precision. This can improve result quality but adds overhead.

---

## 5. Augmentation & Generation

### 5.1 Retrieval-Augmented Generation Flow

Once you have the top relevant code chunks, you can feed them into a “context window” of your LLM along with the query. A typical approach:

1. **Prompt**: Provide user query + relevant code chunks.
2. **Generate**: The LLM uses these retrieved snippets as context to generate an answer.

### 5.2 Chunk Combination

- If multiple chunks from different files are relevant, you can combine them in the prompt.
- Consider summarizing or lightly compressing long chunks before feeding them to the LLM to stay within context limits.

### 5.3 Output Formatting

- Encourage the LLM to produce code in a well-formatted way—sometimes with instructions such as: _“Return a Python code snippet with necessary imports.”_
- You can also request a concise or step-by-step explanation.

---

## 6. System Optimization & Maintenance

### 6.1 Embedding & Index Updates

- **Continuous integration**: If your codebase changes frequently, automate re-embedding. A “diff-based” approach can embed only changed/added code.
- **Index refresh**: After new embeddings are generated, update your vector store to keep it consistent.

### 6.2 Scaling & Caching

- **Vector caching**: Cache frequent queries and their top results in an in-memory store (like Redis).
- **Prompt caching**: If you’re using an LLM, caching final answers for repeated queries can greatly reduce cost and latency.
- **Load balancing**: If you scale out to multiple worker nodes, ensure your vector store can handle concurrency (or replicate the index as needed).

### 6.3 Latency Minimization

- **Hardware acceleration**: Use GPUs for similarity search in solutions that support GPU-based FAISS or HNSW.
- **Distillation/Quantization**: If the embedding model is too large, consider quantization or smaller dimension embeddings to speed up both embedding generation and vector search.
- **Short-circuiting**: If you detect queries that do not require code context, route them to a simpler text-based model or FAQ system.

### 6.4 Ensuring Consistent, High-Quality Results

- **Quality monitoring**: Periodically evaluate retrieval results on a set of known queries (search for known functions, tasks, or bug references) to see if your system retrieves the correct chunks.
- **Feedback loop**: If users can upvote/downvote or provide corrections, feed that back into your indexing or reranking logic.

---

# Putting It All Together

1. **Preprocess & Parse** your repo: Identify logical code units.
2. **Chunk & Embed**: Use an LLM-based or code-specific model, store embeddings with metadata.
3. **Index** in a high-performance vector DB (FAISS, Pinecone, Weaviate, etc.).
4. **Query**: Convert user query to an embedding, retrieve top k nearest chunks, optionally rerank them.
5. **Generate**: Feed the top matches to your LLM to produce an answer or snippet.
6. **Optimize** for speed: caching, efficient indexing, hardware acceleration, robust chunking.
7. **Maintain**: Continuously update the index and monitor retrieval quality.

This approach will allow you to:

- Quickly retrieve relevant code snippets and documentation for large repos.
- Provide context-enriched, accurate code completions or explanations.
- Scale and adapt as your codebase and user queries grow.

By following the best practices around chunking, specialized embeddings for code, and well-structured RAG pipelines, you’ll get consistent, high-quality results _and_ speed.
