<p align="center">
  <h1 align="center">✈️ Multimodal RAG System for Airline Operations Intelligence</h1>
  <p align="center">
    <strong>Enterprise Edition v2.0</strong>
    <br />
    A production-grade Retrieval-Augmented Generation system for safety-critical airline operations
    <br />
    <br />
    <a href="#-quick-start">Quick Start</a>
    ·
    <a href="#-api-documentation">API Docs</a>
    ·
    <a href="#-architecture">Architecture</a>
    ·
    <a href="#-usage-guide">Usage Guide</a>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.109+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/OpenAI-GPT--4-orange.svg" alt="OpenAI">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  <img src="https://img.shields.io/badge/Docker-Ready-blue.svg" alt="Docker">
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Technical Deep Dive](#-technical-deep-dive)
- [Deployment](#-deployment)
- [License](#-license)

---

## 🎯 Overview

This project implements a **production-grade, multimodal Retrieval-Augmented Generation (RAG) system** specifically designed for airline operations intelligence. The system addresses the critical challenge of **reducing LLM hallucinations** in safety-critical operational workflows.

### Problem Statement

In airline operations, incorrect information can have serious safety implications. Standard LLMs:
- ❌ Hallucinate procedures and specifications
- ❌ Lack access to proprietary SOPs and maintenance manuals
- ❌ Cannot cite sources for verification
- ❌ Provide inconsistent answers

### Solution

This RAG system provides:
- ✅ **Grounded responses** backed by actual documents
- ✅ **Citation enforcement** for every claim
- ✅ **Confidence scoring** with abstention for uncertain answers
- ✅ **Domain-specific retrieval** optimized for aviation terminology

### Target Use Cases

| Use Case | Example Query |
|----------|---------------|
| **SOP Lookup** | "What is the pre-flight inspection procedure for B737?" |
| **Maintenance Reasoning** | "What are the MEL requirements for APU failure on A320?" |
| **Delay Root Cause** | "Why was flight UA234 delayed due to maintenance?" |
| **Crew Operations** | "What are the rest requirements for international flights?" |

---

## 🔥 Key Features

### Core Capabilities

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Hybrid Retrieval** | Dense (FAISS) + Sparse (BM25) with RRF | Best of semantic + keyword search |
| **Agentic RAG** | Query classification + multi-step decomposition | Handles complex questions |
| **Citation Enforcement** | Claim extraction + source attribution | Full traceability |
| **Confidence Scoring** | Abstention logic for low confidence | Prevents hallucinations |
| **Streaming Responses** | Real-time SSE streaming | Better UX |
| **Multimodal Processing** | PDF parsing with tables/figures | Rich document support |

### Production Features

| Feature | Implementation |
|---------|----------------|
| **Latency Budgets** | Retrieval < 200ms, Total < 5s |
| **SLA Monitoring** | 99% compliance target |
| **Structured Logging** | JSON logs with structlog |
| **Health Checks** | Kubernetes-ready endpoints |
| **Docker Support** | Multi-container deployment |

---

## 🏗 Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER QUERY                                         │
│                    "What is the B737 pre-flight checklist?"                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         QUERY CLASSIFIER                                     │
│                    Classifies: SOP_LOOKUP (confidence: 0.95)                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        QUERY DECOMPOSER                                      │
│              Breaks complex queries into sub-questions                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      HYBRID RETRIEVAL ENGINE                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  Dense Search   │  │  Sparse Search  │  │    Metadata Filtering       │  │
│  │  (FAISS + OAI)  │  │    (BM25)       │  │  (fleet, airport, phase)    │  │
│  └────────┬────────┘  └────────┬────────┘  └──────────────┬──────────────┘  │
│           └────────────────────┼──────────────────────────┘                  │
│                                ▼                                             │
│                    Reciprocal Rank Fusion (RRF)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      GROUNDING PIPELINE                                      │
│  ┌─────────────────────────┐    ┌─────────────────────────────────────────┐ │
│  │   Citation Enforcer     │    │        Confidence Scorer                │ │
│  │  - Claim extraction     │    │  - Retrieval confidence                 │ │
│  │  - Source matching      │    │  - Citation coverage                    │ │
│  │  - Citation generation  │    │  - Abstention logic                     │ │
│  └─────────────────────────┘    └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GROUNDED RESPONSE                                     │
│  Answer: "The B737 pre-flight checklist includes: 1. Check fuel levels..."  │
│  Citations: [SOP-B737-001.pdf, Page 2, Section 1.1]                         │
│  Confidence: 0.92                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```


### Component Details

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Query Classifier** | GPT-4 + Few-shot | Categorizes queries into SOP, Maintenance, Delay, Crew |
| **Query Decomposer** | GPT-4 + Chain-of-Thought | Breaks complex queries into sub-questions |
| **Dense Retrieval** | OpenAI Embeddings + FAISS | Semantic similarity search |
| **Sparse Retrieval** | BM25 (rank_bm25) | Keyword-based search with aviation terminology |
| **Hybrid Fusion** | Reciprocal Rank Fusion | Combines dense + sparse results |
| **Citation Enforcer** | GPT-4 + Claim Extraction | Ensures every claim has a source |
| **Confidence Scorer** | Multi-signal aggregation | Determines when to abstain |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **OpenAI API Key** (GPT-4 access recommended)
- **8GB+ RAM** (for FAISS index)

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-org/multimodal-rag-airline-ops.git
cd multimodal-rag-airline-ops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
# Required: OPENAI_API_KEY=sk-...
```

### 3. Seed Sample Data

```bash
# Load sample airline documents
python scripts/seed_data.py
```

### 4. Start the API

```bash
# Development mode with hot reload
uvicorn backend.main:app --reload --port 8000

# Production mode
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 5. Start the Dashboard

```bash
streamlit run frontend/ops_dashboard.py
```

### 6. Test the System

```bash
# Health check
curl http://localhost:8000/health

# Sample query
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the pre-flight checklist for B737?"}'
```

---

## 📖 Usage Guide

### Basic Query

```python
import requests

response = requests.post(
    "http://localhost:8000/query/",
    json={
        "query": "What are the MEL requirements for APU failure?",
        "session_id": "user-123"
    }
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
print(f"Citations: {result['citations']}")
```

### Streaming Query

```python
import requests

with requests.post(
    "http://localhost:8000/stream/query",
    json={"query": "Explain the B787 fuel system"},
    stream=True
) as response:
    for line in response.iter_lines():
        if line:
            print(line.decode('utf-8'))
```

### Document Ingestion

```python
import requests

with open("sop_manual.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/ingest/document",
        files={"file": f},
        data={
            "doc_type": "SOP",
            "fleet_type": "B737",
            "effective_date": "2024-01-01"
        }
    )
```

### Query with Metadata Filters

```python
response = requests.post(
    "http://localhost:8000/query/",
    json={
        "query": "What is the engine start procedure?",
        "filters": {
            "fleet_type": "A320",
            "doc_type": "SOP",
            "phase": "pre-flight"
        }
    }
)
```

---

## 📚 API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query/` | POST | Process query with full RAG pipeline |
| `/stream/query` | POST | Stream response with SSE |
| `/ingest/document` | POST | Ingest PDF document |
| `/health` | GET | Health check |
| `/metrics` | GET | Latency and SLA metrics |

### Query Request Schema

```json
{
  "query": "string (required)",
  "session_id": "string (optional)",
  "filters": {
    "fleet_type": "string (optional)",
    "doc_type": "string (optional)",
    "airport": "string (optional)",
    "phase": "string (optional)"
  },
  "top_k": "integer (default: 5)",
  "include_sources": "boolean (default: true)"
}
```

### Query Response Schema

```json
{
  "answer": "string",
  "confidence": "float (0.0-1.0)",
  "citations": [
    {
      "source": "string",
      "page": "integer",
      "section": "string",
      "quote": "string"
    }
  ],
  "query_type": "string (SOP_LOOKUP|MAINTENANCE|DELAY|CREW)",
  "abstained": "boolean",
  "latency_ms": "integer"
}
```


---

## 📁 Project Structure

```
multimodal-rag-airline-ops/
├── backend/
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Enterprise configuration (Pydantic Settings)
│   ├── schemas.py              # Request/Response Pydantic models
│   ├── __init__.py
│   ├── api/
│   │   ├── ingest.py           # Document ingestion endpoints
│   │   ├── query.py            # Query processing endpoints
│   │   ├── stream.py           # SSE streaming endpoints
│   │   └── __init__.py
│   ├── agents/
│   │   ├── query_classifier.py # Query type classification (SOP/Maintenance/Delay/Crew)
│   │   ├── decomposer.py       # Multi-step query decomposition
│   │   └── __init__.py
│   ├── retrieval/
│   │   ├── dense.py            # FAISS + OpenAI embeddings
│   │   ├── sparse.py           # BM25 keyword search
│   │   ├── hybrid.py           # Reciprocal Rank Fusion
│   │   └── __init__.py
│   ├── grounding/
│   │   ├── citation_enforcer.py # Claim extraction + source attribution
│   │   ├── confidence_scorer.py # Multi-signal confidence scoring
│   │   └── __init__.py
│   ├── vision/
│   │   ├── layout_parser.py    # PDF structure extraction
│   │   ├── table_extractor.py  # Table detection and parsing
│   │   └── __init__.py
│   ├── evaluation/
│   │   ├── factuality.py       # LLM-as-a-Judge hallucination detection
│   │   ├── coverage.py         # Response completeness metrics
│   │   └── __init__.py
│   └── memory/
│       ├── conversation_store.py # Session-based conversation history
│       └── __init__.py
├── vectorstore/
│   ├── faiss_index.py          # FAISS index management
│   ├── bm25_index.py           # BM25 index management
│   └── __init__.py
├── monitoring/
│   ├── latency.py              # Latency budget tracking
│   ├── logging.py              # Structured JSON logging
│   └── __init__.py
├── frontend/
│   └── ops_dashboard.py        # Streamlit operations dashboard
├── scripts/
│   └── seed_data.py            # Sample data seeding script
├── assets/
│   └── sample_data.py          # Sample airline documents
├── docker-compose.yml          # Multi-container deployment
├── Dockerfile                  # Backend container
├── Dockerfile.frontend         # Frontend container
├── requirements.txt            # Python dependencies
├── .env.example                # Environment template
└── .gitignore
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ | - | OpenAI API key |
| `OPENAI_MODEL` | ❌ | `gpt-4` | Model for generation |
| `EMBEDDING_MODEL` | ❌ | `text-embedding-3-small` | Model for embeddings |
| `LOG_LEVEL` | ❌ | `INFO` | Logging level |
| `ENVIRONMENT` | ❌ | `development` | Environment name |

### Configuration File (`backend/config.py`)

```python
class Settings(BaseSettings):
    # API
    openai_api_key: str
    openai_model: str = "gpt-4"
    embedding_model: str = "text-embedding-3-small"

    # Retrieval
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    similarity_threshold: float = 0.75
    top_k: int = 5

    # Grounding
    confidence_threshold: float = 0.78
    min_citations_required: int = 1
    abstention_message: str = "I don't have enough information..."

    # Latency Budgets (milliseconds)
    retrieval_budget_ms: int = 200
    generation_budget_ms: int = 3000
    total_budget_ms: int = 5000

    # SLA
    sla_target: float = 0.99
```

---

## 🔬 Technical Deep Dive

### Hybrid Retrieval

The system combines dense and sparse retrieval using **Reciprocal Rank Fusion (RRF)**:

```python
def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    """
    Combine rankings using RRF formula:
    score(d) = Σ 1 / (k + rank(d))
    """
    scores = defaultdict(float)

    for rank, doc in enumerate(dense_results):
        scores[doc.id] += 1 / (k + rank + 1)

    for rank, doc in enumerate(sparse_results):
        scores[doc.id] += 1 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### Citation Enforcement

Every claim in the response is verified against source documents:

1. **Claim Extraction**: GPT-4 extracts factual claims from the response
2. **Source Matching**: Each claim is matched to retrieved documents
3. **Citation Generation**: Citations are formatted with source, page, section

### Confidence Scoring

Multi-signal confidence aggregation:

```python
confidence = (
    0.4 * retrieval_confidence +  # Top-k similarity scores
    0.3 * citation_coverage +      # % of claims with citations
    0.2 * query_match_score +      # Query-answer relevance
    0.1 * source_authority         # Document authority score
)
```

### Abstention Logic

The system abstains when:
- Confidence < threshold (default: 0.78)
- No citations found for claims
- Retrieved documents below similarity threshold
- Query classified as out-of-scope

---

## 🐳 Deployment

### Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Docker Compose Configuration

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "8501:8501"
    depends_on:
      - api
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    spec:
      containers:
      - name: api
        image: rag-airline-ops:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "1"
          limits:
            memory: "8Gi"
            cpu: "2"
```


---

## 📊 Monitoring & Observability

### Latency Tracking

The system tracks latency at each stage:

```python
# Latency budgets
RETRIEVAL_BUDGET_MS = 200    # Dense + Sparse + Fusion
GENERATION_BUDGET_MS = 3000  # LLM generation
TOTAL_BUDGET_MS = 5000       # End-to-end

# Metrics endpoint
GET /metrics
{
  "retrieval_p50_ms": 85,
  "retrieval_p99_ms": 180,
  "generation_p50_ms": 1200,
  "generation_p99_ms": 2800,
  "total_p50_ms": 1500,
  "total_p99_ms": 4200,
  "sla_compliance": 0.994
}
```

### Structured Logging

All logs are JSON-formatted for easy parsing:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "event": "query_processed",
  "query_id": "abc123",
  "query_type": "SOP_LOOKUP",
  "confidence": 0.92,
  "latency_ms": 1450,
  "citations_count": 3,
  "abstained": false
}
```

### Health Checks

```bash
# Liveness probe
GET /health
{"status": "healthy", "version": "2.0.0"}

# Readiness probe (checks dependencies)
GET /health/ready
{"status": "ready", "faiss_index": "loaded", "bm25_index": "loaded"}
```

---

## 🧪 Testing

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=backend --cov-report=html

# Specific module
pytest tests/test_retrieval.py -v
```

### Test Categories

| Category | Description |
|----------|-------------|
| `tests/unit/` | Unit tests for individual components |
| `tests/integration/` | Integration tests for API endpoints |
| `tests/e2e/` | End-to-end tests with real LLM calls |
| `tests/evaluation/` | Factuality and coverage evaluation |

---

## 🔒 Security Considerations

- **API Key Management**: Use environment variables, never commit keys
- **Input Validation**: All inputs validated with Pydantic
- **Rate Limiting**: Implement rate limiting for production
- **Document Access**: Implement document-level access control
- **Audit Logging**: Log all queries for compliance

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [OpenAI](https://openai.com) for GPT-4 and embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [FastAPI](https://fastapi.tiangolo.com) for the API framework
- [Streamlit](https://streamlit.io) for the dashboard

---

<p align="center">
  <strong>Built for safety-critical airline operations</strong>
  <br />
  <sub>Reducing hallucinations, one query at a time ✈️</sub>
</p>