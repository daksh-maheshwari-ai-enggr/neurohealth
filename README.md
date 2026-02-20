# NeuroHealth – AI-Powered Health Assistant

NeuroHealth is a health assistance system built around a structured orchestration pipeline rather than a single LLM prompt.

The core idea is simple:

Health queries are high-risk.  
They should not be handled by one uncontrolled model call.

Instead, NeuroHealth separates risk detection, structured extraction, evidence retrieval, reasoning, and response generation into distinct layers. This makes the system easier to control, audit, extend, and evaluate.

---

# 🎯 Project Direction

Most online symptom checkers:
- Use rigid rule trees  
- Ignore conversational nuance  
- Cannot adapt to ambiguous symptom descriptions  
- Struggle with contextual reasoning  

NeuroHealth is being designed as:

- A reasoning-first system  
- Retrieval-grounded  
- Urgency-aware  
- Structurally constrained  
- Extendable with tools  

The objective is not to “replace doctors” —  
but to intelligently assist users in understanding next steps.

---

# 🏗 System Architecture (10-Layer Design)

Instead of:
User → LLM → Response

NeuroHealth follows:

User → Risk Gate → Structured Model → Retrieval → Planner → Controlled Generation → Safety Validation → Explainability → Memory

---

## Layer 1 – Risk Classification

Initial safety gate.

- Categorizes input into EMERGENCY / HIGH_RISK / MODERATE / LOW  
- Prevents downstream reasoning on life-threatening patterns  
- Designed to fail-safe toward escalation  

---

## Layer 2 – Structured User Modeling

Transforms free-text health queries into structured state.

Extracts:
- Symptom list
- Duration
- Severity
- Behavioral changes
- Risk markers
- Physiological signal categories (cardiac, respiratory, metabolic, etc.)

Strict JSON schema enforcement reduces drift and hallucination.

---

## Layer 3 – Hybrid Retrieval (Evidence Grounding)

Uses weighted hybrid search:

- Dense retrieval → FAISS (MiniLM embeddings)
- Sparse retrieval → BM25
- Weighted merge for semantic + lexical balance

Current knowledge ingestion includes:
- MedlinePlus scraping (BeautifulSoup)
- Mayo Clinic extraction (Trafilatura)
- Structured DailyMed API access (drug label parsing)

The knowledge base is expanding and continuously being refined.

---

## Layer 4 – Clinical Planning Engine

Structured reasoning module.

Key constraints:
- Does NOT diagnose
- Does NOT prescribe medication dosage
- Avoids long narrative text
- Uses high-density structured reasoning

Outputs:
- Clinical summary
- Risk interpretation
- Possible physiological explanation
- Immediate safe actions
- Monitoring steps
- Clear escalation triggers
- Urgent care flag
- Follow-up questions

This layer separates reasoning from response wording.

---

## Layer 5 – Tool Integration (Under Development)

Planned capabilities:
- Structured DailyMed ingestion (SPL XML parsing)
- Drug–symptom lookup
- Dosage section extraction
- Future appointment routing logic

Tools will be invoked only when necessary — not blindly by the LLM.

---

## Layer 6 – Controlled Response Generator

Converts structured planner output into:

- User-friendly guidance
- Health-literacy-aware explanation
- Clear and non-alarming tone
- Safety-aligned messaging

Separates reasoning logic from surface language.

---

## Layer 7 – Self-Reflection Module

Internal verification layer.

- Checks internal consistency
- Detects contradictions
- Flags unsupported claims

Designed to reduce silent reasoning errors.

---

## Layer 8 – Hallucination & Safety Filter

- Ensures claims are grounded in retrieved evidence
- Blocks unsafe or speculative medical statements
- Detects overconfident phrasing

Acts as a second safety gate before final output.

---

## Layer 9 – Explainability Layer

Provides:
- Evidence snippets used
- Structured reasoning trace
- Transparency for debugging and evaluation

Critical for clinical trust and research validation.

---

## Layer 10 – Context & Feedback Memory

Planned support for:
- Multi-turn dialogue continuity
- Chronic condition tracking
- Structured user state persistence

---

# 📚 Knowledge Base Construction

Current ingestion experiments:

### MedlinePlus
- Condition and test data scraped using BeautifulSoup
- Cleaned and chunked for embedding

### Mayo Clinic
- Long-form condition articles extracted via Trafilatura
- Structured into retrieval-friendly segments

### DailyMed API
- SPL XML parsing
- Drug label and dosage section extraction
- Structured metadata storage

The knowledge base is currently partial and being expanded.

---

# 🧠 Engineering Approach

Key design principles:

- Separation of reasoning and generation
- Retrieval before response
- Risk-first architecture
- Schema-restricted outputs
- Modular extensibility
- Safety over verbosity

The system is built using LangGraph to maintain explicit state transitions between layers.

---

# 📊 Planned Evaluation Strategy

Future evaluation will include:

- Urgency classification precision
- Retrieval grounding accuracy
- Safety violation rate
- Human review for clinical appropriateness
- Stress testing with adversarial inputs
- Latency and inference efficiency analysis

---

# 🛠 Tech Stack

- Python
- Regex
- Pydantic
- LangGraph
- LangChain
- Langextract
- FAISS
- HuggingFace Embeddings (MiniLM)
- BM25 Retriever
- Groq / Open-AI LLM's
- BeautifulSoup
- Trafilatura

---

# 📦 Deliverables

- Open-source repository
- Modular architecture
- Structured ingestion pipeline
- Reproducible setup instructions
- Interactive demo (planned)
- Evaluation documentation (planned)

---
