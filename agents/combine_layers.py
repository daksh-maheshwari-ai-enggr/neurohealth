from pydantic import BaseModel,Field
from typing import List,Dict,Optional
from langgraph.graph import StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from chunking import load_n_chunk_data
from hybrid import HybridRetriever,HybridRetrieverWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
import sys
import os

sys.path.append(os.path.abspath(".."))


load_dotenv()

llm=ChatGroq(model="llama-3.3-70b-versatile")

DATA_PATH = r"C:\orchestration\data"
INDEX_PATH = r"C:\orchestration\rag\rag\vector_store\faiss_index"

embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)


vectordb = FAISS.load_local(
    INDEX_PATH,
    embedding,
    allow_dangerous_deserialization=True
)

dense_retriever = vectordb.as_retriever()

docs = load_n_chunk_data(DATA_PATH)
sparse_retriever = BM25Retriever.from_documents(docs)
sparse_retriever.k = 3


hybrid = HybridRetriever(
    dense=dense_retriever,
    sparse=sparse_retriever,
    weights=(0.7, 0.3)
)

hybrid_wrapper = HybridRetrieverWrapper(hybrid=hybrid)

class Signals(BaseModel):
    cardiac: bool = False
    respiratory: bool = False
    neurological: bool = False
    metabolic: bool = False
    injury: bool = False
    mental_health: bool = False
    general: bool = False

class HealthState(BaseModel):
    user_input: str


    risk_level: Optional[str] = None
    risk_reason: Optional[str] = None


    signals: Signals = Field(default_factory=Signals)
    symptoms: List[str] = Field(default_factory=list)
    duration: Optional[str] = None
    severity: Optional[str] = None
    behavior_change: List[str] = Field(default_factory=list)
    risk_markers: List[str] = Field(default_factory=list)
    additional_context: Dict = Field(default_factory=dict)    
    retrieved_docs: List[str] = Field(default_factory=list)
    plan: Optional[str] = None

class PlannerOutput(BaseModel):
    clinical_summary: str
    risk_interpretation: str
    probable_mechanism: str
    immediate_actions: List[str]
    monitoring_steps: List[str]
    escalation_triggers: List[str]
    needs_urgent_care: bool
    follow_up_questions: List[str]




prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a medical risk classifier."),
    ("human", """
Classify the health risk level of the following message.

Categories:
- EMERGENCY (life-threatening symptoms)
- HIGH_RISK (serious medical condition)
- MODERATE
- LOW

Message: {user_input}

Return only JSON like:
{{
    "risk_level": "",
    "reason": ""
}}
""")
])

def risk_classifier(state: HealthState) -> HealthState:

    chain = prompt | llm
    response = chain.invoke({"user_input": state.user_input})

    new_state = state.model_copy()

    # Direct string store karo
    new_state.risk_level = response.content.strip()
    new_state.risk_reason = None

    return new_state



extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a medical signal extraction engine.

STRICT RULES:
- Use only provided JSON structure.
- Do NOT invent new top-level keys.
- If extra info is necessary, place it in "additional_context".
- Do NOT hallucinate.
"""),

    ("human", """
Extract structured health information.

Return JSON:

{{
  "signals": {{
    "cardiac": false,
    "respiratory": false,
    "neurological": false,
    "metabolic": false,
    "injury": false,
    "mental_health": false,
    "general": false
  }},
  "symptoms": [],
  "duration": null,
  "severity": null,
  "behavior_change": [],
  "risk_markers": [],
  "additional_context": {{}}
}}

Message: {user_input}
""")
])


def user_model_node(state: HealthState) -> HealthState:

    chain = extraction_prompt | llm
    response = chain.invoke({"user_input": state.user_input})

    new_state = state.model_copy()

    # Testing mode: raw output store karo
    new_state.additional_context["raw_extraction"] = response.content.strip()

    return new_state






def retrieval_node(state: HealthState) -> HealthState:
    user_input = state.user_input
    query = f"{user_input}"

    results = hybrid_wrapper._get_relevant_documents(query)

    new_state = state.model_copy()
    new_state.retrieved_docs = [
        d.page_content for d in results
    ]

    return new_state

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a clinical reasoning engine.

STRICT RULES:
- Do NOT diagnose.
- Do NOT prescribe medication dosage.
- Do NOT write essays.
- Avoid repetition.
- Use high clinical information density.
- Base reasoning ONLY on provided evidence.
- Use structured concise logic.
"""),

    ("human", """
Risk Level: {risk_level}

Structured Signals:
{signals}

Symptoms:
{symptoms}

Risk Markers:
{risk_markers}

Retrieved Evidence:
{retrieved_docs}

TASKS:
1. Provide concise clinical summary.
2. Interpret risk severity logically.
3. Explain probable physiological mechanism.
4. List immediate self-care actions (if safe).
5. List monitoring steps.
6. List clear escalation triggers.
7. Indicate if urgent care required.
8. Provide focused follow-up questions.

Output must be structured.
No markdown.
No headings.
No narrative filler.
""")
])



def planner_node(state: HealthState) -> HealthState:

    chain = planner_prompt | llm

    result = chain.invoke({
        "risk_level": state.risk_level,
        "signals": state.signals.model_dump(),
        "symptoms": state.symptoms,
        "risk_markers": state.risk_markers,
        "retrieved_docs": state.retrieved_docs[:3]
    })

    new_state = state.model_copy()

    new_state.plan = result.model_dump()

    return new_state








builder = StateGraph(HealthState)

# Add nodes
builder.add_node("risk_classifier", risk_classifier)
builder.add_node("user_model", user_model_node)
builder.add_node("retrieval",retrieval_node)
builder.add_node("planner", planner_node)

# Entry
builder.set_entry_point("risk_classifier")

# Edges
builder.add_edge("risk_classifier", "user_model")
builder.add_edge("user_model","retrieval")
builder.add_edge("retrieval","planner")

# Compile
app = builder.compile()


initial_state = HealthState(
    user_input="Blood sugar level is low and feeling headache,dizziness and feels no hunger."
)

result = app.invoke(initial_state)

print(result)


