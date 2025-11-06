import os
import streamlit as st
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# --- Cached initializers so Streamlit doesn't rebuild on every rerun ---
@st.cache_resource
def get_vectorstore():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME", "my-rag-index")

    # Ensure index exists (no-op if already exists)
    if index_name not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=768,  # Gemini embeddings dim
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    emb = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    return PineconeVectorStore(index_name=index_name, embedding=emb)

@st.cache_resource
def get_llm():
    return GoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=os.getenv("GEMINI_API_KEY"))

# Build a graph (same 4 nodes) but also accumulate logs in state["logs"]
def build_agent():
    vectorstore = get_vectorstore()
    llm = get_llm()

    def add_log(state, msg):
        logs = state.get("logs", [])
        logs.append(msg)
        return logs

    def plan_node(state):
        q = state["question"].lower()
        logs = add_log(state, "[PLAN] deciding...")
        if any(tok in q for tok in ["what", "explain", "why", "benefit", "advantages"]):
            return {"action": "retrieve", "question": state["question"], "logs": logs}
        else:
            return {"action": "answer", "question": state["question"], "logs": logs}

    def retrieve_node(state):
        logs = add_log(state, "[RETRIEVE] searching vector DB...")
        docs = vectorstore.similarity_search(state["question"], k=1)
        context = docs[0].page_content if docs else ""
        return {"context": context, "question": state["question"], "logs": logs, "retrieved": context}

    def answer_node(state):
        logs = add_log(state, "[ANSWER] generating final output...")
        context = state.get("context", "")
        prompt = f"Context: {context}\n\nQuestion: {state['question']}\nAnswer:"
        ans = llm.invoke(prompt)
        return {"answer": ans, "question": state["question"], "logs": logs, "context": context}

    def reflect_node(state):
        logs = add_log(state, "[REFLECT] evaluating answer relevance...")
        q = state["question"]
        ans = state["answer"]
        prompt = f"Question: {q}\nAnswer: {ans}\n\nIs this answer relevant to the question? Reply YES or NO:"
        score = llm.invoke(prompt)
        return {"answer": ans, "reflection": score, "logs": logs, "context": state.get("context", "")}

    graph = StateGraph(dict)
    graph.add_node("PLAN", plan_node)
    graph.add_node("RETRIEVE", retrieve_node)
    graph.add_node("ANSWER", answer_node)
    graph.add_node("REFLECT", reflect_node)

    graph.set_entry_point("PLAN")
    graph.add_conditional_edges(
        "PLAN",
        lambda out: out["action"],
        {"retrieve": "RETRIEVE", "answer": "ANSWER"}
    )
    graph.add_edge("RETRIEVE", "ANSWER")
    graph.add_edge("ANSWER", "REFLECT")
    graph.add_edge("REFLECT", END)

    return graph.compile()

agent = build_agent()

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="RAG Agent (LangGraph + Gemini + Pinecone)", page_icon="🤖")
st.title("🤖 RAG Agent (LangGraph + Gemini + Pinecone)")
st.caption("Nodes: plan → retrieve → answer → reflect")

if "history" not in st.session_state:
    st.session_state.history = []

with st.form("qa_form"):
    question = st.text_input("Ask a question", placeholder="e.g., What are the benefits of renewable energy?")
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    with st.spinner("Running agent..."):
        result = agent.invoke({"question": question})
    st.session_state.history.append(
        {
            "question": question,
            "answer": result.get("answer", ""),
            "reflection": result.get("reflection", ""),
            "context": result.get("context", ""),
            "logs": result.get("logs", []),
        }
    )

st.subheader("Chat")
for turn in reversed(st.session_state.history):
    with st.expander(f"Q: {turn['question']}"):
        st.markdown(f"**Answer:**\n\n{turn['answer']}")
        st.markdown(f"**Reflection:** {turn['reflection']}")
        if turn.get("context"):
            st.markdown("**Retrieved Context:**")
            st.code(turn["context"])
        if turn.get("logs"):
            st.markdown("**Logs:**")
            for line in turn["logs"]:
                st.write(line)
