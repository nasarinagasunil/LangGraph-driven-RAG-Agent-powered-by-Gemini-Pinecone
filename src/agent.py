import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

# embeddings + pinecone
emb = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GEMINI_API_KEY")
)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
vectorstore = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX_NAME"), embedding=emb)

# LLM model
llm = GoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=os.getenv("GEMINI_API_KEY"))

####  NODE-1: PLAN  ####
def plan_node(state):
    q = state["question"].lower()
    print("[PLAN] deciding...")

    # simple logic
    if "what" in q or "explain" in q or "why" in q:
        return {"action": "retrieve", "question": state["question"]}
    else:
        return {"action": "answer", "question": state["question"]}

####  NODE-2: RETRIEVE  ####
def retrieve_node(state):
    print("[RETRIEVE] searching vector DB...")
    docs = vectorstore.similarity_search(state["question"], k=1)
    return {
        "context": docs[0].page_content if docs else "",
        "question": state["question"]
    }


####  NODE-3: ANSWER  ####
def answer_node(state):
    print("[ANSWER] generating final output...")
    context = state.get("context", "")
    prompt = f"Context: {context}\n\nQuestion: {state['question']}\nAnswer:"
    ans = llm.invoke(prompt)
    return {"answer": ans, "question": state["question"]}

####  NODE-4: REFLECT  ####
def reflect_node(state):
    print("[REFLECT] evaluating answer relevance...")
    q = state["question"]
    ans = state["answer"]
    prompt = f"Question: {q}\nAnswer: {ans}\n\nIs this answer relevant to the question? Reply YES or NO:"
    score = llm.invoke(prompt)
    return {"answer": ans, "reflection": score}

##### Graph build #####
graph = StateGraph(dict)
graph.add_node("PLAN", plan_node)
graph.add_node("RETRIEVE", retrieve_node)
graph.add_node("ANSWER", answer_node)
graph.add_node("REFLECT", reflect_node)

graph.set_entry_point("PLAN")

graph.add_conditional_edges(
    "PLAN",
    lambda out: out["action"],
    {
        "retrieve": "RETRIEVE",
        "answer": "ANSWER"
    }
)

graph.add_edge("RETRIEVE", "ANSWER")
graph.add_edge("ANSWER", "REFLECT")
graph.add_edge("REFLECT", END)

agent = graph.compile()

if __name__ == "__main__":
    while True:
        q = input("\nAsk (exit to stop): ")
        if q.lower() == "exit":
            break
        result = agent.invoke({"question": q})
        print("\nFINAL ANSWER:", result["answer"])
        print("REFLECTION:", result["reflection"])
