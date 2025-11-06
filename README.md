
# LangGraph-driven RAG Agent powered by Gemini + Pinecone

## 1. Problem Definition & Objective

Modern LLMs are powerful but they hallucinate and cannot recall external knowledge unless we feed it every time.
RAG (Retrieval Augmented Generation) solves this by retrieving relevant knowledge from a database and giving it to LLM.

**Goal of this project:**
Build a question-answering AI Agent that uses a small knowledge base stored locally and answers queries through a RAG pipeline.

---

## 2. High-Level Overview of the System

This system behaves like an intelligent agent.
When a user asks a question, instead of directly generating an answer, the system:

1. **Understands** the query and decides whether retrieval from a knowledge base is required.
2. **Retrieves** relevant context chunks from Pinecone vector DB.
3. **Generates** a grounded answer using Google Gemini.
4. **Reflects** and validates if the answer truly matches the question.

This ensures answers are relevant, grounded, and not hallucinations.

---

## 3. Overall Approach

| Step                 | Description                                                       |
| -------------------- | ----------------------------------------------------------------- |
| Data Preparation     | Small `.txt` files placed inside `/data` folder                   |
| Embedding & Indexing | Convert text chunks → embeddings → stored in Pinecone             |
| Agent Graph Creation | 4 nodes created via LangGraph: PLAN → RETRIEVE → ANSWER → REFLECT |
| Interactive UI       | Users ask questions via Streamlit                                 |
| Evaluation           | Gemini is used as a Judge to validate answers (YES/NO)            |

This modular approach makes the system interpretable, testable, traceable and aligns with modern enterprise RAG agent standards.

---

## 4. Tech Stack Used

| Component               | Technology                |
| ----------------------- | ------------------------- |
| Orchestration Framework | **LangGraph**             |
| LLM + Embeddings        | **Gemini / Google GenAI** |
| Vector Database         | **Pinecone**              |
| UI                      | **Streamlit**             |
| Evaluation              | **LLM-as-a-Judge**        |

---

## 5. Architecture Diagram

```
   ┌─────────┐
   │ PLAN    │  → decides whether to retrieve
   └────┬────┘
        │
   ┌────▼─────┐
   │ RETRIEVE │  → Pinecone vector search
   └────┬─────┘
        │
   ┌────▼────┐
   │ ANSWER  │  → Gemini LLM builds final response
   └────┬────┘
        │
   ┌────▼─────┐
   │ REFLECT  │  → Gemini LLM checks correctness
   └──────────┘
        ↓
   Final Answer
```

---

## 6. Why Gemini instead of OpenAI?

OpenAI quota was exhausted during development.
Switched to Gemini which supports:

* text generation
* embeddings creation

in a single ecosystem.

---

## 7. Dataset Used

Generic test text stored inside `/data`.
The dataset is intentionally small because the focus is on RAG pipeline correctness.

---

## 8. Project Structure

```
rag-agent/
├── data/                  # text files used for embeddings
├── src/
│   ├── ingest.py          # embeds text → pushes to Pinecone
│   ├── agent.py           # LangGraph CLI agent
│   ├── app.py             # Streamlit UI
│   └── evaluate.py        # evaluation script (LLM-as-a-Judge)
├── requirements.txt
├── .env
└── README.md
```

---

## 9. Setup Instructions

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put keys in `.env`:

```
PINECONE_API_KEY=xxxx
PINECONE_INDEX_NAME=my-rag-index
GEMINI_API_KEY=xxxx
```

---

## 10. Running the System

### 1) Ingest Data

```bash
python src/ingest.py
```

### 2) Run Agent in Terminal

```bash
python src/agent.py
```

### 3) Launch Streamlit App

```bash
python -m streamlit run src/app.py
```

### 4) Evaluate Answers (LLM-as-Judge)

```bash
python src/evaluate.py
```

---

## 11. Evaluation Approach

We implemented a simple and effective evaluation method using **LLM-as-a-Judge**.

* Input: question + model’s generated answer
* Output: YES / NO + justification

This method is valid, accepted in industry and suggested directly in assignment PDF.

---

## 12. Challenges Faced

| Challenge                             | Fix                                      |
| ------------------------------------- | ---------------------------------------- |
| OpenAI credits finished               | moved to Gemini                          |
| LangGraph changed API                 | updated set_entrypoint → set_entry_point |
| State lost between nodes              | returned question + logs on every step   |
| Multi-line input in powershell breaks | restricted evaluation to 1-line answers  |

---

## 13. Conclusion

This project demonstrates a fully working RAG Agent with:

* vector retrieval
* reasoning graph
* answer generation
* reflection evaluation
* interactive user interface


