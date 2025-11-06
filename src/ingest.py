import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

load_dotenv()

# load env vars
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# init pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# create index if not exists
if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
    name=INDEX_NAME,
    dimension=768,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)


index = pc.Index(INDEX_NAME)

# embeddings
embed_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# load all txt files
docs = []
for f in Path("data").glob("*.txt"):
    text = f.read_text(encoding="utf-8")
    docs.append(Document(page_content=text, metadata={"source": f.name}))

# split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# embed + upsert
vectors = []
for i, chunk in enumerate(chunks):
    emb = embed_model.embed_query(chunk.page_content)  # create embedding
    vectors.append({"id": f"doc-{i}", "values": emb, "metadata": chunk.metadata})

index.upsert(vectors=vectors)

print(f"✅ Ingested {len(vectors)} chunks into Pinecone index: {INDEX_NAME}")
