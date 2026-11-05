"""
Simple retriever will sometimes give you redundant results.

With MMR you will not only get relevant results, but you will also
get diverse results instead of redundant ones.
"""
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

# 1. Initialize the correct Gemini embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# 2. Create an in-memory or persistent Chroma vector store
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="my_mmr_db",
    collection_name="mmr_test_collection",
)

# Relevant health & wellness documents
all_docs = [
    Document(
        page_content="Regular walking boosts heart health and can reduce symptoms of depression.",
        metadata={"source": "H1"},
    ),
    Document(
        page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.",
        metadata={"source": "H2"},
    ),
    Document(
        page_content="Deep sleep is crucial for cellular repair and emotional regulation.",
        metadata={"source": "H3"},
    ),
    Document(
        page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.",
        metadata={"source": "H4"},
    ),
    Document(
        page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.",
        metadata={"source": "H5"},
    ),
    Document(
        page_content="The solar energy system in modern homes helps balance electricity demand.",
        metadata={"source": "I1"},
    ),
    Document(
        page_content="Python balances readability with power, making it a popular system design language.",
        metadata={"source": "I2"},
    ),
    Document(
        page_content="Photosynthesis enables plants to produce energy by converting sunlight.",
        metadata={"source": "I3"},
    ),
    Document(
        page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.",
        metadata={"source": "I4"},
    ),
    Document(
        page_content="Black holes bend spacetime and store immense gravitational energy.",
        metadata={"source": "I5"},
    ),
]

# Add documents to the initialized vector store
vector_store.add_documents(documents=all_docs)

# 3. Configure MMR (Maximal Marginal Relevance) Retriever
mmr_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5}
)

# 4. Define a query relevant to your corpus (or keep custom query)
query = "How to improve mental health and reduce stress naturally?"

# 5. Initialize MultiQueryRetriever using MMR retriever
MQR = MultiQueryRetriever.from_llm(
    retriever=mmr_retriever,
    llm=ChatGoogleGenerativeAI(model="gemini-3.8-flash")
)

# 6. Invoke MultiQueryRetriever
result = MQR.invoke(query)

for i, doc in enumerate(result):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)