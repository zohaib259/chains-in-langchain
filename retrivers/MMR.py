"""
Simple retriever will sometimes give you redundant results.

With MMR you will not only get relevant results, but you will also 
get diverse results instead of redundant ones.
"""

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

# 1. Initialize the correct Gemini embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# 2. Create an in-memory or persistent Chroma vector store
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="my_mmr_db",
    collection_name="mmr_test_collection"
)

# 3. Create sample documents with intentional redundancy to test MMR diversity
docs = [
    # Document 1: Focus on action (Very similar to Doc 2)
    Document(page_content="Jasprit Bumrah has a very unique, unorthodox, and hyper-extended bowling action with a short, stuttering run-up and a slinging arm delivery."),
    
    # Document 2: Redundant with Doc 1 (Focus on action and release)
    Document(page_content="The key to Bumrah's bowling is his unusual slinging arm action and short run-up, which makes it hard for batters to pick the ball early."),
    
    # Document 3: Focus on IPL / Mumbai Indians (Distinct topic)
    Document(page_content="Jasprit Bumrah plays for the Mumbai Indians in the Indian Premier League (IPL) and is a master of bowling during the death overs."),
    
    # Document 4: Redundant with Doc 3 (Focus on IPL death overs)
    Document(page_content="In the IPL, Bumrah has been the cornerstone of Mumbai Indians' bowling attack, consistently choking run rates in the death overs."),
    
    # Document 5: Focus on Test cricket / SENA countries (Completely distinct topic)
    Document(page_content="In Test cricket, Bumrah has achieved incredible success, delivering historic multi-wicket spells and match-winning performances across SENA countries.")
]

# Add documents to the already initialized vector store
vector_store.add_documents(documents=docs)

# Create the MMR retriever (fixed spelling: as_retriever)
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.5}
)

# 4. Define a query
query = "Tell me about Jasprit Bumrah's bowling action and his performance in Test matches."

result = retriever.invoke(query)

for i, doc in enumerate(result):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)