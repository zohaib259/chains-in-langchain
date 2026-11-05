import os
from dotenv import load_dotenv
from langchain_core.documents import Document 
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# 1. Initialize Google Embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# 2. Define your documents
doc1 = Document(
    page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
    metadata={"team": "Royal Challengers Bangalore"},
)
doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
    metadata={"team": "Mumbai Indians"},
)
doc3 = Document(
    page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
    metadata={"team": "Chennai Super Kings"},
)
doc4 = Document(
    page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
    metadata={"team": "Mumbai Indians"},
)
doc5 = Document(
    page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
    metadata={"team": "Chennai Super Kings"},
)

docs = [doc1, doc2, doc3, doc4, doc5]

vector_store = Chroma.from_documents(docs, embeddings)

query = "Who is a great fast bowler for death overs?"
results = vector_store.similarity_search(query, k=1)


retriever = vector_store.as_retriever(search_kwargs={"k":2})

query = "who is jasprit bumrah"

result = retriever.invoke(query)
print(result)
"""
If we have vector_store.similarity_search() then why use retriver both give the same output?

yes both return same output but in similarity_search() we only can use 1 strategy [by companring the query vector with all the other vectors],
while in retriever() we can use different different advance search strategies to connect with retriver().

"""