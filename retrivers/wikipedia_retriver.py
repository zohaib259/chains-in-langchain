import wikipedia  # Import the raw wikipedia package first
from langchain_community.retrievers import WikipediaRetriever

# 1. Set a User-Agent so Wikipedia's API doesn't block or return empty responses
wikipedia.set_user_agent(
    "MyLangChainRAGApp/1.0 (zohaibafridi1234567@example.com)"
)  # You can put any app name/email here

# 2. Initialize the Wikipedia Retriever
retriever = WikipediaRetriever(top_k_results=2, lang="en")

# 3. Define your search query
query = "Jasprit Bumrah"

# 4. Invoke the retriever
docs = retriever.invoke(query) # here we are using .invoke function on retriver which mean it is runnable

# 5. Print out the results
for i, doc in enumerate(docs):
  print(f"\n--- Result {i+1} ---")
  print(f"Source/Title Metadata: {doc.metadata}")
  print(doc.page_content[:500])