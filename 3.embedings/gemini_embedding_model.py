import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()


embeddings = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-2-preview")

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "Tell me about bumrah"

doc_embeddings = embeddings.embed_documents(documents)
query_embeddings = embeddings.embed_query(query)

scores = cosine_similarity([query_embeddings],doc_embeddings)[0]

index, score = max(enumerate(scores), key=lambda x: x[1])

print("query : ", query)
print("matched documnets : ",documents[index])
print("similarity scores : ", score)



