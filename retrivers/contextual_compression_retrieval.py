import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
load_dotenv()

docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

     Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"}),
]
 
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="compression_test"
)
base_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# 3. Initialize valid LLM for the compressor
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
compressor = LLMChainExtractor.from_llm(llm)

# 4. Wrap with Contextual Compression Retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 5. Run a query to extract only the relevant sentences
query = "How does photosynthesis work?"
compressed_docs = compression_retriever.invoke(query)

for i, doc in enumerate(compressed_docs):
    print(f"\n--- Compressed Result {i+1} ({doc.metadata['source']}) ---")
    print(doc.page_content)