import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint

# Load environment variables from .env file
load_dotenv()

# Option 1: Set token via code if not using a .env file
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_huggingface_token_here"

# Initialize an open-source model via Hugging Face Serverless Inference API
# We use 'mistralai/Mistral-7B-Instruct-v0.2' as an open-source model example
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=100,
    temperature=0.7,
)

# Define prompt
prompt = "How are you?"

# Send prompt to the model
response = llm.invoke(prompt)

# Print response
print("Model Response:")
print(response)