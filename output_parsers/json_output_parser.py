import os
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()

# 1. Initialize the model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")



# 2. Define the desired data structure using Pydantic
class TopicDetails(BaseModel):
  topic: str = Field(description="The name of the technology or topic")
  difficulty: str = Field(
      description="Beginner, Intermediate, or Advanced level"
  )
  key_features: list[str] = Field(description="List of 3 key features or traits")


# 3. Initialize the JsonOutputParser with your Pydantic schema
output_parser = JsonOutputParser(pydantic_object=TopicDetails)

# 4. Create a PromptTemplate that includes format instructions
prompt = PromptTemplate(
    template=(
        "Provide information about the given topic.\n{format_instructions}\nTopic:"
        " {topic}"
    ),
    input_variables=["topic"],
    partial_variables={
        "format_instructions": output_parser.get_format_instructions()
    },
)

# 5. Build the LCEL chain
chain = prompt | model | output_parser

# 6. Invoke the chain
result = chain.invoke({"topic": "FastAPI"})

print(result)
print(type(result))  # Output: <class 'dict'>