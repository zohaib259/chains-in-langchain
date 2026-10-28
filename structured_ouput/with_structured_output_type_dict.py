"""
Structured output in LangChain allows you to force a Large Language Model (LLM) to return its response in a predictable, strongly typed format (such as a JSON object or a Python class instance) rather than unstructured free-form text. This eliminates fragile string manipulation and regex parsing in downstream applications.


1. Autonomous AI Agents & Tool Calling
Enabling LLMs to reliably control software and APIs.

API Integration: Forcing the LLM to format its response as a valid JSON payload required to call external APIs (e.g., booking a flight, sending an email, or querying weather data).

Multi-Step Workflows: Generating specific commands/actions that a code interpreter or browser agent can execute safely without failing on syntax errors.

2. Application UI & Front-End Integration
Powering dynamic user interfaces directly from LLM responses.

Interactive Dashboards: Returning structured JSON to build dynamic charts, tables, or cards on the fly.

E-Commerce Search: Converting natural language user requests ("Show me blue running shoes under $100") into direct search filter objects: {"color": "blue", "category": "running", "max_price": 100}.


3. Classification & Content Moderation
Standardizing decisions into strict enums or pre-defined categories.

Support Ticket Routing: Mapping incoming support tickets to fixed departments (e.g., ["Billing", "Tech Support", "Sales"]) along with a priority score (1-5).

Content Moderation: Flagging user posts with boolean flags (contains_hate_speech: true/false, confidence: 0.95).

etc etc
"""


from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel
"""
In Python and LangChain, Literal acts like a multiple-choice test for the AI.

When you use Literal, you give the AI a strict list of allowed answers. The AI is not allowed to make up its own words, use synonyms, or add extra punctuation—it must pick one of the exact options you provide.
"""

load_dotenv()


model = ChatGoogleGenerativeAI(model = "gemini-3.8-flash")

class Review(TypedDict):
    key_themes : Annotated[str,"write down all the key themes disscused in the review in a list"]
    summary : Annotated[str,"brief summary of the review"]
    sentiment : Annotated[Literal[str,]"Return the sentiment of the review, negative , positive or neutral"]
    pros: Annotated[Optional[list[str]], "write down all the pros in a list"]
    cons: Annotated[Optional[list[str]], "write down all the cons in a list"]
    name:Annotated[Optional[str],"write the name of the reviewer"]
    
structured_model = model.with_structured_output(Review)

result = structured_model.invoke("I bought this desk lamp a few weeks ago for late-night reading, and it has been a great addition to my workspace.Pros: The light brightness is easy to adjust with simple touch controls, and the built-in rechargeable battery lasts for hours without needing to stay plugged in.Cons: The charging cable that comes in the box is a bit too short.")

print(result)