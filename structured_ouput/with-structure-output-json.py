"""we use json schema format when our project is built on defferent tech programming languages not just one"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

# Define the schema using standard JSON Schema format
review_schema = {
    "title": "Review",
    "description": "Schema for analyzing customer product reviews.",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "string",
            "description": "Write down all the key themes discussed in the review in a list",
        },
        "summary": {
            "type": "string",
            "description": "Brief summary of the review",
        },
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"],
            "description": "Return the sentiment of the review",
        },
        "pros": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Write down all the pros in a list",
        },
        "cons": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Write down all the cons in a list",
        },
        "name": {
            "type": ["string", "null"],
            "description": "Write the name of the reviewer",
        },
    },
    "required": ["key_themes", "summary", "sentiment", "pros", "cons", "name"],
}

# Pass the schema directly into with_structured_output
structured_model = model.with_structured_output(review_schema)

result = structured_model.invoke(
    "I bought this desk lamp a few weeks ago for late-night reading, and it has been a great addition to my workspace. Pros: The light brightness is easy to adjust with simple touch controls, and the built-in rechargeable battery lasts for hours without needing to stay plugged in. Cons: The charging cable that comes in the box is a bit too short."
)

print(result)