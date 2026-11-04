from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Annotated, Optional, Literal
from pydantic import BaseModel, Field

"""
In Python and LangChain, Literal acts like a multiple-choice test for the AI.

When you use Literal, you give the AI a strict list of allowed answers. The AI is not allowed to make up its own words, use synonyms, or add extra punctuation—it must pick one of the exact options you provide.
"""

load_dotenv()


model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")


class Review(BaseModel):
    key_themes: list[str] = Field(
        description="write down all the key themes disscused in the review in a list"
    )
    summary: str = Field(description="brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(
        description="Return the sentiment of the review, negative, positive or neutral"
    )
    pros: Optional[list[str]] = Field(
        default=None, description="write down all the pros in a list"
    )
    cons: Optional[list[str]] = Field(
        default=None, description="write down all the cons in a list"
    )
    name: Optional[str] = Field(
        default=None, description="write the name of the reviewer"
    )


structured_model = model.with_structured_output(Review)

result = structured_model.invoke(
    "I bought this desk lamp a few weeks ago for late-night reading, and it has been a great addition to my workspace.Pros: The light brightness is easy to adjust with simple touch controls, and the built-in rechargeable battery lasts for hours without needing to stay plugged in.Cons: The charging cable that comes in the box is a bit too short."
)

print(result)
