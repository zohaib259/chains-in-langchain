from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# 1. Use a valid model name
model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")


class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="provide the sentiment on the feedback"
    )


parser = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt = PromptTemplate(
    template="Classify the {feedback} into positive or negative sentiment.\n{format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()},
)

classifier_chain = prompt | model | parser2

prompt2 = PromptTemplate(
    template="Write exactly ONE professional response directly to the user for this positive feedback. "
    "Do not provide options, variations, or bullet points. Output only the single final message.\n\nFeedback: {feedback}",
    input_variables=["feedback"],
)

prompt3 = PromptTemplate(
    template="Write exactly ONE professional response directly to the user for this negative feedback. "
    "Do not provide options, variations, or bullet points. Output only the single final message.\n\nFeedback: {feedback}",
    input_variables=["feedback"],
)

# 3. Combined input dictionary with sentiment classification
chain_input = {
    "feedback": RunnablePassthrough(),
    "sentiment_obj": classifier_chain,
}

branch_chain = RunnableBranch(
    (lambda x: x["sentiment_obj"].sentiment == "positive", prompt2 | model | parser),
    (lambda x: x["sentiment_obj"].sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment"),
)

full_chain = chain_input | branch_chain

# 4. Execute the chain
result = full_chain.invoke(
    {
        "feedback": "Great app for gym members! The QR check-in and class booking features work seamlessly. Payment tracking and auto-renewals run smoothly. Would love to see Apple Watch / Wear OS sync for workout tracking in a future update!"
    }
)

print(result)
