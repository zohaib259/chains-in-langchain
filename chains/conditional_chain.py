# from dotenv import load_dotenv
# from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
# from langchain_google_genai import ChatGoogleGenerativeAI
# from pydantic import BaseModel, Field
# from typing import Literal

# load_dotenv()

# # 1. Use a valid model name
# model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")


# class Feedback(BaseModel):
#     sentiment: Literal["positive", "negative"] = Field(
#         description="provide the sentiment on the feedback"
#     )


# parser = StrOutputParser()
# parser2 = PydanticOutputParser(pydantic_object=Feedback)

# prompt = PromptTemplate(
#     template="Classify the {feedback} into positive or negative sentiment.\n{format_instructions}",
#     input_variables=["feedback"],
#     partial_variables={"format_instructions": parser2.get_format_instructions()},
# )

# classifier_chain = prompt | model | parser2

# prompt2 = PromptTemplate(
#     template="Write exactly ONE professional response directly to the user for this positive feedback. "
#     "Do not provide options, variations, or bullet points. Output only the single final message.\n\nFeedback: {feedback}",
#     input_variables=["feedback"],
# )

# prompt3 = PromptTemplate(
#     template="Write exactly ONE professional response directly to the user for this negative feedback. "
#     "Do not provide options, variations, or bullet points. Output only the single final message.\n\nFeedback: {feedback}",
#     input_variables=["feedback"],
# )

# # 3. Combined input dictionary with sentiment classification
# chain_input = {
#     "feedback": RunnablePassthrough(),
#     "sentiment_obj": classifier_chain,
# }

# branch_chain = RunnableBranch(
#     (lambda x: x["sentiment_obj"].sentiment == "positive", prompt2 | model | parser),
#     (lambda x: x["sentiment_obj"].sentiment == "negative", prompt3 | model | parser),
#     RunnableLambda(lambda x: "could not find sentiment"),
# )

# full_chain = chain_input | branch_chain

# # 4. Execute the chain
# result = full_chain.invoke(
#     {
#         "feedback": "Great app for gym members! The QR check-in and class booking features work seamlessly. Payment tracking and auto-renewals run smoothly. Would love to see Apple Watch / Wear OS sync for workout tracking in a future update!"
#     }
# )

# print(result)





from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# 1. Use a valid model name
model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")


# --- Step 1: Sentiment Classification Setup ---
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Provide the sentiment of the feedback"
    )

sentiment_parser = PydanticOutputParser(pydantic_object=Feedback)

classifier_prompt = PromptTemplate(
    template="Classify the following feedback into positive or negative sentiment.\n{format_instructions}\n\nFeedback: {feedback}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": sentiment_parser.get_format_instructions()},
)

classifier_chain = classifier_prompt | model | sentiment_parser


# --- Step 2: Response Generation Prompts ---
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


# --- Step 3: Chain Construction ---
# Fix: Extract the string explicitly for "feedback" so downstream prompts receive a string, not a dict.
chain_input = {
    "feedback": lambda x: x["feedback"],
    "sentiment_obj": classifier_chain,
}


class FeedbackResponse(BaseModel):
    response: str = Field(description="Write a professional response to the feedback")

response_parser = PydanticOutputParser(pydantic_object=FeedbackResponse)

# Update prompts to include format instructions for the final response parser
prompt2_formatted = prompt2.partial(format_instructions=response_parser.get_format_instructions())
prompt3_formatted = prompt3.partial(format_instructions=response_parser.get_format_instructions())

branch_chain = RunnableBranch(
    (lambda x: x["sentiment_obj"].sentiment == "positive", prompt2_formatted | model | response_parser),
    (lambda x: x["sentiment_obj"].sentiment == "negative", prompt3_formatted | model | response_parser),
    RunnableLambda(lambda x: "Could not find sentiment"),
)

full_chain = chain_input | branch_chain


# --- Step 4: Execution ---
result = full_chain.invoke(
    {
        "feedback": "Great app for gym members! The QR check-in and class booking features work seamlessly. Payment tracking and auto-renewals run smoothly. Would love to see Apple Watch / Wear OS sync for workout tracking in a future update!"
    }
)

# Since result is a Pydantic object, access the string via .response
print(result.response)