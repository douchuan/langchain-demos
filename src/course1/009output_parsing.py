"""
Output Parsing

Demonstrates PydanticOutputParser and JsonOutputParser for structured output.
Shows how to enforce a specific output format from the model using parsers
and format instructions injected into the prompt.
"""


from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field

from utils.config import load_settings

# Load environment variables from .env file and application settings
settings = load_settings()

# Initialize the chat model
model = init_chat_model(
    model=settings.model,
    model_provider="openai",
    base_url=settings.base_url,
    api_key=settings.api_key,
    temperature=0.1,
)


# --- Example 1: PydanticOutputParser ---
# Define the expected output structure using Pydantic
class MovieReview(BaseModel):
    """A structured movie review with rating and analysis."""

    title: str = Field(description="The movie title")
    rating: float = Field(description="Rating from 1.0 to 10.0")
    pros: list[str] = Field(description="List of positive aspects")
    cons: list[str] = Field(description="List of negative aspects")
    summary: str = Field(description="One-sentence summary")


# Create the parser and extract its format instructions
parser = PydanticOutputParser(pydantic_object=MovieReview)

# Inject format instructions into the prompt so the model knows the expected format
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a movie critic. Respond in the specified JSON format.\n\nFORMAT:\n{format_instructions}",
        ),
        ("human", "Review the movie: {movie}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

chain = prompt | model | parser
review = chain.invoke({"movie": "The Matrix"})
print("=== Example 1: PydanticOutputParser ===")
print(f"Title: {review.title}")
print(f"Rating: {review.rating}")
print(f"Pros: {review.pros}")
print(f"Cons: {review.cons}")
print(f"Summary: {review.summary}")


# --- Example 2: JsonOutputParser (simpler, no Pydantic validation) ---
# Use when you want JSON but don't need strict schema validation
class Recipe(BaseModel):
    """A simple recipe structure."""

    name: str
    ingredients: list[str]
    steps: list[str]
    prep_time_minutes: int


parser_json = PydanticOutputParser(pydantic_object=Recipe)

prompt_json = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a chef. Respond in JSON format.\n\nFORMAT:\n{format_instructions}",
        ),
        ("human", "Give me a recipe for {dish}"),
    ]
).partial(format_instructions=parser_json.get_format_instructions())

chain_json = prompt_json | model | parser_json
recipe = chain_json.invoke({"dish": "tomato egg stir-fry"})
print("\n=== Example 2: JsonOutputParser (Pydantic) ===")
print(f"Name: {recipe.name}")
print(f"Ingredients: {', '.join(recipe.ingredients)}")
print(f"Prep time: {recipe.prep_time_minutes} minutes")

# --- Example 3: Raw JsonOutputParser (dict output, no schema) ---
# For when you just want the model's JSON as a Python dict
raw_parser = JsonOutputParser()
raw_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Respond with a JSON object containing 'capital', 'population', and 'language' keys.",
        ),
        ("human", "Tell me about {country}"),
    ]
)

raw_chain = raw_prompt | model | raw_parser
info = raw_chain.invoke({"country": "Japan"})
print("\n=== Example 3: Raw JsonOutputParser ===")
print(info)

# --- Example 4: with_structured_output (modern alternative) ---
# The simplest way to get structured output — built into the model itself
# This is the recommended approach for LangChain 0.2+
model_with_structured = model.with_structured_output(MovieReview)

# Can be used directly or in a chain
structured_chain = prompt | model_with_structured
review2 = structured_chain.invoke(
    {
        "movie": "Inception",
        "format_instructions": parser.get_format_instructions(),
    }
)
print("\n=== Example 4: with_structured_output ===")
print(f"Title: {review2.title}")
print(f"Rating: {review2.rating}")
print(f"Summary: {review2.summary}")
