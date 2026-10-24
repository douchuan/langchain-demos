"""
Prompt Templates

Demonstrates how to use ChatPromptTemplate with variable substitution,
MessagesPlaceholder for dynamic conversation history, and FewShot examples.
This is the foundational building block for constructing prompts in LangChain.
"""

from langchain.chat_models import init_chat_model
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)

from utils.config import load_settings

# Load environment variables from .env file and application settings
settings = load_settings()

# Initialize the chat model with OpenAI-compatible API configuration
model = init_chat_model(
    model=settings.model,
    model_provider="openai",
    base_url=settings.base_url,
    api_key=settings.api_key,
    temperature=0.1,
)

# --- Example 1: Simple template with variable substitution ---
# {topic} and {language} are placeholders filled at invoke time
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant specializing in {topic}."),
        ("human", "Explain {concept} in {language}."),
    ]
)

# Format the prompt and invoke the model
chain = prompt | model
response = chain.invoke(
    {
        "topic": "programming",
        "concept": "list comprehension",
        "language": "Chinese",
    }
)
print("=== Example 1: Simple Template ===")
print(response.content)

# --- Example 2: Template with conversation history (MessagesPlaceholder) ---
# MessagesPlaceholder inserts a list of prior messages at this position
prompt_with_history = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful programming tutor."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)

# Simulate a conversation history
from langchain.messages import AIMessage, HumanMessage

history = [
    HumanMessage("What is Python?"),
    AIMessage("Python is a high-level programming language known for its readability."),
]

chain_with_history = prompt_with_history | model
response = chain_with_history.invoke(
    {
        "history": history,
        "input": "What are its main use cases?",
    }
)
print("\n=== Example 2: Template with History ===")
print(response.content)

# --- Example 3: Few-shot prompting ---
# Provide examples to guide the model's response format
examples = [
    {"input": "Happy", "output": "This person is feeling joyful and content."},
    {"input": "Furious", "output": "This person is experiencing intense anger."},
    {"input": "Melancholy", "output": "This person feels a deep, pensive sadness."},
]

# Template for each individual example
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "Word: {input}"),
        ("ai", "{output}"),
    ]
)

# Few-shot prompt that formats all examples into the conversation
few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)

# Combine few-shot examples with the main prompt
final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Classify the emotion expressed in each word. Respond in the format shown.",
        ),
        few_shot_prompt,
        ("human", "Word: {word}"),
    ]
)

chain_few_shot = final_prompt | model
response = chain_few_shot.invoke({"word": "Ecstatic"})
print("\n=== Example 3: Few-Shot Prompting ===")
print(response.content)
