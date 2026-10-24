"""
Middleware

dynamic_prompt
"""

from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_openai import ChatOpenAI

from utils.config import load_settings

settings = load_settings()


@dataclass
class Context:
    user_role: str


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role

    base_prompt = "You are a helpful and very concise assistant."

    match user_role:
        case "expert":
            return f"{base_prompt} Provide detail technical responses."
        case "beginner":
            return f"{base_prompt} Keep your explanations simple and basic."
        case "child":
            return f"{base_prompt} Explain everyting as if your were literally talking to a five-year old."
        case _:
            return base_prompt


client = ChatOpenAI(
    api_key=settings.api_key,
    model=settings.model,
    base_url=settings.base_url,
)

agent = create_agent(
    model=client, middleware=[user_role_prompt], context_schema=Context
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain PCA."}]},
    context=Context(user_role="child"),
)

print(response)
