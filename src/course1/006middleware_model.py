"""
Middleware

dynamic model selection
"""

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from utils.config import load_settings

settings = load_settings()

basic_model = ChatOpenAI(
    api_key=settings.api_key,
    model=settings.model,
    base_url=settings.base_url,
)
advance_model = ChatOpenAI(
    api_key=settings.api_key,
    model=settings.model2,
    base_url=settings.base_url,
)


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    message_count = len(request.state["messages"])

    if message_count > 3:
        model = advance_model
    else:
        model = basic_model

    request.model = model

    return handler(request)


agent = create_agent(model=basic_model, middleware=[dynamic_model_selection])

response = agent.invoke(
    {
        "messages": [
            SystemMessage("Your are a helpful assistant."),
            HumanMessage("What is 1+1?"),
            # HumanMessage("What is 1+1?"),
            # HumanMessage("What is 1+1?"),
            # HumanMessage("What is 1+1?"),
        ]
    },
)

print(response["messages"][-1].content)
print(response["messages"][-1].response_metadata["model_name"])
