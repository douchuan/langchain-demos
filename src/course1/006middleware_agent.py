import time

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from utils.config import load_settings


class HooksDemo(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.start_time = 0.0

    def before_agent(self, state: AgentState, runtime):
        self.start_time = time.time()
        print("before_agent triggered")

    def before_model(self, state: AgentState, runtime):
        print("before_model triggered")

    def after_model(self, state: AgentState, runtime):
        print("after_model triggered")

    def after_agent(self, state: AgentState, runtime):
        print("after_agent:", time.time() - self.start_time)


settings = load_settings()

basic_model = ChatOpenAI(
    api_key=settings.api_key,
    model=settings.model,
    base_url=settings.base_url,
)

agent = create_agent(model=basic_model, middleware=[HooksDemo()])

response = agent.invoke(
    {
        "messages": [
            SystemMessage("You are a helpful assistant."),
            HumanMessage("What is PCA ?"),
        ]
    },
)

print(response["messages"][-1].content)
