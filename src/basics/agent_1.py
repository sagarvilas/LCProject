from langchain.agents import create_agent
import os
from getpass import getpass

if "ANTHROPIC_API_KEY" not in os.environ:
    print("Please set the environment variable 'ANTHROPIC_API_KEY'")
    os.environ["ANTHROPIC_API_KEY"] = getpass()

def get_weather(city: str) -> str:
    """A mock function to get weather information for a given city."""
    return f"The weather in {city} is sunny!"

agent = create_agent(
    model = "anthropic:claude-sonnet-4-5",
    tools = [get_weather],
    system_prompt = "You're a helpful assistant that provides weather information.",
    debug = True
)

agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in New York?"}]}
)