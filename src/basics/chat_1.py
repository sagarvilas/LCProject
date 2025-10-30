from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv(encoding='utf-8-sig')

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

model = ChatAnthropic(model="claude-sonnet-4-5")

chain = prompt | model

chain.invoke({"question": "What is LangChain?"})