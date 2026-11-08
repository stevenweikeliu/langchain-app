import os

from typing import List
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_core.messages import HumanMessage
# from langchain_core.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from tavily import TavilyClient
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

class Source(BaseModel):
    """
    Schema for a source used by the agent
    """
    url: str = Field(dsecription = "The URL of the source")

class AgentResponse(BaseModel):
    """
    Schema for agent response with oanswer and sources
    """
    answer: str = Field(description="The agent's response")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")


tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """
    Tool that searches over Internet
    Args:
       query: The query to search for
    Returns:
       The search result
    """
    print(f"Search for {query}")
    return tavily.search(query = query)

def main() -> None:
    # llm = ChatOpenAI(temperature=0, model="gpt-5")
    llm = ChatOllama(temperature=0, model="gemma4")
    # tools = [search]
    better_tools = [TavilySearch(max_results=3)]
    agent = create_agent(
        model = llm,
        tools = better_tools,
        response_format = ToolStrategy(AgentResponse),
        system_prompt=(
            "you are a job search assistant."
            "Parse the results into AgentResponse as structured response field in the result" # I have to include this prompt to force gemma4 to parse the results into AgentResponse pydantic object
            "search at most 3 times."
            "You should only search on linkedin."
            "once you have results, immediately provide your final answer."
        ))
    query = HumanMessage(content = "search for maximum 3 ai engineers/software engineers jobs. Title: software engineers or AI engineers; Location: Sydney Australia")
    result = agent.invoke({"messages":[query]})
    print(result["structured_response"])


if __name__ == "__main__":
    main()
