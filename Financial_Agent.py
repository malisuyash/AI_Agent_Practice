from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
from phi.model.openai import OpenAIChat
import os
import openai

# Load .env variables
load_dotenv()

openai.api_key = os.getenv("OpenAI_API_Key")
Groq.api_key = os.getenv("GROQ_API_KEY")

# Agents
Web_Search_Agent = Agent(
    name="Web Searcher",
    role="Search web for the information",
    tools=[DuckDuckGo()],
    model=Groq(id="llama-3.1-8b-instant"),
    show_tool_calls=True,
    instructions=["Always include Sources"],
    markdown=True,
)

Financial_Agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama-3.1-8b-instant"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
    instructions=["Format your response using markdown and use tables to display data where possible."],
    show_tool_calls=True,
    markdown=True,
)

Multi_AI_Agent = Agent(
    name="Agents Team",
    team=[Web_Search_Agent, Financial_Agent],
    instructions=[
        "First, ask the web searcher to search for each story to get more information.",
        "Finally, provide a thoughtful and engaging summary.",
        "Then, use tables to display the data."
    ],
    show_tool_calls=True,
    markdown=True,
)

# Run the agent
Multi_AI_Agent.print_response("Summarize analyst recommendations and share the latest news for NVDA", stream=True)
