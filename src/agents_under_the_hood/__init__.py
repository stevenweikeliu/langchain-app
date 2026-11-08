import os

from langchain_ollama import ChatOllama

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_RETRIES = 10
MODEL = "qwen3:1.7b"

# --- Tools (Langchain @tool decorator)---
@tool
def get_product_price(product: str) -> float:
    """Look up the price of a given product"""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {
        "laptop": 1299.95,
        "headphone": 149.95,
        "keyboard": 69.99,
    }
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount tire to a price and return the final price.
    Available tiers: bronze, silver, gold
    """
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discounts_percentages = {
        'bronze': 5,
        'silver': 12,
        'gold': 23,
    }
    discount = discounts_percentages.get(discount_tier, 0)
    return round(price * (100 - discount) / 100, 2)

# --- Agent loop ---
@traceable(name = "LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    llm = init_chat_model(f"ollama:{MODEL}", temperature = 0)
    llm_with_tools = llm.bind_tools(tools)
    print(f"Question: {question}")
    print("=" * 60)
    messages = [
        SystemMessage(
            content = (
                "You are a helpful shopping assistant."
                "You have access to a product catalog tool and a discount tool.\n\n"
                "STRICT RULES - you must follow these exactly: \n"
                "1. NEVER guess the product price."
                "you MUST call get_product_price to get the real price\n\n"
                "2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price "
                " returned by get_product_price - do NOT pass made-up numbers.\n"
                "3. NEVER calculate the price by youself, always use the apply_discount."
                "4. If the user does not give you the product, NEVER assume one."
            )
        ),
        HumanMessage(content = question),
    ]
    for iteration in range(1, MAX_RETRIES + 1):
        print(f"\n--- Iteration {iteration} ---")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
        
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"   [Tool Selected] {tool_name} with args: {tool_args}")
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        observation = tool_to_use.invoke(tool_args)

        print(f"   [Tool Observation] {observation}")
        messages.append(ai_message)
        messages.append(ToolMessage(content = str(observation), tool_call_id = tool_call_id))

    print(f"Error: reached max retries during reasoning")
    return None

def main() -> None:
    """
    Entry point of the agents_under_the_hood
    """
    print("Hello I am your E-commerce assistant")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")

if __name__ == '__main__':
    main()
