import os
from dotenv import load_dotenv
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

# -------------------------
# Load API Key
# -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------------
# LLM
# -------------------------
llm = ChatOpenAI(
    temperature=0.5,
    model="gpt-4o"
)

# -------------------------
# State Definition
# -------------------------
class ChatState(TypedDict):
    user_input: str
    response: str
    category: str

# -------------------------
# Node 1: Classify Input
# -------------------------
def classify_input(state: ChatState):
    text = state["user_input"].lower()

    if "hello" in text or "hi" in text:
        category = "greeting"
    elif "skill" in text or "career" in text:
        category = "career"
    else:
        category = "general"

    return {"user_input": state["user_input"], "category": category}

# -------------------------
# Node 2: Generate Response
# -------------------------
def generate_response(state):
    category = state["category"]
    user_text = state["user_input"]

    if category == "greeting":
        prompt = f"Reply politely to: {user_text}"
    elif category == "career":
        prompt = f"Suggest career advice for: {user_text}"
    else:
        prompt = f"Answer this question clearly: {user_text}"

    response = llm.invoke(prompt)

    return {
        "response": response.content
    }

# -------------------------
# Build LangGraph
# -------------------------
graph = StateGraph(ChatState)

graph.add_node("classifier", classify_input)
graph.add_node("responder", generate_response)

graph.set_entry_point("classifier")
graph.add_edge("classifier", "responder")
graph.add_edge("responder", END)

app = graph.compile()

# -------------------------
# Chat Interface (Terminal)
# -------------------------
if __name__ == "__main__":
    print("🤖 LangGraph Chatbot (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break

        result = app.invoke({"user_input": user_input})
        print("\nBot:", result["response"])
       

