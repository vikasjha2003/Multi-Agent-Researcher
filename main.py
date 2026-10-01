# Imports
import os
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph , START, END
from langchain_groq import ChatGroq

import tools as custom_tools
import prompts
from dotenv import load_dotenv
load_dotenv()

# tools
tools = custom_tools.tool_list

# States
class State(TypedDict) :
    topic : str
    messages : Annotated[list,add_messages]
    searcher : str
    reader : str
    writer : str
    critic : str

# Mdoel / llm
llm = ChatGroq(model="openai/gpt-oss-120b")
llm_with_tools = llm.bind_tools(tools=tools)

# Nodes
tool_node = ToolNode(tools=tools)

def searcher_node(state: State):
    topic = state["topic"]

    messages = prompts.SEARCH_PROMPT.invoke({
        "topic": topic
    })

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }

# def reader_node(state: State) :
#     topic = state["searcher"]



# Graph
graph = StateGraph(State)

graph.add_node("searcher",searcher_node)
graph.add_node("tools",tool_node)
# graph.add_node("reader",reader_node)

graph.add_edge(START,"searcher")
graph.add_conditional_edges("searcher",tools_condition)
graph.add_edge("searcher","tools")
# graph.add_edge("tools","reader")

app = graph.compile()

# testing
initial_state = {
    "topic": "Impact of AI agents on software development",
    "messages": [],
    "searcher": "",
    "reader": "",
    "writer": "",
    "critic": ""
}

result = app.invoke(initial_state)

print(result["messages"])