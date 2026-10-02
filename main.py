# Imports
import os
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph , START, END
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage
from rich import print

import tools as custom_tools
import prompts
from dotenv import load_dotenv
load_dotenv()

#tools

searcher_tools = [custom_tools.web_search]
reader_tools = [custom_tools.scrape_url]

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
searcher_llm = llm.bind_tools(tools=searcher_tools)
reader_llm = llm.bind_tools(tools=reader_tools)

# Nodes
def searcher_node(state: State):
    topic = state["topic"]

    messages = prompts.SEARCH_PROMPT.invoke({
        "topic": topic
    })

    response = searcher_llm.invoke(messages)

    return {
        "messages": [response]
    }

searcher_tool_node = ToolNode(tools=searcher_tools)

def extract_search_results_node(state: State):

    search_results = []

    for message in state["messages"]:
        if isinstance(message, ToolMessage):
            search_results.append(message.content)

    return {
        "searcher": "\n\n".join(search_results)
    }

def reader_node(state: State) :
    topic = state["topic"]
    search_results = state["searcher"]

    messages = prompts.READER_PROMPT.invoke({
        "topic" : topic,
        "search_results" : search_results
    })

    response = reader_llm.invoke(messages)

    return {
        "messages" : [response]
    }

reader_tool_node = ToolNode(tools=reader_tools)

def extract_reader_results_node(state: State):

    reader_results = []

    for message in state["messages"]:
        if (
            isinstance(message, ToolMessage)
            and message.name == "scrape_url"
        ):
            reader_results.append(message.content)

    return {
        "reader": "\n\n---\n\n".join(reader_results)
    }

def writer_node (state : State) :
    topic = state["topic"]
    research = (
        f"SEARCH RESULTS:\n\n"
        f"{state['searcher']}\n\n"
        f"DETAILED READING:\n\n"
        f"{state['reader']}"
    )

    messages = prompts.WRITER_PROMPT.invoke({
        "topic": topic,
        "research": research
    })

    response = llm.invoke(messages)

    return {
        "writer": response.content
    }

# Routers
def searcher_router(state : State) :
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        if len(state["messages"]) >= 8:
            return "extract_search_result"
        return "search_tool"

    return "extract_search_result"

def reader_router(state : State) :
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        if len(state["messages"]) >= 10:
            return "extract_read_result"
        return "read_tool"

    return "extract_read_result"

def critic_node(state: State):
    report = state["writer"]

    messages = prompts.CRITIC_PROMPT.invoke({
        "report": report
    })

    response = llm.invoke(messages)

    return {"critic": response.content}

# Graph Creation
graph = StateGraph(State)

# Adding Nodes
graph.add_node("searcher",searcher_node)
graph.add_node("search_tool",searcher_tool_node)
graph.add_node("extract_search_result",extract_search_results_node)

graph.add_node("reader",reader_node)
graph.add_node("read_tool",reader_tool_node)
graph.add_node("extract_read_result",extract_reader_results_node)

graph.add_node("writer",writer_node)

graph.add_node("critic",critic_node)

# Adding Edges
graph.add_edge(START,"searcher")

graph.add_conditional_edges(
    "searcher" ,
     searcher_router , {
        "search_tool" : "search_tool",
        "extract_search_result" : "extract_search_result"
    }
)
graph.add_edge("search_tool","searcher")
graph.add_edge("extract_search_result","reader")

graph.add_conditional_edges(
    "reader" ,
     reader_router , {
        "read_tool" : "read_tool",
        "extract_read_result" : "extract_read_result"
    }
)
graph.add_edge("read_tool","reader")
graph.add_edge("extract_read_result","writer")

graph.add_edge("writer","critic")

graph.add_edge("critic",END)

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

print("\n" + "=" * 60)
print("SEARCHER RESULTS")
print("=" * 60)
print(result["searcher"])

print("\n" + "=" * 60)
print("READER RESULTS")
print("=" * 60)
print(result["reader"])

print("\n" + "=" * 60)
print("WRITER")
print("=" * 60)
print(result["writer"])