import os
import warnings
from typing import List, TypedDict

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

# Load environment variables
load_dotenv()

# Initialize model and memory
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
memory = SqliteSaver.from_conn_string(":memory:")
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


# Define Agent State
class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int


# Prompts
PLAN_PROMPT = (
    "You are an expert writer tasked with writing a high level outline of an essay. "
    "Write such an outline for the user provided topic. Give an outline of the essay "
    "along with any relevant notes or instructions for the sections."
)

WRITER_PROMPT = (
    "You are an essay assistant tasked with writing excellent 5-paragraph essays. "
    "Generate the best essay possible for the user's request and the initial outline. "
    "If the user provides critique, respond with a revised version of your previous attempts. "
    "Utilize all the information below as needed:\n\n------\n\n{content}"
)

REFLECTION_PROMPT = (
    "You are a teacher grading an essay submission. Generate critique and recommendations "
    "for the user's submission. Provide detailed recommendations, including requests for "
    "length, depth, style, etc."
)

RESEARCH_PLAN_PROMPT = (
    "You are a researcher charged with providing information that can be used when writing the "
    "following essay. Generate a list of search queries that will gather any relevant information. "
    "Only generate 3 queries max."
)

RESEARCH_CRITIQUE_PROMPT = (
    "You are a researcher charged with providing information that can be used when making any "
    "requested revisions (as outlined below). Generate a list of search queries that will gather "
    "any relevant information. Only generate 3 queries max."
)


# Structured Output Model
class Queries(BaseModel):
    queries: List[str]


# Node Functions
def plan_node(state: AgentState):
    messages = [
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=state["task"]),
    ]
    response = model.invoke(messages)
    return {"plan": response.content}


def research_plan_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state["task"]),
    ])
    content = state["content"] or []
    for query in queries.queries:
        response = tavily.search(query=query, max_results=2)
        for result in response["results"]:
            content.append(result["content"])
    return {"content": content}


def generation_node(state: AgentState):
    content = "\n\n".join(state["content"] or [])
    messages = [
        SystemMessage(content=WRITER_PROMPT.format(content=content)),
        HumanMessage(content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}"),
    ]
    response = model.invoke(messages)
    return {
        "draft": response.content,
        "revision_number": state.get("revision_number", 1) + 1,
    }


def reflection_node(state: AgentState):
    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state["draft"]),
    ]
    response = model.invoke(messages)
    return {"critique": response.content}


def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state["critique"]),
    ])
    content = state["content"] or []
    for query in queries.queries:
        response = tavily.search(query=query, max_results=2)
        for result in response["results"]:
            content.append(result["content"])
    return {"content": content}


def should_continue(state: AgentState):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"


# Build the Graph
builder = StateGraph(AgentState)
builder.set_entry_point("planner")

builder.add_node("planner", plan_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_critique", research_critique_node)

builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")
builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")

builder.add_conditional_edges("generate", should_continue, {
    END: END,
    "reflect": "reflect",
})

graph = builder.compile(checkpointer=memory)


# Run Example
if __name__ == "__main__":
    thread_config = {"configurable": {"thread_id": "1"}}
    initial_state = {
        "task": "what is the difference between langchain and langsmith",
        "max_revisions": 2,
        "revision_number": 1,
    }

    for state_update in graph.stream(initial_state, thread_config):
        print(state_update)

    # GUI launch (Optional if helper exists)
    warnings.filterwarnings("ignore")
    try:
        from helper import ewriter, writer_gui

        MultiAgent = ewriter()
        app = writer_gui(MultiAgent.graph)
        app.launch()
    except ImportError:
        print("GUI helper not found. Skipping GUI launch.")
