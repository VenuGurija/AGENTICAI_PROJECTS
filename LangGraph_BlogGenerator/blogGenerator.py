# llm setup
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# build graph
from typing import TypedDict
class BlogState(TypedDict):
    topic: str
    title: str
    content: str

# creating nodes
from langchain.schema import SystemMessage,HumanMessage

def title_creator(state: BlogState) -> BlogState:
    response = llm.invoke([
        SystemMessage(content="You are a blog post title generator."),
        HumanMessage(content=f"Generate a catchy title for the following blog post content:\n{state['topic']}")
    ])
    state['title'] = response.content
    return state

def content_creator(state: BlogState) -> BlogState:
    response = llm.invoke([
        SystemMessage(content="You are a blog post content generator."),
        HumanMessage(content=f"Generate a detailed blog post on the following topic with the title '{state['title']}':\n{state['topic']}")
    ])
    state['content'] = response.content
    return state

def display_node(state: BlogState) -> BlogState:
    # Optionally print or log here
    print(f"\n📝 Title: {state['title']}\n\n{state['content']}\n")
    return state  # Return full state, not just a string

# build graph

from langgraph.graph import StateGraph, START, END

builder = StateGraph(BlogState)

builder.add_node("title_creator", title_creator)
builder.add_node("content_creator", content_creator)
builder.add_node("display_node", display_node)

builder.add_edge(START, "title_creator")
builder.add_edge("title_creator", "content_creator")
builder.add_edge("content_creator", "display_node") 
builder.add_edge("display_node", END)

blogCreatorGraph = builder.compile()

# visualize graph
#from IPython.display import Image, display
#display(Image(blogCreatorGraph.get_graph().draw_mermaid_png()))

result = blogCreatorGraph.invoke({"topic": "what if mjolnir comes to ironman after snapping infinity stones"})