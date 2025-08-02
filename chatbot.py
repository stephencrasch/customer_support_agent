from langgraph.graph import StateGraph, MessagesState
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from tools import query_knowledge_base, search_for_product_recommendations, query_past_summaries, summarize_conversation
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Production readiness check
if not os.environ.get('OPENAI_API_KEY'):
    raise ValueError(
        "OPENAI_API_KEY not found in environment variables. "
        "Please set your OpenAI API key in a .env file or environment variable."
    )

prompt = """#Purpose
You are a customer service chatbot for a flower shop company. You can help the customer achieve the goals listed below.
Always attempt to reference summaries from prior conversations to make the experience feel unique. 

#Goals
1. Answer questions the user might have relating to services offered
2. Recommend products to the user based on their preferences
3. Reference summaries from past conversations to make the experience feel unique

#Tone
Helpful and friendly."""

chat_template = ChatPromptTemplate.from_messages([
    ('system', prompt),
    ('placeholder', "{messages}")
])

tools = [query_knowledge_base, search_for_product_recommendations, query_past_summaries, summarize_conversation]

llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=os.environ['OPENAI_API_KEY']
)

llm_with_prompt = chat_template | llm.bind_tools(tools)

def call_agent(state: MessagesState):
    """Agent node that processes messages and returns response."""
    response = llm_with_prompt.invoke(state)
    return {
        'messages': [response]
    }

def should_continue(state: MessagesState):
    """Conditional edge function to determine next step."""
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return 'tool_node'
    else:
        return '__end__'

# Create the graph
graph = StateGraph(MessagesState)

# Create tool node
tool_node = ToolNode(tools)

# Add nodes
graph.add_node('agent', call_agent)
graph.add_node('tool_node', tool_node)

# Add edges
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        'tool_node': 'tool_node',
        '__end__': '__end__'
    }
)
graph.add_edge('tool_node', 'agent')

# Set entry point
graph.set_entry_point('agent')

# Compile the graph
app = graph.compile()
