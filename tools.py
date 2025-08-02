from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from typing import List, Dict, Union
from vector_store import FlowerShopVectorStore
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize vector store
vector_store = FlowerShopVectorStore()

# Create LLM instance for summarization
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=os.environ['OPENAI_API_KEY']
)

@tool
def query_knowledge_base(query: str) -> List[Dict[str, str]]:
    """
    Looks up information in a knowledge base to help with answering customer questions and getting information on business processes.
    
    Args:
        query: The search query to look up in the knowledge base
        
    Returns:
        List of dictionaries containing relevant FAQ information
    """
    try:
        results = vector_store.query_faqs(query=query)
        # Convert ChromaDB results to proper format
        if 'metadatas' in results and results['metadatas']:
            return results['metadatas'][0]  # Return first result set
        return []
    except Exception as e:
        return [{"error": f"Failed to query knowledge base: {str(e)}"}]

@tool
def search_for_product_recommendations(description: str) -> List[Dict[str, str]]:
    """
    Looks up information in a knowledge base to help with product recommendations for customers.
    
    Args:
        description: Description of what the customer is looking for
        
    Returns:
        List of dictionaries containing relevant product information
    """
    try:
        results = vector_store.query_inventories(query=description)
        # Convert ChromaDB results to proper format
        if 'metadatas' in results and results['metadatas']:
            return results['metadatas'][0]  # Return first result set
        return []
    except Exception as e:
        return [{"error": f"Failed to search for products: {str(e)}"}]

@tool
def summarize_conversation(conversation: str) -> str:
    """
    Summarizes a conversation using the LLM and saves the summary to the vector store.
    Note: This tool expects a string input, not a list of messages.
    
    Args:
        conversation: String representation of the conversation to summarize
        
    Returns:
        String summary of the conversation
    """
    try:
        # Create the summarization prompt
        prompt = f"""Summarize the following flower shop customer service conversation in 2-3 sentences, focusing on:
- Any personal information shared by the customer (name, preferences, favorite colors, etc.)
- What the customer was looking for
- What recommendations or help was provided
- Any key outcomes or next steps

Make sure to include specific details like names, colors, or preferences mentioned.

Conversation:
{conversation}

Summary:"""
        
        # Get summary from LLM
        response = llm.invoke([HumanMessage(content=prompt)])
        summary = response.content.strip()
        
        # Save to vector store
        vector_store.save_conversation_summary(summary)
        return summary
        
    except Exception as e:
        error_msg = f"Failed to summarize conversation: {str(e)}"
        return error_msg

@tool
def query_past_summaries(query: str) -> str:
    """
    Searches past conversation summaries using semantic similarity.
    
    Args:
        query: The search query to find relevant past conversation summaries
        
    Returns:
        String containing relevant conversation summaries, formatted nicely
    """
    try:
        print(f"DEBUG: Searching summaries for query: '{query}'")
        summaries = vector_store.query_conversation_summaries(query)
        print(f"DEBUG: Found {len(summaries)} summaries")
        
        if not summaries:
            return "No relevant past conversations found."
        
        # Format the summaries nicely as a single string
        if len(summaries) == 1:
            return f"Here's what I found from our past conversations:\n\n{summaries[0]}"
        else:
            formatted = "Here's what I found from our past conversations:\n\n"
            for i, summary in enumerate(summaries[:3], 1):  # Limit to top 3 for readability
                formatted += f"{i}. {summary}\n\n"
            return formatted.strip()
        
    except Exception as e:
        print(f"DEBUG: Error querying summaries: {e}")
        return f"I had trouble accessing past conversation records: {str(e)}"

# Helper function for Streamlit (without @tool decorator to avoid validation issues)
def summarize_conversation_direct(conversation_messages) -> str:
    """
    Direct function to summarize conversation for Streamlit frontend.
    Converts message objects to string format then uses the main summarization logic.
    FILTERS OUT tool calls and tool responses to avoid feedback loops.
    """
    try:
        # Format conversation as string, filtering out tool-related content
        messages_text = []
        for msg in conversation_messages:
            if hasattr(msg, 'content'):
                # Skip messages that look like tool calls or tool responses
                content = msg.content
                
                # Skip if it's a tool call result (contains past conversation summaries)
                if ("Here's what I found from our past conversations" in content or 
                    "PAST CONVERSATION:" in content or
                    content.startswith('["') or  # Skip raw list responses
                    "No relevant past conversations found" in content):
                    continue
                
                # Skip if it's an empty or very short message
                if not content or len(content.strip()) < 10:
                    continue
                
                role = "Assistant" if isinstance(msg, AIMessage) else "User"
                messages_text.append(f"{role}: {content}")
        
        if len(messages_text) < 2:  # Need at least some back-and-forth
            return "Brief conversation - no summary needed."
        
        conversation_string = "\n".join(messages_text)
        
        # Use the same summarization logic
        prompt = f"""Summarize the following flower shop customer service conversation in 2-3 sentences, focusing on:
- Any personal information shared by the customer (name, preferences, favorite colors, etc.)
- What the customer was looking for
- What recommendations or help was provided
- Any key outcomes or next steps

IMPORTANT: Only summarize the actual conversation between the customer and assistant. Do not include any references to past conversations or previous summaries.

Conversation:
{conversation_string}

Summary:"""
        
        # Get summary from LLM
        response = llm.invoke([HumanMessage(content=prompt)])
        summary = response.content.strip()
        
        # Save to vector store
        vector_store.save_conversation_summary(summary)
        return summary
        
    except Exception as e:
        error_msg = f"Failed to summarize conversation: {str(e)}"
        return error_msg