from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from db import save_summary_to_db, search_summaries_in_db
from typing import List, Dict, Union
from vector_store import FlowerShopVectorStore

vector_store = FlowerShopVectorStore()

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
        return vector_store.query_faqs(query=query)
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
        return vector_store.query_inventories(query=description)
    except Exception as e:
        return [{"error": f"Failed to search for products: {str(e)}"}]

@tool
def summarize_conversation(conversation: Union[List[Dict], List]) -> str:
    """
    Summarizes a conversation using the LLM and saves the summary to the database.
    
    Args:
        conversation: List of message objects or dictionaries representing the conversation
        
    Returns:
        String summary of the conversation
    """
    try:
        # Handle different message formats
        messages_text = []
        for msg in conversation:
            if hasattr(msg, 'content'):
                # LangChain message object
                content = msg.content
            elif isinstance(msg, dict) and 'content' in msg:
                # Dictionary with content key
                content = msg['content']
            elif isinstance(msg, str):
                # Plain string
                content = msg
            else:
                # Skip invalid message formats
                continue
            
            if content and content.strip():
                messages_text.append(content)
        
        if not messages_text:
            return "No valid messages found to summarize."
        
        # Create the summarization prompt
        conversation_text = "\n".join(messages_text)
        prompt = f"Summarize the following conversation in 1-2 sentences:\n{conversation_text}"
        
        # Get summary from LLM
        response = llm.invoke([HumanMessage(content=prompt)])
        summary = response.content
        
        # Save to database
        save_summary_to_db(summary)
        return summary
        
    except Exception as e:
        error_msg = f"Failed to summarize conversation: {str(e)}"
        return error_msg

@tool
def query_past_summaries(query: str) -> List[str]:
    """
    Searches past conversation summaries for relevance to the current query.
    
    Args:
        query: The search query to find relevant past conversation summaries
        
    Returns:
        List of relevant conversation summaries
    """
    try:
        summaries = search_summaries_in_db(query)
        if not summaries:
            return ["No relevant past conversations found."]
        return summaries
    except Exception as e:
        return [f"Failed to query past summaries: {str(e)}"]