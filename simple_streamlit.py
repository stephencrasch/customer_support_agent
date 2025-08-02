"""
Simple Streamlit Customer Support Chat Interface
This is a beginner-friendly version of the flower shop chatbot.
"""
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

# Only import what we need - with error handling for missing dependencies
try:
    from vector_store import FlowerShopVectorStore
    from chatbot import app
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Configure the page
st.set_page_config(
    page_title='Flower Shop Chat',
    page_icon='🌸',
    layout='centered'  # Simple centered layout instead of wide
)

# Simple title and description
st.title("🌸 Flower Shop Assistant")
st.markdown("Welcome! I'm here to help with your flower needs. Ask me anything!")

# Initialize the vector store if dependencies are available
if DEPENDENCIES_AVAILABLE:
    @st.cache_resource
    def init_vector_store():
        """Initialize vector store once and cache it"""
        return FlowerShopVectorStore()
    
    try:
        vector_store = init_vector_store()
    except Exception as e:
        st.error(f"Error initializing vector store: {e}")
        st.stop()
else:
    st.error(f"Missing dependencies: {IMPORT_ERROR}")
    st.markdown("""
    **Setup Instructions:**
    1. Install required packages: `pip install -r requirements.txt`
    2. Set up your OpenAI API key in a `.env` file
    3. Restart the application
    """)
    st.stop()

# Simple session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hello! I'm your flower shop assistant. How can I help you today? 🌺")
    ]

# Clear chat button (simple and obvious)
if st.button("🆕 Start New Chat", type="secondary"):
    st.session_state.messages = [
        AIMessage(content="Hello! I'm your flower shop assistant. How can I help you today? 🌺")
    ]
    st.rerun()

# Display chat messages
st.markdown("### Chat")
for message in st.session_state.messages:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.write(message.content)
    else:
        with st.chat_message("user", avatar="👤"):
            st.write(message.content)

# Chat input
user_input = st.chat_input("Type your message here...")

# Handle user input
if user_input:
    # Add user message to chat
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    # Display user message immediately
    with st.chat_message("user", avatar="👤"):
        st.write(user_input)
    
    # Get bot response
    try:
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                # Call the chatbot
                response = app.invoke({'messages': st.session_state.messages})
                
                # Get the latest AI message from response
                latest_response = None
                for msg in reversed(response['messages']):
                    if isinstance(msg, AIMessage):
                        latest_response = msg
                        break
                
                if latest_response:
                    st.write(latest_response.content)
                    # Update session state with the full conversation
                    st.session_state.messages = response['messages']
                else:
                    st.error("Sorry, I couldn't generate a response. Please try again.")
    
    except Exception as e:
        with st.chat_message("assistant", avatar="🤖"):
            st.error(f"Sorry, I encountered an error: {str(e)}")
            st.markdown("Please check your OpenAI API key and try again.")
    
    # Refresh the page to show the new message
    st.rerun()

# Simple footer with basic info
st.markdown("---")
st.markdown("""
**💡 Tips for new users:**
- Ask about our flowers, arrangements, or services
- Try: "What flowers do you have?" or "Do you do wedding arrangements?"
- Use the "Start New Chat" button to begin fresh conversations
""")

# Optional: Show system info for debugging
with st.expander("🔧 System Info (for debugging)"):
    st.write(f"Total messages in chat: {len(st.session_state.messages)}")
    st.write(f"Dependencies loaded: {DEPENDENCIES_AVAILABLE}")
    if DEPENDENCIES_AVAILABLE:
        try:
            faq_count = vector_store.faq_collection.count() if vector_store else 0
            inventory_count = vector_store.inventory_collection.count() if vector_store else 0
            st.write(f"FAQ entries: {faq_count}")
            st.write(f"Inventory entries: {inventory_count}")
        except:
            st.write("Vector store counts unavailable")