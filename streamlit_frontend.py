import streamlit as st
from vector_store import FlowerShopVectorStore
from chatbot import app
from langchain_core.messages import AIMessage, HumanMessage

st.set_page_config(layout='wide', page_title='Flower Shop Chatbot', page_icon='💐')

# Initialize session state
if 'message_history' not in st.session_state:
    st.session_state.message_history = [AIMessage(content="Hiya, I'm the flower shop chatbot. How can I help?")]

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Create layout columns
left_col, main_col, right_col = st.columns([0.5, 2, 0.5])

# Left column - Clear chat functionality
with left_col:
    if st.button('Clear Chat'):
        if len(st.session_state.message_history) > 1:  # More than just the initial greeting
            try:
                # Generate summary using the tool
                from tools import summarize_conversation
                summary = summarize_conversation(st.session_state.message_history)
                
                # Save current conversation to history
                st.session_state.conversation_history.append(st.session_state.message_history.copy())
                
                # Reset message history
                st.session_state.message_history = [AIMessage(content="Hiya, I'm the flower shop chatbot. How can I help?")]
                
                st.success("Chat cleared and conversation saved!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error saving conversation: {str(e)}")
        else:
            st.info("No conversation to clear yet.")

# Main column - Chat interface
with main_col:
    st.title("💐 Flower Shop Assistant")
    
    # Display chat messages in correct order (oldest first)
    for message in st.session_state.message_history:
        if isinstance(message, AIMessage):
            with st.chat_message('assistant'):
                st.markdown(message.content)
        else:
            with st.chat_message('user'):
                st.markdown(message.content)
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to history
        st.session_state.message_history.append(HumanMessage(content=user_input))
        
        # Display user message immediately
        with st.chat_message('user'):
            st.markdown(user_input)
        
        try:
            # Get response from chatbot
            with st.chat_message('assistant'):
                with st.spinner("Thinking..."):
                    response = app.invoke({'messages': st.session_state.message_history})
                    
                    # Update message history with all messages from response
                    st.session_state.message_history = response['messages']
                    
                    # Display the latest AI response
                    latest_ai_message = None
                    for msg in reversed(response['messages']):
                        if isinstance(msg, AIMessage):
                            latest_ai_message = msg
                            break
                    
                    if latest_ai_message:
                        st.markdown(latest_ai_message.content)
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Error getting response: {str(e)}")

# Right column - Chat history
with right_col:
    st.markdown("#### 📚 Chat History")
    
    if st.session_state.conversation_history:
        # Create options for conversation selection
        options = [f"Conversation {idx + 1}" for idx in range(len(st.session_state.conversation_history))]
        
        selected = st.selectbox("Select a conversation:", options, key="conversation_selector")
        
        if selected:
            selected_idx = options.index(selected)
            
            # Load conversation button
            if st.button("Load Selected Conversation", key="load_conversation"):
                st.session_state.message_history = st.session_state.conversation_history[selected_idx].copy()
                st.success(f"Loaded {selected}")
                st.rerun()
            
            # Preview of selected conversation
            st.markdown(f"**Preview of {selected}:**")
            conversation = st.session_state.conversation_history[selected_idx]
            
            # Show first few messages as preview
            preview_count = min(4, len(conversation))
            for i, msg in enumerate(conversation[:preview_count]):
                role = "🤖 Assistant" if isinstance(msg, AIMessage) else "👤 User"
                # Truncate long messages for preview
                content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
                st.markdown(f"**{role}:** {content}")
            
            if len(conversation) > preview_count:
                st.markdown(f"... and {len(conversation) - preview_count} more messages")
            
            st.markdown("---")
    else:
        st.markdown("_No past conversations yet._")
        st.markdown("Start chatting to create conversation history!")

# Add some styling
st.markdown("""
<style>
    .stChatMessage {
        margin-bottom: 1rem;
    }
    .stSelectbox {
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)