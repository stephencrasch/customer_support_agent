import streamlit as st
from vector_store import FlowerShopVectorStore
from chatbot import app
from langchain_core.messages import AIMessage, HumanMessage
from tools import summarize_conversation_direct

st.set_page_config(layout='wide', page_title='Flower Shop Chatbot', page_icon='💐')

# Initialize vector store (this will create the summary collection)
@st.cache_resource
def init_vector_store():
    return FlowerShopVectorStore()

vector_store = init_vector_store()

# Initialize session state
if 'message_history' not in st.session_state:
    st.session_state.message_history = [AIMessage(content="Hiya, I'm the flower shop chatbot. How can I help? 🌸")]

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Create layout columns
left_col, main_col, right_col = st.columns([0.5, 2, 0.5])

# Left column - Clear chat functionality
with left_col:
    st.markdown("### 🗂️ Chat Management")
    
    if st.button('🆕 Clear Chat', use_container_width=True):
        if len(st.session_state.message_history) > 1:  # More than just the initial greeting
            try:
                with st.spinner("Saving conversation..."):
                    # Generate summary using the direct function
                    summary_result = summarize_conversation_direct(st.session_state.message_history)
                    
                    # Save current conversation to history
                    st.session_state.conversation_history.append({
                        'messages': st.session_state.message_history.copy(),
                        'summary': summary_result,
                        'timestamp': st.session_state.get('current_time', 'Unknown')
                    })
                    
                    # Reset message history
                    st.session_state.message_history = [AIMessage(content="Hiya, I'm the flower shop chatbot. How can I help? 🌸")]
                    
                    st.success("✅ Chat cleared and conversation saved!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error saving conversation: {str(e)}")
        else:
            st.info("💬 No conversation to clear yet.")


# Main column - Chat interface
with main_col:
    st.title("💐 Flower Shop Assistant")
    st.markdown("*Your friendly neighborhood florist bot! 🌺*")
    
    # Display chat messages in correct order (oldest first)
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.message_history:
            if isinstance(message, AIMessage):
                with st.chat_message('assistant', avatar="🤖"):
                    st.markdown(message.content)
            else:
                with st.chat_message('user', avatar="👤"):
                    st.markdown(message.content)
    
    # Chat input
    user_input = st.chat_input("Type your message here... 🌻")
    
    if user_input:
        # Store timestamp for this conversation
        import datetime
        st.session_state.current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add user message to history
        st.session_state.message_history.append(HumanMessage(content=user_input))
        
        # Display user message immediately
        with st.chat_message('user', avatar="👤"):
            st.markdown(user_input)
        
        try:
            # Get response from chatbot
            with st.chat_message('assistant', avatar="🤖"):
                with st.spinner("Thinking... 🌸"):
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
            st.error(f"❌ Error getting response: {str(e)}")

# Right column - Chat history and stats
with right_col:
    st.markdown("### 📚 Chat History")
    
    if st.session_state.conversation_history:
        # Show conversation count
        st.metric("Total Conversations", len(st.session_state.conversation_history))
        
        # Create options for conversation selection
        options = []
        for idx, conv in enumerate(st.session_state.conversation_history):
            summary = conv.get('summary', 'No summary')
            # Truncate summary for display
            display_summary = summary[:40] + "..." if len(summary) > 40 else summary
            timestamp = conv.get('timestamp', 'Unknown time')
            options.append(f"Chat {idx + 1}")
        
        selected = st.selectbox("Select a conversation:", options, key="conversation_selector")
        
        if selected:
            selected_idx = int(selected.split()[1]) - 1
            selected_conv = st.session_state.conversation_history[selected_idx]
            
            # Load conversation button
            if st.button("📥 Load Conversation", key="load_conversation", use_container_width=True):
                st.session_state.message_history = selected_conv['messages'].copy()
                st.success(f"✅ Loaded {selected}")
                st.rerun()
            
            # Show timestamp
            if 'timestamp' in selected_conv:
                st.caption(f"🕒 {selected_conv['timestamp']}")
            
            # Show full summary in expandable section
            with st.expander("📝 Full Summary", expanded=True):
                st.markdown(selected_conv.get('summary', 'No summary available'))
            
            # Preview of selected conversation
            with st.expander("👀 Message Preview", expanded=False):
                conversation = selected_conv['messages']
                
                # Show first few messages as preview
                preview_count = min(4, len(conversation))
                for i, msg in enumerate(conversation[:preview_count]):
                    role = "🤖 Bot" if isinstance(msg, AIMessage) else "👤 User"
                    # Truncate long messages for preview
                    content = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
                    st.markdown(f"**{role}:** {content}")
                
                if len(conversation) > preview_count:
                    st.markdown(f"*... and {len(conversation) - preview_count} more messages*")
            
            st.markdown("---")
    else:
        st.markdown("*No past conversations yet.*")
        st.markdown("💡 Start chatting to create conversation history!")
        




# Add some custom CSS for better styling
st.markdown("""
<style>
    .stChatMessage {
        margin-bottom: 1rem;
    }
    .stSelectbox {
        margin-bottom: 0.5rem;
    }
    .stButton > button {
        width: 100%;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)