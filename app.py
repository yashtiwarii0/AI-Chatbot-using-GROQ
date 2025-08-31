import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import time
from datetime import datetime

# Load environment variables
load_dotenv()

# LangSmith tracking setup
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with GROQ"

# Configure page
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        height: 400px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 5px 18px;
        margin: 0.5rem 0;
        margin-left: 20%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .assistant-message {
        background: white;
        color: #333;
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 18px 5px;
        margin: 0.5rem 0;
        margin-right: 20%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .timestamp {
        font-size: 0.7rem;
        opacity: 0.6;
        margin-top: 0.25rem;
    }
    
    .sidebar-section {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .model-info {
        background: #e8f4fd;
        padding: 0.75rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    
    .stTextInput > div > div > input {
        border-radius: 20px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem 1rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Provide clear, informative, and engaging responses to user queries. Be conversational yet professional."),
    ("user", "Question: {question}")
])

def generate_response(question, model, temperature, max_tokens):
    """Generate response using GROQ API"""
    try:
        llm = ChatGroq(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        answer = chain.invoke({'question': question})
        return answer
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'message_count' not in st.session_state:
    st.session_state.message_count = 0

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 AI Assistant</h1>
    <p>Powered by GROQ • Fast & Intelligent Responses</p>
</div>
""", unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([3, 1])

with col2:
    # Sidebar settings
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Model Settings")
    
    model = st.selectbox(
        "Select Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b"],
        help="Choose the AI model for generating responses"
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Controls randomness: 0=focused, 1=creative"
    )
    
    max_tokens = st.slider(
        "Max Tokens",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="Maximum length of the response"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model information
    st.markdown(f"""
    <div class="model-info">
        <strong>Current Model:</strong><br>
        {model}<br><br>
        <strong>Settings:</strong><br>
        Temperature: {temperature}<br>
        Max Tokens: {max_tokens}
    </div>
    """, unsafe_allow_html=True)
    
    # Chat statistics
    st.markdown("### 📊 Chat Stats")
    st.metric("Messages", st.session_state.message_count)
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.chat_history = []
        st.session_state.message_count = 0
        st.rerun()

with col1:
    # Chat interface
    st.markdown("### 💬 Chat with AI Assistant")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        if st.session_state.chat_history:
            for i, (user_msg, assistant_msg, timestamp) in enumerate(st.session_state.chat_history):
                # User message
                st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong> {user_msg}
                    <div class="timestamp">{timestamp}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Assistant message
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>🤖 Assistant:</strong> {assistant_msg}
                    <div class="timestamp">{timestamp}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; color: #666;">
                <h3>👋 Welcome to your AI Assistant!</h3>
                <p>Start a conversation by typing your question below.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Input section
    st.markdown("---")
    
    # Create input form
    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_button = st.columns([4, 1])
        
        with col_input:
            user_input = st.text_input(
                "Your message:",
                placeholder="Type your question here...",
                label_visibility="collapsed"
            )
        
        with col_button:
            submit_button = st.form_submit_button("Send 📤", type="primary")
    
    # Process user input
    if submit_button and user_input.strip():
        # Show loading spinner
        with st.spinner("🤔 Thinking..."):
            # Generate response
            response = generate_response(user_input, model, temperature, max_tokens)
            
            # Add to chat history
            timestamp = datetime.now().strftime("%H:%M")
            st.session_state.chat_history.append((user_input, response, timestamp))
            st.session_state.message_count += 1
            
            # Rerun to update chat display
            st.rerun()
    
    elif submit_button and not user_input.strip():
        st.warning("⚠️ Please enter a question to get started!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Built with ❤️ using Streamlit & GROQ API | 
    <a href="#" style="color: #667eea;">Documentation</a> | 
    <a href="#" style="color: #667eea;">Support</a></p>
</div>
""", unsafe_allow_html=True)

# Sidebar footer
with col2:
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; font-size: 0.8rem; color: #666;">
        <p>💡 <strong>Tips:</strong></p>
        <p>• Use lower temperature for factual questions</p>
        <p>• Use higher temperature for creative tasks</p>
        <p>• Increase max tokens for longer responses</p>
    </div>
    """, unsafe_allow_html=True)