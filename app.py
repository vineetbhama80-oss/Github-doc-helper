import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="GitHub Doc Helper",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #8b5cf6;
        --secondary-color: #7c3aed;
        --success-color: #10b981;
        --error-color: #ef4444;
        --background-dark: #1e1b4b;
        --text-light: #e0e7ff;
    }
    
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    }
    
    /* Custom header */
    .custom-header {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(139, 92, 246, 0.3);
    }
    
    .custom-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    
    .custom-header p {
        color: #e0e7ff;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        color: white;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%);
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.6);
        transform: translateY(-2px);
    }
    
    /* File uploader styling */
    .uploadedFile {
        background-color: rgba(139, 92, 246, 0.1);
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #8b5cf6;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: rgba(245, 158, 11, 0.1);
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
    }
    
    /* Card styling */
    .custom-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(139, 92, 246, 0.3);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Stats counter */
    .stat-counter {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        display: block;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(139, 92, 246, 0.1);
        border-radius: 8px;
    }
    
    /* Input field styling */
    .stTextInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.05);
        color: white;
        border-radius: 8px;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #9ca3af;
        font-size: 0.9rem;
        border-top: 1px solid rgba(139, 92, 246, 0.2);
        margin-top: 3rem;
    }
    
    /* Animated loader */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading-text {
        animation: pulse 1.5s ease-in-out infinite;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'total_chunks' not in st.session_state:
    st.session_state.total_chunks = 0
if 'total_questions' not in st.session_state:
    st.session_state.total_questions = 0

# Custom header
st.markdown("""
    <div class="custom-header">
        <h1>📚 GitHub Documentation Helper</h1>
        <p>Upload documentation files and get instant answers using AI-powered search</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # API Key input with improved styling
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    api_key_input = st.text_input(
        "🔑 Groq API Key", 
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Get your free API key from console.groq.com"
    )
    
    if api_key_input:
        st.session_state.api_key = api_key_input
        st.success("✅ API Key configured")
    else:
        st.warning("⚠️ API Key required")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # File upload section
    st.markdown("### 📁 Upload Documents")
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Select files",
        type=["md", "txt"],
        accept_multiple_files=True,
        help="Upload README.md or any documentation files",
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        st.markdown(f'<div class="success-box">✅ {len(uploaded_files)} file(s) selected</div>', unsafe_allow_html=True)
        with st.expander("📄 View files", expanded=False):
            for idx, file in enumerate(uploaded_files, 1):
                st.markdown(f"**{idx}.** {file.name} ({file.size} bytes)")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process button
    st.markdown("---")
    process_button = st.button("🚀 Process & Index Documents", use_container_width=True)
    
    # Stats section
    if st.session_state.vectorstore:
        st.markdown("---")
        st.markdown("### 📊 Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
                <div class="stat-counter">
                    <span class="stat-number">{len(st.session_state.processed_files)}</span>
                    <span class="stat-label">Files</span>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="stat-counter">
                    <span class="stat-number">{st.session_state.total_chunks}</span>
                    <span class="stat-label">Chunks</span>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="stat-counter">
                <span class="stat-number">{st.session_state.total_questions}</span>
                <span class="stat-label">Questions Asked</span>
            </div>
        """, unsafe_allow_html=True)
    
    # Indexed files
    if st.session_state.processed_files:
        st.markdown("---")
        st.markdown("### ✅ Indexed Files")
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        for idx, filename in enumerate(st.session_state.processed_files, 1):
            st.markdown(f"**{idx}.** {filename}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.processed_files = []
            st.session_state.chat_history = []
            st.session_state.total_chunks = 0
            st.session_state.total_questions = 0
            st.rerun()
    
    # Help section
    st.markdown("---")
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. **Get API Key**: Sign up at [console.groq.com](https://console.groq.com)
        2. **Upload Files**: Select your README or docs
        3. **Process**: Click the process button
        4. **Ask Questions**: Type your questions in the chat
        
        **Example questions:**
        - How do I install this?
        - What are the main features?
        - Show me usage examples
        - What configuration is needed?
        """)
    
    with st.expander("🔧 Tech Stack"):
        st.markdown("""
        - **LangChain**: RAG framework
        - **FAISS**: Vector search
        - **Groq API**: LLM inference
        - **Sentence Transformers**: Embeddings
        - **Streamlit**: Web interface
        """)

# Main content area
if process_button:
    if not st.session_state.api_key:
        st.error("❌ Please enter your Groq API key in the sidebar")
    elif not uploaded_files:
        st.error("❌ Please upload at least one file")
    else:
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Read files
            status_text.markdown('<p class="loading-text">📖 Reading files...</p>', unsafe_allow_html=True)
            progress_bar.progress(20)
            time.sleep(0.3)
            
            documents = []
            for uploaded_file in uploaded_files:
                content = uploaded_file.read().decode("utf-8")
                doc = Document(
                    page_content=content,
                    metadata={"source": uploaded_file.name}
                )
                documents.append(doc)
            
            # Step 2: Split into chunks
            status_text.markdown('<p class="loading-text">✂️ Splitting into chunks...</p>', unsafe_allow_html=True)
            progress_bar.progress(40)
            time.sleep(0.3)
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)
            st.session_state.total_chunks = len(chunks)
            
            # Step 3: Create embeddings
            status_text.markdown('<p class="loading-text">🧠 Creating embeddings (first time may take 1-2 minutes)...</p>', unsafe_allow_html=True)
            progress_bar.progress(60)
            
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Step 4: Create vector store
            status_text.markdown('<p class="loading-text">🔍 Building search index...</p>', unsafe_allow_html=True)
            progress_bar.progress(80)
            time.sleep(0.3)
            
            vectorstore = FAISS.from_documents(chunks, embeddings)
            
            # Step 5: Save to session
            progress_bar.progress(100)
            st.session_state.vectorstore = vectorstore
            st.session_state.processed_files = [f.name for f in uploaded_files]
            
            status_text.empty()
            progress_bar.empty()
            
            # Success message
            st.markdown(f"""
                <div class="success-box">
                    <h3>✅ Processing Complete!</h3>
                    <p>Successfully processed <strong>{len(chunks)} chunks</strong> from <strong>{len(uploaded_files)} file(s)</strong></p>
                    <p>You can now ask questions about your documentation!</p>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error: {str(e)}")

# Chat interface
st.markdown("## 💬 Ask Questions")

# Display welcome message or chat history
if not st.session_state.chat_history and not st.session_state.vectorstore:
    st.markdown("""
        <div class="info-box">
            <h3>👋 Welcome!</h3>
            <p>To get started:</p>
            <ol>
                <li>Enter your Groq API key in the sidebar</li>
                <li>Upload your documentation files</li>
                <li>Click "Process & Index Documents"</li>
                <li>Start asking questions!</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)
    
    # Example questions
    st.markdown("""
        <div class="custom-card">
            <h4>💡 Example Questions You Can Ask:</h4>
            <ul>
                <li>"How do I install this project?"</li>
                <li>"What are the main features?"</li>
                <li>"Show me usage examples"</li>
                <li>"What dependencies are required?"</li>
                <li>"How do I configure the settings?"</li>
                <li>"Are there any API endpoints?"</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

elif not st.session_state.chat_history and st.session_state.vectorstore:
    st.markdown("""
        <div class="success-box">
            <h3>✅ Ready to Answer!</h3>
            <p>Your documents are indexed. Type your question below to get started.</p>
        </div>
    """, unsafe_allow_html=True)

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            st.caption(f"📄 Sources: {message['sources']}")

# Chat input
user_question = st.chat_input(
    "Type your question here...",
    disabled=not st.session_state.vectorstore
)

if user_question:
    # Increment question counter
    st.session_state.total_questions += 1
    
    # Add user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_question
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_question)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documentation..."):
            try:
                # Initialize LLM
                llm = ChatGroq(
                    groq_api_key=st.session_state.api_key,
                    model_name="mixtral-8x7b-32768",
                    temperature=0.3
                )
                
                # Create prompt template
                prompt = ChatPromptTemplate.from_template("""
                Answer the question based only on the following context:
                
                {context}
                
                Question: {input}
                
                Answer the question in detail based on the context provided. If you cannot find the answer in the context, say so.
                """)
                
                # Create chains
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                # Get answer
                result = retrieval_chain.invoke({"input": user_question})
                answer = result["answer"]
                source_docs = result["context"]
                
                # Get sources
                sources = list(set([
                    doc.metadata.get("source", "Unknown") 
                    for doc in source_docs
                ]))
                sources_text = ", ".join(sources)
                
                # Display answer
                st.markdown(answer)
                st.caption(f"📄 Sources: {sources_text}")
                
                # Add to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources_text
                })
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"❌ {error_msg}"
                })
    
    st.rerun()

# Footer
st.markdown("""
    <div class="footer">
        <p><strong>GitHub Documentation Helper</strong></p>
        <p>Built with LangChain • FAISS • Groq API • Sentence Transformers • Streamlit</p>
        <p style="font-size: 0.8rem; margin-top: 1rem;">
            💡 Tip: For best results, ask specific questions about installation, features, or usage
        </p>
    </div>
""", unsafe_allow_html=True)