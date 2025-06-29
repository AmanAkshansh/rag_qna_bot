import streamlit as st
import google.generativeai as genai
import PyPDF2
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tempfile
import os
import time

# Configure Google AI (PUT YOUR API KEY HERE)
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

# Page config
st.set_page_config(
    page_title="🧠 RAG QnA Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 0.5rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    
    .bot-message {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        margin-right: 20%;
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #667eea;
        text-align: center;
        margin: 1rem 0;
    }
    
    .sidebar-metric {
        background: #667eea;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🧠 RAG QnA Assistant</h1>
    <p>Advanced Document Intelligence & Question Answering System</p>
    <p><em>Powered by Google Gemini AI & Vector Search Technology</em></p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processing_stats' not in st.session_state:
    st.session_state.processing_stats = {}
if 'clear_input' not in st.session_state:
    st.session_state.clear_input = False

def extract_text_from_pdf(file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def chunk_text(text, chunk_size=1000):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Sidebar with system info
with st.sidebar:
    st.markdown("## 🚀 System Status")
    
    if st.session_state.processed:
        stats = st.session_state.processing_stats
        st.markdown(f"""
        <div class="sidebar-metric">
            📊 Documents: {stats.get('files', 0)}
        </div>
        <div class="sidebar-metric">
            🧩 Text Chunks: {stats.get('chunks', 0)}
        </div>
        <div class="sidebar-metric">
            🔍 Vector Embeddings: {stats.get('embeddings', 0)}
        </div>
        <div class="sidebar-metric">
            💬 Queries Processed: {len(st.session_state.chat_history)}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("## ⚡ AI Model")
        st.info("🤖 **Google Gemini 1.5 Flash**\n🧠 Advanced Language Model\n🔍 **Sentence Transformers**\nall-MiniLM-L6-v2")
        
        st.markdown("---")
        st.markdown("## 🛠️ Tech Stack")
        st.markdown("""
        - 🎯 **RAG Architecture**
        - 🔢 **FAISS Vector DB**
        - 🚀 **Streamlit Frontend**
        - 🧬 **Neural Embeddings**
        """)
    else:
        st.markdown("""
        <div class="sidebar-metric">
            ⏳ Ready to Process
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("## 📋 Supported Formats")
        st.markdown("""
        - 📄 **PDF Documents**
        - 📝 **Word Documents (DOCX)**
        - 🔄 **Multi-file Processing**
        - 🎯 **Intelligent Chunking**
        """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## 📁 Document Upload Center")
    
    st.markdown("""
    <div class="upload-section">
        <h3>🎯 Upload Your Documents</h3>
        <p>Drag & drop or browse to select multiple PDF/DOCX files</p>
        <p><em>AI will process and create intelligent embeddings</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Select documents for processing",
        type=['pdf', 'docx'],
        accept_multiple_files=True,
        help="Upload multiple PDF or DOCX files. The system will process them intelligently."
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) ready for processing")
        
        # Show file details
        with st.expander("📋 File Details", expanded=True):
            for i, file in enumerate(uploaded_files, 1):
                file_size = len(file.getvalue()) / 1024  # KB
                st.markdown(f"**{i}.** {file.name} ({file_size:.1f} KB)")
    
    if uploaded_files and st.button("🚀 Process Documents", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("🧠 AI Processing in Progress..."):
            all_text = ""
            
            # Processing files
            for i, file in enumerate(uploaded_files):
                status_text.text(f"📖 Reading {file.name}...")
                progress_bar.progress((i + 1) / (len(uploaded_files) + 3))
                
                if file.type == "application/pdf":
                    text = extract_text_from_pdf(file)
                elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text = extract_text_from_docx(file)
                all_text += text + "\n"
                time.sleep(0.1)  # Small delay for effect
            
            # Chunking
            status_text.text("🧩 Creating intelligent text chunks...")
            progress_bar.progress(0.7)
            chunks = chunk_text(all_text)
            st.session_state.documents = chunks
            
            # Creating embeddings
            status_text.text("🧬 Generating neural embeddings...")
            progress_bar.progress(0.8)
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(chunks)
            
            # Creating vector index
            status_text.text("🔍 Building vector search index...")
            progress_bar.progress(0.9)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            faiss.normalize_L2(embeddings)
            index.add(embeddings.astype('float32'))
            
            st.session_state.embeddings = (model, index)
            st.session_state.processed = True
            
            # Store stats
            st.session_state.processing_stats = {
                'files': len(uploaded_files),
                'chunks': len(chunks),
                'embeddings': embeddings.shape[0],
                'total_words': len(all_text.split())
            }
            
            progress_bar.progress(1.0)
            status_text.text("✅ Processing Complete!")
            
        # Removed st.balloons() - no more balloon animation
        st.success(f"🎉 Successfully processed {len(uploaded_files)} files into {len(chunks)} intelligent chunks!")

with col2:
    st.markdown("## 💬 AI Chat Interface")
    
    if st.session_state.processed:
        # Chat container
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for chat in st.session_state.chat_history:
                if chat['type'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 You:</strong><br>
                        {chat['message']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>🤖 AI Assistant:</strong><br>
                        {chat['message']}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Question input - will be cleared after AI response
        question_value = "" if st.session_state.clear_input else None
        question = st.text_input(
            "Ask anything about your documents:",
            placeholder="e.g., What are the main findings? Summarize the key points...",
            help="Ask detailed questions about the content in your uploaded documents",
            value=question_value,
            key=f"question_input_{len(st.session_state.chat_history)}"
        )
        
        # Reset the clear_input flag after using it
        if st.session_state.clear_input:
            st.session_state.clear_input = False
        
        col_ask, col_clear = st.columns([3, 1])
        with col_ask:
            ask_button = st.button("🔍 Ask AI", type="primary", use_container_width=True)
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        if question and ask_button:
            # Add user message to history
            st.session_state.chat_history.append({'type': 'user', 'message': question})
            
            with st.spinner("🧠 AI is analyzing your documents..."):
                model, index = st.session_state.embeddings
                
                # Get query embedding
                query_embedding = model.encode([question])
                faiss.normalize_L2(query_embedding)
                
                # Search for relevant chunks
                scores, indices = index.search(query_embedding.astype('float32'), 3)
                
                # Get relevant text
                relevant_chunks = [st.session_state.documents[i] for i in indices[0]]
                context = "\n\n".join(relevant_chunks)
                
                # Generate answer using Gemini
                prompt = f"""
                You are an expert AI assistant analyzing documents. Provide a comprehensive, well-structured answer based on the context provided.
                
                Context from documents:
                {context}
                
                Question: {question}
                
                Instructions:
                - Provide a detailed, informative answer
                - Use bullet points or numbered lists when appropriate
                - If the information isn't in the context, mention that clearly
                - Be professional and thorough
                
                Answer:
                """
                
                model_gen = genai.GenerativeModel('gemini-1.5-flash')
                response = model_gen.generate_content(prompt)
                
                # Add AI response to history
                st.session_state.chat_history.append({'type': 'ai', 'message': response.text})
                
                # Set flag to clear input on next render
                st.session_state.clear_input = True
                
            st.rerun()
            
    else:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Ready for Intelligent Document Analysis</h3>
            <p>Upload your documents on the left to start the AI-powered Q&A experience!</p>
            <br>
            <h4>🚀 Advanced Features:</h4>
            <ul>
                <li>🧠 <strong>Neural Document Understanding</strong></li>
                <li>🔍 <strong>Semantic Vector Search</strong></li>
                <li>💬 <strong>Conversational AI Interface</strong></li>
                <li>📊 <strong>Multi-Document Analysis</strong></li>
                <li>⚡ <strong>Real-time Processing</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🤖 AI Model:** Google Gemini 1.5 Flash")
with col2:
    st.markdown("**🔍 Search:** FAISS Vector Database")
with col3:
    st.markdown("**🧬 Embeddings:** Sentence Transformers")