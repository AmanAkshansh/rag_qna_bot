# 🧠 RAG QnA Assistant

An intelligent document Q&A system powered by Google Gemini AI and vector search technology.

## Features

- **Multi-format Support**: Upload PDF and DOCX documents
- **Smart Processing**: Intelligent text chunking and neural embeddings
- **AI-Powered Q&A**: Ask questions about your documents in natural language
- **Vector Search**: FAISS-powered semantic search for accurate answers
- **Interactive UI**: Clean, modern interface with real-time chat

## How to Use

1. Upload your PDF or DOCX documents
2. Wait for AI processing to complete
3. Ask questions about your documents
4. Get intelligent, context-aware answers

## Tech Stack

- **Frontend**: Streamlit
- **AI Model**: Google Gemini 1.5 Flash
- **Embeddings**: Sentence Transformers
- **Vector DB**: FAISS
- **Document Processing**: PyPDF2, python-docx

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set your Google AI API key
4. Run: `streamlit run app.py`
