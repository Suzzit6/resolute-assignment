
# RAG-Powered Multi-Agent Q&A Assistant

## Overview

This project implements an intelligent document assistant that combines Retrieval-Augmented Generation (RAG) with a multi-agent system to answer questions about document content, calculate mathematical expressions, and provide definitions. The system orchestrates different tools based on query intent, creating a versatile assistant for information retrieval and processing.

## Architecture

The application features a three-layer architecture:

1. **Data Layer**
   - Document ingestion and processing
   - Vector embedding with FAISS
   - Document chunking for optimal retrieval

2. **Agent Layer**
   - Router agent for query classification
   - RAG agent for document-based questions
   - Calculator agent for mathematical operations
   - Dictionary agent for term definitions

3. **Presentation Layer**
   - Streamlit web interface
   - Interactive query system
   - Source attribution display
   - System logging

## Key Design Choices

### Vector Store: FAISS
I chose FAISS (Facebook AI Similarity Search) for the vector database because:
- It provides efficient similarity search for dense vectors
- Performs well with medium-sized document collections
- Supports local deployment without external dependencies
- Fast retrieval speed for real-time applications

### Embeddings: Hugging Face (all-MiniLM-L6-v2)
The system uses the all-MiniLM-L6-v2 model from Sentence Transformers because:
- It produces high-quality embeddings with relatively small dimensions (384)
- Balances performance and resource usage well
- Can run efficiently on CPU for accessibility

### LLM: Google Gemini
I integrated Google's Gemini model via LangChain's ChatGoogleGenerativeAI wrapper because:
- It offers strong reasoning capabilities at lower cost than GPT models
- Handles multi-modal content well (potential future extension)
- Provides good performance for RAG applications

### Agent Routing System
The agent routing system was designed to be:
- Simple but effective (keyword-based classification)
- Extensible (easy to add new agents)
- Transparent (logs decision paths)

## Getting Started

### Requirements

Install all required packages:

```bash
pip install -r requirements.txt
```
Put Gemini Api key 
```bash
api_key = "your api key "
```

The main dependencies include:
- streamlit
- langchain
- langchain-google-genai
- faiss-cpu
- sentence-transformers
- PyDictionary
- sympy

### Document Preparation

1. Create a company folder in the project directory
2. Add 3-5 PDF or TXT documents containing the information you want to query
   - Example documents might include: company policies, product specifications, FAQs, etc.

### Running the Application

Execute the following command in your terminal:

```bash
streamlit run main.py
```

The web interface will open in your default browser at http://localhost:8501

### Using the Application

1. **Process Documents**
   - Go to the "Document Processing" tab
   - Choose between using default company documents or uploading custom files
   - Click "Process Documents" button and wait for processing to complete
   
2. **Ask Questions**
   - Navigate to "Ask Questions" tab
   - Type your query in the input field:
     - Standard questions about document content (e.g., "What is the company's vacation policy?")
     - Calculation queries (e.g., "Calculate 145 + 287")
     - Definition requests (e.g., "Define artificial intelligence")
   - View the agent used, answer, and source documents
   
3. **View System Logs**
   - Check the "System Logs" tab to see processing steps and decisions
   
4. **Configure Settings**
   - In the "Settings" tab, you can enter your Google API key if needed

## Example Queries

- **RAG Agent**: "What are the main benefits of our health insurance plan?"
- **Calculator Agent**: "Calculate the sum of 525 and 176"
- **Dictionary Agent**: "Define the term machine learning"

## Implementation Details

### Document Chunking
Documents are divided into smaller chunks (default: 500 characters with 100 character overlap) to enable more precise retrieval. This granularity helps the system find specific information rather than returning entire documents.

### Retrieval System
The retrieval system uses semantic search to find the top 3 most relevant chunks based on vector similarity. These chunks are then passed to the LLM to generate a comprehensive answer.

### Agent Decision Flow
1. Query is analyzed for intent
2. Routed to the appropriate agent (Calculator, Dictionary, or RAG)
3. Agent processes the query using its specialized tools
4. Results are displayed with appropriate attribution

