# DocChat AI

DocChat AI is an AI-powered PDF Question Answering application built using Retrieval-Augmented Generation (RAG). Users can upload PDF documents and ask questions in natural language. The application retrieves the most relevant content from the document and generates accurate responses using Groq's Llama models.

## Features

- PDF upload and document processing
- Retrieval-Augmented Generation (RAG)
- HuggingFace sentence embeddings
- ChromaDB vector database
- Maximum Marginal Relevance (MMR) retrieval
- Conversational memory
- Dark and Light mode
- Streamlit web interface
- Groq LLM integration

## Tech Stack

- Python
- Streamlit
- LangChain
- HuggingFace Embeddings
- ChromaDB
- Groq API
- PyPDF
- Sentence Transformers

## Project Structure

```
DocChat-AI/
│
├── rag_app_cloud.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/docchat-ai.git
cd docchat-ai
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configure API Key

### Local Development

Windows PowerShell

```powershell
$env:GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

Linux/macOS

```bash
export GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

### Streamlit Community Cloud

Add the following under **App Settings → Secrets**

```toml
GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

## Run the Application

```bash
streamlit run rag_app_cloud.py
```

## How It Works

1. Upload a PDF document.
2. The document is split into text chunks.
3. HuggingFace embeddings are generated.
4. Chunks are stored in ChromaDB.
5. MMR retrieves the most relevant context.
6. Groq Llama generates an answer using the retrieved context.
7. Conversation history is used for follow-up questions.

## Future Improvements

- Multi-PDF support
- Chat history persistence
- Source highlighting
- Citation-based responses
- User authentication
- PDF summarization
- Streaming responses

## License

This project is for educational and portfolio purposes.