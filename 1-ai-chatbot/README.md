# AI Customer Support Chatbot (RAG)

An intelligent customer support chatbot using Retrieval-Augmented Generation (RAG) to provide contextual, accurate responses based on a knowledge base.

## Features
- 🤖 RAG-powered responses using OpenAI
- 📚 Vector-based document retrieval
- 💬 Flask REST API
- 🔄 Conversation history tracking
- ⚡ Real-time streaming responses

## Tech Stack
- Python 3.9+
- Flask
- OpenAI GPT-4
- FAISS (vector database)
- Pandas

## Setup

### 1. Install Dependencies
```bash
cd 1-ai-chatbot
pip install -r requirements.txt
```

### 2. Environment Variables
Create `.env` file:
```
OPENAI_API_KEY=your_openai_api_key
FLASK_ENV=development
```

### 3. Run the Application
```bash
python app.py
```

Server runs on `http://localhost:5000`

## API Endpoints

### POST /api/chat
Send a message to the chatbot.

**Request:**
```json
{
  "message": "How do I reset my password?",
  "conversation_id": "user_123"
}
```

**Response:**
```json
{
  "response": "To reset your password...",
  "sources": ["doc1.pdf", "doc2.pdf"],
  "confidence": 0.95
}
```

### GET /api/health
Health check endpoint.

## Project Structure
```
1-ai-chatbot/
├── app.py                 # Main Flask app
├── rag_engine.py          # RAG logic
├── vector_store.py        # Vector database handler
├── requirements.txt       # Dependencies
├── .env                   # Environment variables
├── data/
│   └── documents/         # Knowledge base documents
└── models/
    └── faiss_index/       # Vector embeddings
```

## How It Works
1. Documents are embedded using OpenAI's embedding model
2. Embeddings stored in FAISS for fast retrieval
3. When user queries, relevant documents retrieved
4. Retrieved context + query sent to GPT-4 for response
5. Response streamed back to user

## Extending
- Add more documents to `data/documents/`
- Modify `RAG_CONFIG` in `rag_engine.py` to adjust retrieval parameters
- Connect to a real database for conversation history
- Add user authentication

## License
MIT