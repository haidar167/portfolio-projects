import os
import json
from typing import Dict, List, Any
import openai
import faiss
import numpy as np
from vector_store import VectorStore

class RAGEngine:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.vector_store = VectorStore()
        self.conversations = {}
        self.model = "gpt-4"
        self.embedding_model = "text-embedding-ada-002"
        
    def initialize_vector_store(self):
        """Initialize vector store with documents"""
        docs_path = 'data/documents'
        if os.path.exists(docs_path):
            documents = self._load_documents(docs_path)
            self.vector_store.add_documents(documents)
        else:
            print(f"Warning: {docs_path} not found. Using sample data.")
            self._add_sample_documents()
    
    def _load_documents(self, path: str) -> List[Dict[str, str]]:
        """Load documents from directory"""
        documents = []
        for filename in os.listdir(path):
            if filename.endswith('.txt'):
                with open(os.path.join(path, filename), 'r') as f:
                    documents.append({
                        'id': filename,
                        'content': f.read(),
                        'source': filename
                    })
        return documents
    
    def _add_sample_documents(self):
        """Add sample documents for demo"""
        sample_docs = [
            {
                'id': 'faq_1',
                'content': 'How to reset password: Go to login page, click "Forgot Password", enter email, and follow the link in your email.',
                'source': 'FAQ'
            },
            {
                'id': 'faq_2',
                'content': 'Account issues: If you cannot access your account, contact support@company.com with your username.',
                'source': 'FAQ'
            },
            {
                'id': 'faq_3',
                'content': 'Billing questions: Check your invoice in Settings > Billing. For disputes, email billing@company.com.',
                'source': 'Billing'
            }
        ]
        self.vector_store.add_documents(sample_docs)
    
    def query(self, query: str, conversation_id: str) -> Dict[str, Any]:
        """Process user query and return response"""
        # Retrieve relevant documents
        retrieved_docs = self.vector_store.search(query, top_k=3)
        
        # Build context from retrieved documents
        context = self._build_context(retrieved_docs)
        
        # Generate response using GPT-4
        response = self._generate_response(query, context, conversation_id)
        
        # Store in conversation history
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        self.conversations[conversation_id].append({
            'user': query,
            'assistant': response['answer']
        })
        
        return {
            'answer': response['answer'],
            'sources': [doc['source'] for doc in retrieved_docs],
            'confidence': response.get('confidence', 0.85)
        }
    
    def _build_context(self, documents: List[Dict]) -> str:
        """Build context string from retrieved documents"""
        context = "\n\nRelevant information:\n"
        for doc in documents:
            context += f"- {doc['content']}\n"
        return context
    
    def _generate_response(self, query: str, context: str, conversation_id: str) -> Dict[str, Any]:
        """Generate response using GPT-4"""
        conversation_history = self.conversations.get(conversation_id, [])
        
        # Build messages for API
        messages = [
            {"role": "system", "content": "You are a helpful customer support assistant. Use the provided information to answer questions accurately."},
        ]
        
        # Add conversation history
        for msg in conversation_history[-5:]:  # Last 5 messages
            messages.append({"role": "user", "content": msg['user']})
            messages.append({"role": "assistant", "content": msg['assistant']})
        
        # Add current query with context
        messages.append({
            "role": "user",
            "content": f"{context}\n\nUser question: {query}"
        })
        
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return {
            'answer': response['choices'][0]['message']['content'],
            'confidence': 0.9
        }
    
    def reset_conversation(self, conversation_id: str):
        """Reset conversation history"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]