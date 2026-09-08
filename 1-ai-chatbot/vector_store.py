import numpy as np
import faiss
import openai
from typing import List, Dict, Any

class VectorStore:
    def __init__(self, embedding_model: str = "text-embedding-ada-002"):
        self.embedding_model = embedding_model
        self.documents = []
        self.index = None
        self.dimension = 1536  # Ada embedding dimension
        
    def add_documents(self, documents: List[Dict[str, str]]):
        """Add documents to vector store"""
        self.documents = documents
        
        # Generate embeddings
        embeddings = []
        for doc in documents:
            embedding = self._get_embedding(doc['content'])
            embeddings.append(embedding)
        
        # Create FAISS index
        embeddings_array = np.array(embeddings).astype('float32')
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings_array)
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        if self.index is None:
            return []
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_vector, top_k)
        
        # Return top results
        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        
        return results
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI API"""
        response = openai.Embedding.create(
            input=text,
            model=self.embedding_model
        )
        return response['data'][0]['embedding']