from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from rag_engine import RAGEngine
import logging

load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize RAG engine
rag_engine = RAGEngine(api_key=os.getenv('OPENAI_API_KEY'))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'AI Chatbot'}), 200

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message')
        conversation_id = data.get('conversation_id', 'default')
        
        if not user_message:
            return jsonify({'error': 'Message required'}), 400
        
        # Get response from RAG engine
        response = rag_engine.query(user_message, conversation_id)
        
        return jsonify({
            'response': response['answer'],
            'sources': response['sources'],
            'confidence': response['confidence'],
            'conversation_id': conversation_id
        }), 200
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize vector store with documents"""
    try:
        rag_engine.initialize_vector_store()
        return jsonify({'message': 'Vector store initialized successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset conversation"""
    conversation_id = request.json.get('conversation_id')
    rag_engine.reset_conversation(conversation_id)
    return jsonify({'message': 'Conversation reset'}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)