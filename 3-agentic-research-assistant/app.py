from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from research_agent import ResearchOrchestrator
import logging
import uuid

load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize research orchestrator
orchestrator = ResearchOrchestrator(api_key=os.getenv('OPENAI_API_KEY'))

# Store research tasks
research_tasks = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'Research Assistant'}), 200

@app.route('/api/research', methods=['POST'])
def start_research():
    try:
        data = request.json
        topic = data.get('topic')
        depth = data.get('depth', 'standard')
        
        if not topic:
            return jsonify({'error': 'Topic required'}), 400
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Start research asynchronously
        try:
            report = orchestrator.research(topic, depth)
            research_tasks[task_id] = {
                'status': 'completed',
                'topic': topic,
                'report': report,
                'depth': depth
            }
        except Exception as e:
            research_tasks[task_id] = {
                'status': 'error',
                'error': str(e)
            }
        
        return jsonify({
            'task_id': task_id,
            'status': research_tasks[task_id]['status'],
            'topic': topic
        }), 201
    
    except Exception as e:
        logger.error(f"Error starting research: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/research/<task_id>', methods=['GET'])
def get_research_status(task_id):
    try:
        if task_id not in research_tasks:
            return jsonify({'error': 'Task not found'}), 404
        
        task = research_tasks[task_id]
        return jsonify({
            'task_id': task_id,
            'status': task['status'],
            'topic': task.get('topic'),
            'progress': 100 if task['status'] == 'completed' else 50
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/research/<task_id>/report', methods=['GET'])
def get_research_report(task_id):
    try:
        if task_id not in research_tasks:
            return jsonify({'error': 'Task not found'}), 404
        
        task = research_tasks[task_id]
        
        if task['status'] != 'completed':
            return jsonify({'error': 'Research not completed'}), 400
        
        return jsonify({
            'task_id': task_id,
            'topic': task['topic'],
            'report': task['report'],
            'depth': task['depth']
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)