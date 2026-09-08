# Agentic AI Research Assistant ⭐

A multi-agent AI research assistant that autonomously researches topics, gathers information, and generates comprehensive reports.

## Features
- 🤖 Multi-agent architecture
- 🔍 Web research capabilities
- 📝 Autonomous report generation
- 💡 Reasoning & analysis
- 🔄 Iterative refinement
- 📊 Structured output

## Tech Stack
- Python 3.9+
- OpenAI GPT-4
- Flask
- LangChain
- BeautifulSoup4

## Setup

```bash
cd 3-agentic-research-assistant
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5002`

## Agents

1. **Researcher Agent** - Gathers information on topics
2. **Analyzer Agent** - Analyzes and synthesizes data
3. **Writer Agent** - Creates formatted reports
4. **Fact-Checker Agent** - Validates information

## API Endpoints

```
POST /api/research - Start research task
GET /api/research/<task_id> - Get research status
GET /api/research/<task_id>/report - Get final report
```

## Example

```bash
curl -X POST http://localhost:5002/api/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "AI in Healthcare", "depth": "comprehensive"}'
```

## License
MIT