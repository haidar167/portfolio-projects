import openai
from typing import Dict, List, Any
import json

class ResearchOrchestrator:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.model = "gpt-4"
        self.agents = {
            'researcher': ResearcherAgent(api_key),
            'analyzer': AnalyzerAgent(api_key),
            'writer': WriterAgent(api_key),
            'fact_checker': FactCheckerAgent(api_key)
        }
    
    def research(self, topic: str, depth: str = 'standard') -> Dict[str, Any]:
        """Orchestrate the research process"""
        
        # Phase 1: Research
        research_data = self.agents['researcher'].research(topic, depth)
        
        # Phase 2: Analyze
        analysis = self.agents['analyzer'].analyze(topic, research_data)
        
        # Phase 3: Write Report
        report = self.agents['writer'].write_report(topic, analysis, depth)
        
        # Phase 4: Fact Check
        verified_report = self.agents['fact_checker'].verify(report)
        
        return {
            'topic': topic,
            'depth': depth,
            'research_data': research_data,
            'analysis': analysis,
            'report': verified_report,
            'timestamp': self._get_timestamp()
        }
    
    def _get_timestamp(self):
        from datetime import datetime
        return datetime.now().isoformat()

class ResearcherAgent:
    def __init__(self, api_key: str):
        openai.api_key = api_key
    
    def research(self, topic: str, depth: str) -> Dict[str, Any]:
        """Research a topic"""
        prompt = f"""
        Research the following topic thoroughly: {topic}
        Depth level: {depth}
        
        Provide:
        1. Key concepts
        2. Recent developments
        3. Important statistics
        4. Notable experts/organizations
        5. References
        
        Format as JSON.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return {'raw_research': response['choices'][0]['message']['content']}

class AnalyzerAgent:
    def __init__(self, api_key: str):
        openai.api_key = api_key
    
    def analyze(self, topic: str, data: Dict) -> Dict[str, Any]:
        """Analyze research data"""
        prompt = f"""
        Analyze this research data about {topic}:
        {data.get('raw_research', '')}
        
        Provide:
        1. Key themes
        2. Emerging trends
        3. Implications
        4. Gaps in knowledge
        
        Format as JSON.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return {'analysis': response['choices'][0]['message']['content']}

class WriterAgent:
    def __init__(self, api_key: str):
        openai.api_key = api_key
    
    def write_report(self, topic: str, analysis: Dict, depth: str) -> str:
        """Write comprehensive report"""
        prompt = f"""
        Write a {depth} report about {topic} based on this analysis:
        {analysis.get('analysis', '')}
        
        Structure:
        # Executive Summary
        # Introduction
        # Key Findings
        # Analysis & Implications
        # Trends & Future Outlook
        # Recommendations
        # Conclusion
        # References
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        
        return response['choices'][0]['message']['content']

class FactCheckerAgent:
    def __init__(self, api_key: str):
        openai.api_key = api_key
    
    def verify(self, report: str) -> str:
        """Verify facts in report"""
        prompt = f"""
        Review this report for accuracy and consistency:
        {report}
        
        Ensure:
        1. No contradictions
        2. Logical flow
        3. Evidence-based statements
        
        Return verified report with any corrections marked.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        
        return response['choices'][0]['message']['content']