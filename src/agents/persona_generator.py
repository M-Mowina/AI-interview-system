"""
Agent 1: Persona Generator
Customizes the interviewer persona based on company name, role, and preferences.
"""

from typing import Dict, Any
from google import genai
from google.genai import types
import os


class PersonaGenerator:
    def __init__(self, api_key: str):
        """Initialize the Persona Generator agent."""
        self.client = genai.Client(api_key=api_key)
        self.model = 'gemini-2.5-flash-lite-preview-06-17'
        
    def generate_persona(self, company_name: str, role: str, preferences: Dict[str, Any] = None) -> str:
        """
        Generate a customized interviewer persona.
        
        Args:
            company_name: Name of the company
            role: Position being interviewed for
            preferences: Additional preferences (interview style, focus areas, etc.)
            
        Returns:
            Customized system prompt for the interviewer
        """
        
        # Default preferences
        default_preferences = {
            "interview_style": "professional yet approachable",
            "focus_areas": ["technical skills", "problem-solving", "communication", "experience"],
            "tone": "friendly but professional",
            "duration": "30-45 minutes"
        }
        
        if preferences:
            default_preferences.update(preferences)
        
        # Create the persona generation prompt
        persona_prompt = f"""
        Create a detailed interviewer persona for conducting interviews at {company_name} for the {role} position.
        
        Company: {company_name}
        Role: {role}
        Interview Style: {default_preferences['interview_style']}
        Focus Areas: {', '.join(default_preferences['focus_areas'])}
        Tone: {default_preferences['tone']}
        Duration: {default_preferences['duration']}
        
        The persona should include:
        1. Introduction and greeting style
        2. Question types and progression
        3. Evaluation criteria
        4. Communication style
        5. Specific focus areas for the role
        
        Generate a comprehensive system instruction that can be used to configure an AI interviewer.
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=persona_prompt
            )
            
            return response.text
            
        except Exception as e:
            print(f"Error generating persona: {e}")
            # Fallback to a basic persona
            return self._get_fallback_persona(company_name, role)
    
    def _get_fallback_persona(self, company_name: str, role: str) -> str:
        """Fallback persona if generation fails."""
        return f"""
        You are an AI interviewer for {company_name}, conducting an interview for the {role} position.
        
        Your goal is to assess the candidate's technical skills, problem-solving abilities, communication skills, and experience relevant to this role.
        
        Maintain a professional yet approachable tone. Start by introducing yourself and asking the candidate to introduce themselves.
        
        Focus on:
        - Technical skills relevant to the position
        - Problem-solving and analytical thinking
        - Communication and collaboration abilities
        - Relevant experience and projects
        - Cultural fit and motivation
        
        Ask follow-up questions based on their responses and provide a comprehensive evaluation.
        """
    
    def get_role_specific_questions(self, role: str) -> Dict[str, list]:
        """Get role-specific question categories."""
        question_templates = {
            "Data Scientist": {
                "technical": [
                    "Explain the difference between supervised and unsupervised learning",
                    "How would you handle missing data in a dataset?",
                    "Describe a machine learning project you've worked on"
                ],
                "behavioral": [
                    "Tell me about a time you had to explain complex technical concepts to non-technical stakeholders",
                    "How do you stay updated with the latest developments in data science?",
                    "Describe a challenging problem you solved using data analysis"
                ]
            },
            "Software Engineer": {
                "technical": [
                    "Explain the difference between REST and GraphQL APIs",
                    "How would you optimize a slow database query?",
                    "Describe your experience with version control systems"
                ],
                "behavioral": [
                    "Tell me about a time you had to work with a difficult team member",
                    "How do you handle tight deadlines?",
                    "Describe a project where you had to learn a new technology quickly"
                ]
            },
            "Product Manager": {
                "technical": [
                    "How do you prioritize features in a product roadmap?",
                    "Explain your approach to user research and validation",
                    "How do you measure product success?"
                ],
                "behavioral": [
                    "Tell me about a time you had to make a difficult product decision",
                    "How do you handle conflicting stakeholder requirements?",
                    "Describe a product launch you managed"
                ]
            }
        }
        
        return question_templates.get(role, question_templates["Data Scientist"])


if __name__ == "__main__":
    # Test the persona generator
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        generator = PersonaGenerator(api_key)
        persona = generator.generate_persona("TechCorp", "Data Scientist")
        print("Generated Persona:")
        print(persona)
    else:
        print("Please set GOOGLE_API_KEY environment variable") 