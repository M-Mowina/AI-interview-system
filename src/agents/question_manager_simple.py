"""
Agent 3: Question Manager (Simplified)
Pulls questions tailored to the role from in-memory storage and manages question flow.
"""

from typing import List, Dict, Any, Optional
import random
from google import genai
from google.genai import types
import os


class QuestionManager:
    def __init__(self, api_key: str):
        """
        Initialize the Question Manager.
        
        Args:
            api_key: Google API key for LLM
        """
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.model = 'gemini-2.5-flash-lite-preview-06-17'
        
        # Simple in-memory question storage (replaces ChromaDB)
        self.questions_db = {
            "Data Scientist": {
                "technical": [
                    "Explain the difference between supervised and unsupervised learning",
                    "How would you handle missing data in a dataset?",
                    "Describe a machine learning project you've worked on",
                    "What is overfitting and how do you prevent it?",
                    "Explain the concept of cross-validation",
                    "What is the bias-variance tradeoff?",
                    "How do you evaluate a classification model?",
                    "Explain the difference between precision and recall"
                ],
                "behavioral": [
                    "Tell me about a time you had to explain complex technical concepts to non-technical stakeholders",
                    "How do you stay updated with the latest developments in data science?",
                    "Describe a challenging problem you solved using data analysis",
                    "How do you handle conflicting requirements from different stakeholders?",
                    "Tell me about a time you had to work with a difficult team member",
                    "How do you prioritize multiple projects with tight deadlines?",
                    "Describe a situation where you had to learn a new technology quickly"
                ],
                "experience": [
                    "What tools and technologies are you most comfortable with?",
                    "Describe your experience with big data technologies",
                    "How do you approach data cleaning and preprocessing?",
                    "What's your experience with cloud platforms for ML deployment?"
                ]
            },
            "Software Engineer": {
                "technical": [
                    "Explain the difference between REST and GraphQL APIs",
                    "How would you optimize a slow database query?",
                    "Describe your experience with version control systems",
                    "What is the difference between synchronous and asynchronous programming?",
                    "How do you handle memory management in your applications?",
                    "Explain the concept of microservices architecture",
                    "How do you approach testing in your development process?"
                ],
                "behavioral": [
                    "Tell me about a time you had to work with a difficult team member",
                    "How do you handle tight deadlines?",
                    "Describe a project where you had to learn a new technology quickly",
                    "How do you approach code reviews?",
                    "Tell me about a time you had to debug a complex issue",
                    "How do you handle technical disagreements with colleagues?"
                ],
                "experience": [
                    "What programming languages are you most proficient in?",
                    "Describe your experience with cloud platforms",
                    "How do you stay updated with new technologies?",
                    "What's your experience with CI/CD pipelines?"
                ]
            },
            "Product Manager": {
                "technical": [
                    "How do you prioritize features in a product roadmap?",
                    "Explain your approach to user research and validation",
                    "How do you measure product success?",
                    "What metrics would you track for a mobile app?",
                    "How do you handle competing stakeholder requirements?"
                ],
                "behavioral": [
                    "Tell me about a time you had to make a difficult product decision",
                    "How do you handle conflicting stakeholder requirements?",
                    "Describe a product launch you managed",
                    "How do you gather and prioritize user feedback?",
                    "Tell me about a time you had to pivot a product strategy"
                ],
                "experience": [
                    "What tools do you use for product management?",
                    "Describe your experience with agile methodologies",
                    "How do you approach market research?",
                    "What's your experience with A/B testing?"
                ]
            }
        }
        
        # Question flow state
        self.current_question_index = 0
        self.asked_questions = []
        self.question_categories = ["introduction", "technical", "behavioral", "experience", "closing"]
        
    def add_questions_to_knowledge_base(self, questions: List[Dict[str, Any]]):
        """
        Add questions to the knowledge base.
        
        Args:
            questions: List of question dictionaries with keys: text, category, role, difficulty
        """
        try:
            for question in questions:
                role = question.get("role", "Data Scientist")
                category = question.get("category", "technical")
                
                if role not in self.questions_db:
                    self.questions_db[role] = {}
                if category not in self.questions_db[role]:
                    self.questions_db[role][category] = []
                
                self.questions_db[role][category].append(question["text"])
            
            print(f"Added {len(questions)} questions to knowledge base")
            
        except Exception as e:
            print(f"Error adding questions to knowledge base: {e}")
    
    def get_questions_by_role(self, role: str, category: Optional[str] = None, 
                            difficulty: Optional[str] = None, limit: int = 10) -> List[str]:
        """
        Retrieve questions from knowledge base based on criteria.
        
        Args:
            role: Job role
            category: Question category
            difficulty: Question difficulty (ignored in simple implementation)
            limit: Maximum number of questions to return
            
        Returns:
            List of question texts
        """
        try:
            if role not in self.questions_db:
                return self._get_fallback_questions(role, category)
            
            if category and category in self.questions_db[role]:
                questions = self.questions_db[role][category]
            else:
                # Return all questions for the role
                questions = []
                for cat_questions in self.questions_db[role].values():
                    questions.extend(cat_questions)
            
            # Apply limit and randomize
            random.shuffle(questions)
            return questions[:limit]
            
        except Exception as e:
            print(f"Error retrieving questions: {e}")
            return self._get_fallback_questions(role, category)
    
    def _get_fallback_questions(self, role: str, category: Optional[str] = None) -> List[str]:
        """Fallback questions if knowledge base query fails."""
        fallback_questions = {
            "Data Scientist": {
                "technical": [
                    "Explain the difference between supervised and unsupervised learning",
                    "How would you handle missing data in a dataset?",
                    "Describe a machine learning project you've worked on",
                    "What is overfitting and how do you prevent it?",
                    "Explain the concept of cross-validation"
                ],
                "behavioral": [
                    "Tell me about a time you had to explain complex technical concepts to non-technical stakeholders",
                    "How do you stay updated with the latest developments in data science?",
                    "Describe a challenging problem you solved using data analysis",
                    "How do you handle conflicting requirements from different stakeholders?",
                    "Tell me about a time you had to work with a difficult team member"
                ]
            },
            "Software Engineer": {
                "technical": [
                    "Explain the difference between REST and GraphQL APIs",
                    "How would you optimize a slow database query?",
                    "Describe your experience with version control systems",
                    "What is the difference between synchronous and asynchronous programming?",
                    "How do you handle memory management in your applications?"
                ],
                "behavioral": [
                    "Tell me about a time you had to work with a difficult team member",
                    "How do you handle tight deadlines?",
                    "Describe a project where you had to learn a new technology quickly",
                    "How do you approach code reviews?",
                    "Tell me about a time you had to debug a complex issue"
                ]
            }
        }
        
        if role in fallback_questions:
            if category and category in fallback_questions[role]:
                return fallback_questions[role][category]
            else:
                # Return all questions for the role
                all_questions = []
                for cat_questions in fallback_questions[role].values():
                    all_questions.extend(cat_questions)
                return all_questions
        
        # Default questions
        return [
            "Tell me about yourself",
            "What are your strengths and weaknesses?",
            "Why are you interested in this position?",
            "Where do you see yourself in 5 years?",
            "What are your salary expectations?"
        ]
    
    def generate_follow_up_question(self, candidate_response: str, original_question: str, 
                                  role: str) -> str:
        """
        Generate a follow-up question based on candidate's response.
        
        Args:
            candidate_response: Candidate's answer to the previous question
            original_question: The original question that was asked
            role: Job role
            
        Returns:
            Follow-up question
        """
        try:
            prompt = f"""
            Based on the candidate's response to an interview question, generate a relevant follow-up question.
            
            Original Question: {original_question}
            Candidate's Response: {candidate_response}
            Job Role: {role}
            
            Generate a follow-up question that:
            1. Builds upon their response
            2. Probes deeper into their experience or knowledge
            3. Is relevant to the job role
            4. Maintains a professional tone
            
            Return only the follow-up question, nothing else.
            """
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            return response.text.strip()
            
        except Exception as e:
            print(f"Error generating follow-up question: {e}")
            return "Could you elaborate on that?"
    
    def get_next_question(self, role: str, current_category: Optional[str] = None, 
                         candidate_response: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the next question in the interview flow.
        
        Args:
            role: Job role
            current_category: Current question category
            candidate_response: Candidate's response to previous question (for follow-ups)
            
        Returns:
            Dictionary with question text and metadata
        """
        # Determine next category if not provided
        if not current_category:
            if not self.asked_questions:
                current_category = "introduction"
            else:
                # Progress through categories
                category_progression = ["introduction", "technical", "behavioral", "experience", "closing"]
                current_index = len(self.asked_questions) // 3  # Change category every 3 questions
                current_category = category_progression[min(current_index, len(category_progression) - 1)]
        
        # Get questions for the category
        questions = self.get_questions_by_role(role, category=current_category, limit=20)
        
        if not questions:
            questions = self._get_fallback_questions(role, current_category)
        
        # Filter out already asked questions
        available_questions = [q for q in questions if q not in self.asked_questions]
        
        if not available_questions:
            # If no questions available, generate a follow-up
            if candidate_response and self.asked_questions:
                follow_up = self.generate_follow_up_question(
                    candidate_response, 
                    self.asked_questions[-1], 
                    role
                )
                return {
                    "text": follow_up,
                    "category": current_category,
                    "type": "follow_up"
                }
            else:
                # Fallback to any question
                available_questions = questions
        
        # Select next question
        next_question = random.choice(available_questions)
        self.asked_questions.append(next_question)
        
        return {
            "text": next_question,
            "category": current_category,
            "type": "standard"
        }
    
    def reset_interview(self):
        """Reset the interview state."""
        self.current_question_index = 0
        self.asked_questions = []
    
    def get_interview_progress(self) -> Dict[str, Any]:
        """Get current interview progress."""
        return {
            "total_questions_asked": len(self.asked_questions),
            "asked_questions": self.asked_questions,
            "current_category": self._get_current_category()
        }
    
    def _get_current_category(self) -> str:
        """Determine current category based on questions asked."""
        if not self.asked_questions:
            return "introduction"
        
        category_progression = ["introduction", "technical", "behavioral", "experience", "closing"]
        current_index = len(self.asked_questions) // 3
        return category_progression[min(current_index, len(category_progression) - 1)]


if __name__ == "__main__":
    # Test the question manager
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        qm = QuestionManager(api_key)
        
        # Test getting questions
        questions = qm.get_questions_by_role("Data Scientist", category="technical")
        print("Technical questions for Data Scientist:")
        for i, q in enumerate(questions[:3], 1):
            print(f"{i}. {q}")
        
        # Test follow-up generation
        follow_up = qm.generate_follow_up_question(
            "I worked on a machine learning project that predicted customer churn using random forests.",
            "Describe a machine learning project you've worked on",
            "Data Scientist"
        )
        print(f"\nFollow-up question: {follow_up}")
    else:
        print("Please set GOOGLE_API_KEY environment variable") 