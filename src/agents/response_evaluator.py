"""
Agent 4: Response Evaluator
Uses LLM and rule-based scoring to assess clarity, confidence, technical correctness, etc.
"""

from typing import Dict, Any, List, Tuple
from google import genai
from google.genai import types
import re
import json
import os


class ResponseEvaluator:
    def __init__(self, api_key: str):
        """
        Initialize the Response Evaluator.
        
        Args:
            api_key: Google API key for LLM
        """
        self.client = genai.Client(api_key=api_key)
        self.model = 'gemini-2.5-flash-lite-preview-06-17'
        
        # Evaluation criteria weights
        self.criteria_weights = {
            "clarity": 0.2,
            "confidence": 0.15,
            "technical_correctness": 0.25,
            "completeness": 0.15,
            "relevance": 0.15,
            "communication_skills": 0.1
        }
    
    def evaluate_response(self, question: str, response: str, role: str, 
                         question_category: str = "general") -> Dict[str, Any]:
        """
        Evaluate a candidate's response comprehensively.
        
        Args:
            question: The question that was asked
            response: Candidate's response
            role: Job role being interviewed for
            question_category: Category of the question (technical, behavioral, etc.)
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            # Get LLM-based evaluation
            llm_evaluation = self._evaluate_with_llm(question, response, role, question_category)
            
            # Get rule-based metrics
            rule_based_metrics = self._calculate_rule_based_metrics(response)
            
            # Combine evaluations
            combined_score = self._combine_evaluations(llm_evaluation, rule_based_metrics)
            
            # Generate detailed feedback
            feedback = self._generate_feedback(llm_evaluation, rule_based_metrics, combined_score)
            
            return {
                "overall_score": combined_score,
                "llm_evaluation": llm_evaluation,
                "rule_based_metrics": rule_based_metrics,
                "feedback": feedback,
                "question": question,
                "response": response,
                "role": role,
                "category": question_category
            }
            
        except Exception as e:
            print(f"Error evaluating response: {e}")
            return self._get_fallback_evaluation(question, response, role)
    
    def _evaluate_with_llm(self, question: str, response: str, role: str, 
                          question_category: str) -> Dict[str, Any]:
        """Evaluate response using LLM."""
        try:
            prompt = f"""
            Evaluate the candidate's response to an interview question. Provide scores and detailed analysis.
            
            Question: {question}
            Candidate's Response: {response}
            Job Role: {role}
            Question Category: {question_category}
            
            Evaluate the response on the following criteria (0-10 scale):
            1. Clarity: How clear and understandable is the response?
            2. Confidence: How confident does the candidate appear?
            3. Technical Correctness: How accurate is the technical information?
            4. Completeness: How complete is the answer?
            5. Relevance: How relevant is the response to the question?
            6. Communication Skills: How well does the candidate communicate?
            
            Provide your evaluation in the following JSON format:
            {{
                "clarity": {{"score": 8, "explanation": "Clear explanation with good examples"}},
                "confidence": {{"score": 7, "explanation": "Shows confidence but could be more assertive"}},
                "technical_correctness": {{"score": 9, "explanation": "Technically accurate with good understanding"}},
                "completeness": {{"score": 6, "explanation": "Covers main points but could be more detailed"}},
                "relevance": {{"score": 8, "explanation": "Directly addresses the question"}},
                "communication_skills": {{"score": 7, "explanation": "Good communication but could improve structure"}}
            }}
            
            Also provide:
            - Overall assessment (1-2 sentences)
            - Key strengths (list)
            - Areas for improvement (list)
            - Suggested follow-up questions (list)
            """
            
            response_llm = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            # Parse JSON response
            try:
                evaluation_text = response_llm.text
                # Extract JSON part
                json_start = evaluation_text.find('{')
                json_end = evaluation_text.rfind('}') + 1
                json_str = evaluation_text[json_start:json_end]
                
                evaluation = json.loads(json_str)
                
                # Extract additional information
                lines = evaluation_text.split('\n')
                overall_assessment = ""
                strengths = []
                improvements = []
                follow_ups = []
                
                for line in lines:
                    if "Overall assessment:" in line or "Overall:" in line:
                        overall_assessment = line.split(":", 1)[1].strip()
                    elif "Key strengths:" in line or "Strengths:" in line:
                        # Extract strengths from subsequent lines
                        pass
                    elif "Areas for improvement:" in line or "Improvements:" in line:
                        # Extract improvements from subsequent lines
                        pass
                    elif "Suggested follow-up:" in line or "Follow-up:" in line:
                        # Extract follow-ups from subsequent lines
                        pass
                
                evaluation.update({
                    "overall_assessment": overall_assessment,
                    "strengths": strengths,
                    "improvements": improvements,
                    "follow_ups": follow_ups
                })
                
                return evaluation
                
            except json.JSONDecodeError:
                # Fallback parsing
                return self._parse_llm_response_fallback(response_llm.text)
                
        except Exception as e:
            print(f"Error in LLM evaluation: {e}")
            return self._get_default_llm_evaluation()
    
    def _parse_llm_response_fallback(self, response_text: str) -> Dict[str, Any]:
        """Fallback parsing for LLM response."""
        # Extract scores using regex
        scores = {}
        for criterion in ["clarity", "confidence", "technical_correctness", "completeness", "relevance", "communication_skills"]:
            pattern = rf"{criterion}.*?(\d+)"
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                scores[criterion] = {"score": int(match.group(1)), "explanation": "Score extracted from response"}
            else:
                scores[criterion] = {"score": 5, "explanation": "Default score"}
        
        return scores
    
    def _calculate_rule_based_metrics(self, response: str) -> Dict[str, Any]:
        """Calculate rule-based metrics for the response."""
        metrics = {}
        
        # Response length
        word_count = len(response.split())
        metrics["word_count"] = word_count
        metrics["response_length_score"] = min(10, max(1, word_count / 10))  # 1-10 scale
        
        # Technical terms count (basic heuristic)
        technical_terms = [
            "algorithm", "data", "model", "analysis", "machine learning", "AI", "database",
            "API", "framework", "optimization", "scalability", "performance", "testing",
            "deployment", "architecture", "design pattern", "version control", "agile",
            "scrum", "stakeholder", "requirement", "user experience", "interface"
        ]
        
        tech_term_count = sum(1 for term in technical_terms if term.lower() in response.lower())
        metrics["technical_terms_count"] = tech_term_count
        metrics["technical_depth_score"] = min(10, tech_term_count * 2)
        
        # Sentence structure
        sentences = response.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        metrics["avg_sentence_length"] = avg_sentence_length
        metrics["sentence_structure_score"] = min(10, max(1, 20 - avg_sentence_length))
        
        # Confidence indicators
        confidence_indicators = ["i believe", "i think", "maybe", "perhaps", "might", "could"]
        uncertainty_count = sum(1 for indicator in confidence_indicators if indicator in response.lower())
        metrics["uncertainty_count"] = uncertainty_count
        metrics["confidence_score"] = max(1, 10 - uncertainty_count)
        
        # Specificity (numbers, dates, specific examples)
        specificity_indicators = re.findall(r'\d+', response)
        metrics["specificity_count"] = len(specificity_indicators)
        metrics["specificity_score"] = min(10, len(specificity_indicators) * 2)
        
        return metrics
    
    def _combine_evaluations(self, llm_evaluation: Dict[str, Any], 
                           rule_based_metrics: Dict[str, Any]) -> float:
        """Combine LLM and rule-based evaluations into overall score."""
        try:
            # Extract scores from LLM evaluation
            llm_scores = {}
            for criterion, weight in self.criteria_weights.items():
                if criterion in llm_evaluation:
                    llm_scores[criterion] = llm_evaluation[criterion]["score"]
                else:
                    llm_scores[criterion] = 5  # Default score
            
            # Calculate weighted LLM score
            llm_weighted_score = sum(llm_scores[criterion] * weight 
                                   for criterion, weight in self.criteria_weights.items())
            
            # Calculate rule-based score (simple average)
            rule_based_scores = [
                rule_based_metrics.get("response_length_score", 5),
                rule_based_metrics.get("technical_depth_score", 5),
                rule_based_metrics.get("sentence_structure_score", 5),
                rule_based_metrics.get("confidence_score", 5),
                rule_based_metrics.get("specificity_score", 5)
            ]
            rule_based_avg = sum(rule_based_scores) / len(rule_based_scores)
            
            # Combine scores (70% LLM, 30% rule-based)
            combined_score = (llm_weighted_score * 0.7) + (rule_based_avg * 0.3)
            
            return round(combined_score, 2)
            
        except Exception as e:
            print(f"Error combining evaluations: {e}")
            return 5.0  # Default score
    
    def _generate_feedback(self, llm_evaluation: Dict[str, Any], 
                          rule_based_metrics: Dict[str, Any], 
                          overall_score: float) -> str:
        """Generate comprehensive feedback based on evaluation."""
        try:
            feedback_prompt = f"""
            Generate constructive feedback for a candidate based on their interview response evaluation.
            
            Overall Score: {overall_score}/10
            LLM Evaluation: {json.dumps(llm_evaluation, indent=2)}
            Rule-based Metrics: {json.dumps(rule_based_metrics, indent=2)}
            
            Provide:
            1. Overall assessment (2-3 sentences)
            2. Specific strengths (bullet points)
            3. Areas for improvement (bullet points)
            4. Actionable recommendations (bullet points)
            
            Keep the feedback constructive, specific, and actionable.
            """
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=feedback_prompt
            )
            
            return response.text
            
        except Exception as e:
            print(f"Error generating feedback: {e}")
            return self._get_default_feedback(overall_score)
    
    def _get_default_feedback(self, overall_score: float) -> str:
        """Default feedback if LLM generation fails."""
        if overall_score >= 8:
            return "Excellent response! Strong technical knowledge and communication skills demonstrated."
        elif overall_score >= 6:
            return "Good response with room for improvement. Consider providing more specific examples."
        elif overall_score >= 4:
            return "Adequate response. Focus on improving clarity and providing more detailed explanations."
        else:
            return "Response needs improvement. Consider practicing interview questions and providing more comprehensive answers."
    
    def _get_fallback_evaluation(self, question: str, response: str, role: str) -> Dict[str, Any]:
        """Fallback evaluation if main evaluation fails."""
        return {
            "overall_score": 5.0,
            "llm_evaluation": self._get_default_llm_evaluation(),
            "rule_based_metrics": self._calculate_rule_based_metrics(response),
            "feedback": "Evaluation could not be completed. Please try again.",
            "question": question,
            "response": response,
            "role": role,
            "category": "general"
        }
    
    def _get_default_llm_evaluation(self) -> Dict[str, Any]:
        """Default LLM evaluation if generation fails."""
        return {
            "clarity": {"score": 5, "explanation": "Default score"},
            "confidence": {"score": 5, "explanation": "Default score"},
            "technical_correctness": {"score": 5, "explanation": "Default score"},
            "completeness": {"score": 5, "explanation": "Default score"},
            "relevance": {"score": 5, "explanation": "Default score"},
            "communication_skills": {"score": 5, "explanation": "Default score"}
        }
    
    def evaluate_interview_session(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate an entire interview session.
        
        Args:
            evaluations: List of individual response evaluations
            
        Returns:
            Session evaluation summary
        """
        if not evaluations:
            return {"error": "No evaluations provided"}
        
        # Calculate session metrics
        total_score = sum(eval["overall_score"] for eval in evaluations)
        avg_score = total_score / len(evaluations)
        
        # Categorize scores
        excellent_responses = sum(1 for eval in evaluations if eval["overall_score"] >= 8)
        good_responses = sum(1 for eval in evaluations if 6 <= eval["overall_score"] < 8)
        needs_improvement = sum(1 for eval in evaluations if eval["overall_score"] < 6)
        
        # Identify strengths and weaknesses across responses
        all_strengths = []
        all_weaknesses = []
        
        for eval in evaluations:
            if "llm_evaluation" in eval and "strengths" in eval["llm_evaluation"]:
                all_strengths.extend(eval["llm_evaluation"]["strengths"])
            if "llm_evaluation" in eval and "improvements" in eval["llm_evaluation"]:
                all_weaknesses.extend(eval["llm_evaluation"]["improvements"])
        
        return {
            "session_score": round(avg_score, 2),
            "total_responses": len(evaluations),
            "excellent_responses": excellent_responses,
            "good_responses": good_responses,
            "needs_improvement": needs_improvement,
            "common_strengths": list(set(all_strengths))[:5],
            "common_weaknesses": list(set(all_weaknesses))[:5],
            "recommendation": self._get_session_recommendation(avg_score)
        }
    
    def _get_session_recommendation(self, avg_score: float) -> str:
        """Get recommendation based on session score."""
        if avg_score >= 8:
            return "Strong candidate - recommend for next round"
        elif avg_score >= 6:
            return "Good candidate - consider for next round with some reservations"
        elif avg_score >= 4:
            return "Adequate candidate - may need additional evaluation"
        else:
            return "Candidate needs significant improvement - not recommended for next round"


if __name__ == "__main__":
    # Test the response evaluator
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        evaluator = ResponseEvaluator(api_key)
        
        # Test evaluation
        question = "Explain the difference between supervised and unsupervised learning"
        response = "Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data. For example, in supervised learning, you might train a model to classify emails as spam or not spam using labeled examples. In unsupervised learning, you might use clustering to group customers based on their behavior without predefined categories."
        
        evaluation = evaluator.evaluate_response(question, response, "Data Scientist", "technical")
        print("Evaluation Results:")
        print(f"Overall Score: {evaluation['overall_score']}/10")
        print(f"Feedback: {evaluation['feedback']}")
    else:
        print("Please set GOOGLE_API_KEY environment variable") 