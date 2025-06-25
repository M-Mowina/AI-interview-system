"""
Agent 5: Report Generator
Summarizes performance and sends feedback to HR email or dashboard.
"""

from typing import Dict, Any, List, Optional
from google import genai
from google.genai import types
import json
import os
from datetime import datetime
from fpdf import FPDF
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


class ReportGenerator:
    def __init__(self, api_key: str):
        """
        Initialize the Report Generator.
        
        Args:
            api_key: Google API key for LLM
        """
        self.client = genai.Client(api_key=api_key)
        self.model = 'gemini-2.5-flash-lite-preview-06-17'
        
    def generate_interview_report(self, candidate_info: Dict[str, Any], 
                                session_evaluation: Dict[str, Any],
                                individual_evaluations: List[Dict[str, Any]],
                                role: str, company_name: str) -> Dict[str, Any]:
        """
        Generate a comprehensive interview report.
        
        Args:
            candidate_info: Information about the candidate
            session_evaluation: Overall session evaluation
            individual_evaluations: List of individual response evaluations
            role: Job role being interviewed for
            company_name: Name of the company
            
        Returns:
            Dictionary containing the complete report
        """
        try:
            # Generate executive summary
            executive_summary = self._generate_executive_summary(
                candidate_info, session_evaluation, role
            )
            
            # Generate detailed analysis
            detailed_analysis = self._generate_detailed_analysis(
                individual_evaluations, session_evaluation
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                session_evaluation, individual_evaluations
            )
            
            # Compile the complete report
            report = {
                "report_id": f"INT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "generated_at": datetime.now().isoformat(),
                "candidate_info": candidate_info,
                "role": role,
                "company": company_name,
                "executive_summary": executive_summary,
                "session_evaluation": session_evaluation,
                "detailed_analysis": detailed_analysis,
                "recommendations": recommendations,
                "individual_evaluations": individual_evaluations
            }
            
            return report
            
        except Exception as e:
            print(f"Error generating report: {e}")
            return self._get_fallback_report(candidate_info, session_evaluation, role)
    
    def _generate_executive_summary(self, candidate_info: Dict[str, Any], 
                                  session_evaluation: Dict[str, Any], 
                                  role: str) -> str:
        """Generate executive summary using LLM."""
        try:
            prompt = f"""
            Generate an executive summary for an interview report.
            
            Candidate Information: {json.dumps(candidate_info, indent=2)}
            Session Evaluation: {json.dumps(session_evaluation, indent=2)}
            Role: {role}
            
            Create a concise executive summary (2-3 paragraphs) that includes:
            1. Overall assessment of the candidate
            2. Key strengths and weaknesses
            3. Final recommendation
            4. Next steps if applicable
            
            Write in a professional tone suitable for HR and hiring managers.
            """
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            return response.text
            
        except Exception as e:
            print(f"Error generating executive summary: {e}")
            return self._get_default_executive_summary(session_evaluation)
    
    def _generate_detailed_analysis(self, individual_evaluations: List[Dict[str, Any]], 
                                  session_evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed analysis of the interview."""
        try:
            # Analyze performance by category
            category_performance = self._analyze_by_category(individual_evaluations)
            
            # Analyze trends
            performance_trends = self._analyze_performance_trends(individual_evaluations)
            
            # Generate detailed insights
            insights_prompt = f"""
            Analyze the interview evaluations and provide detailed insights.
            
            Individual Evaluations: {json.dumps(individual_evaluations, indent=2)}
            Session Evaluation: {json.dumps(session_evaluation, indent=2)}
            
            Provide detailed analysis covering:
            1. Technical competency assessment
            2. Communication skills analysis
            3. Problem-solving approach
            4. Cultural fit indicators
            5. Areas of concern
            6. Potential for growth
            
            Format as structured insights with specific examples from the interview.
            """
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=insights_prompt
            )
            
            return {
                "category_performance": category_performance,
                "performance_trends": performance_trends,
                "detailed_insights": response.text,
                "score_breakdown": self._generate_score_breakdown(individual_evaluations)
            }
            
        except Exception as e:
            print(f"Error generating detailed analysis: {e}")
            return {"error": "Could not generate detailed analysis"}
    
    def _analyze_by_category(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance by question category."""
        category_scores = {}
        category_counts = {}
        
        for eval in evaluations:
            category = eval.get("category", "general")
            score = eval.get("overall_score", 5)
            
            if category not in category_scores:
                category_scores[category] = []
                category_counts[category] = 0
            
            category_scores[category].append(score)
            category_counts[category] += 1
        
        # Calculate averages
        category_averages = {}
        for category, scores in category_scores.items():
            category_averages[category] = {
                "average_score": round(sum(scores) / len(scores), 2),
                "total_questions": category_counts[category],
                "min_score": min(scores),
                "max_score": max(scores)
            }
        
        return category_averages
    
    def _analyze_performance_trends(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends throughout the interview."""
        if len(evaluations) < 2:
            return {"trend": "insufficient_data"}
        
        scores = [eval.get("overall_score", 5) for eval in evaluations]
        
        # Calculate trend
        first_half = scores[:len(scores)//2]
        second_half = scores[len(scores)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        if second_avg > first_avg + 0.5:
            trend = "improving"
        elif first_avg > second_avg + 0.5:
            trend = "declining"
        else:
            trend = "consistent"
        
        return {
            "trend": trend,
            "first_half_average": round(first_avg, 2),
            "second_half_average": round(second_avg, 2),
            "overall_consistency": self._calculate_consistency(scores)
        }
    
    def _calculate_consistency(self, scores: List[float]) -> str:
        """Calculate consistency of scores."""
        if len(scores) < 2:
            return "insufficient_data"
        
        variance = sum((score - sum(scores)/len(scores))**2 for score in scores) / len(scores)
        std_dev = variance ** 0.5
        
        if std_dev < 1.0:
            return "very_consistent"
        elif std_dev < 2.0:
            return "consistent"
        elif std_dev < 3.0:
            return "moderate_variability"
        else:
            return "high_variability"
    
    def _generate_score_breakdown(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate detailed score breakdown."""
        if not evaluations:
            return {}
        
        # Extract LLM evaluation scores
        criteria_scores = {
            "clarity": [],
            "confidence": [],
            "technical_correctness": [],
            "completeness": [],
            "relevance": [],
            "communication_skills": []
        }
        
        for eval in evaluations:
            if "llm_evaluation" in eval:
                for criterion, data in eval["llm_evaluation"].items():
                    if isinstance(data, dict) and "score" in data:
                        if criterion in criteria_scores:
                            criteria_scores[criterion].append(data["score"])
        
        # Calculate averages
        criteria_averages = {}
        for criterion, scores in criteria_scores.items():
            if scores:
                criteria_averages[criterion] = round(sum(scores) / len(scores), 2)
        
        return criteria_averages
    
    def _generate_recommendations(self, session_evaluation: Dict[str, Any], 
                                individual_evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate recommendations based on evaluation."""
        try:
            prompt = f"""
            Generate comprehensive recommendations based on the interview evaluation.
            
            Session Evaluation: {json.dumps(session_evaluation, indent=2)}
            Individual Evaluations: {json.dumps(individual_evaluations, indent=2)}
            
            Provide recommendations for:
            1. Hiring decision (hire, don't hire, or additional evaluation needed)
            2. Next steps in the hiring process
            3. Areas for candidate development if hired
            4. Specific feedback for the candidate
            5. Team fit assessment
            6. Salary/compensation considerations
            
            Be specific and actionable in your recommendations.
            """
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            return {
                "recommendations": response.text,
                "hiring_decision": session_evaluation.get("recommendation", "Additional evaluation needed"),
                "confidence_level": self._calculate_confidence_level(session_evaluation)
            }
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return {
                "recommendations": "Unable to generate recommendations",
                "hiring_decision": session_evaluation.get("recommendation", "Additional evaluation needed"),
                "confidence_level": "medium"
            }
    
    def _calculate_confidence_level(self, session_evaluation: Dict[str, Any]) -> str:
        """Calculate confidence level in the evaluation."""
        score = session_evaluation.get("session_score", 5)
        total_responses = session_evaluation.get("total_responses", 0)
        
        if total_responses < 3:
            return "low"
        elif score >= 8 or score <= 2:
            return "high"
        elif score >= 6 or score <= 4:
            return "medium"
        else:
            return "medium"
    
    def generate_pdf_report(self, report_data: Dict[str, Any], 
                          output_path: str = "interview_report.pdf") -> str:
        """Generate PDF report."""
        try:
            pdf = FPDF()
            pdf.add_page()
            
            # Title
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, f'Interview Report - {report_data["candidate_info"].get("name", "Candidate")}', ln=True)
            pdf.cell(0, 10, f'Role: {report_data["role"]}', ln=True)
            pdf.cell(0, 10, f'Date: {report_data["generated_at"][:10]}', ln=True)
            pdf.ln(10)
            
            # Executive Summary
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Executive Summary', ln=True)
            pdf.set_font('Arial', '', 12)
            summary_lines = report_data["executive_summary"].split('\n')
            for line in summary_lines[:10]:  # Limit to first 10 lines
                pdf.multi_cell(0, 10, line)
            pdf.ln(10)
            
            # Session Evaluation
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Session Evaluation', ln=True)
            pdf.set_font('Arial', '', 12)
            session_eval = report_data["session_evaluation"]
            pdf.cell(0, 10, f'Overall Score: {session_eval.get("session_score", "N/A")}/10', ln=True)
            pdf.cell(0, 10, f'Total Responses: {session_eval.get("total_responses", "N/A")}', ln=True)
            pdf.cell(0, 10, f'Recommendation: {session_eval.get("recommendation", "N/A")}', ln=True)
            pdf.ln(10)
            
            # Recommendations
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Recommendations', ln=True)
            pdf.set_font('Arial', '', 12)
            rec_lines = report_data["recommendations"]["recommendations"].split('\n')
            for line in rec_lines[:15]:  # Limit to first 15 lines
                pdf.multi_cell(0, 10, line)
            
            pdf.output(output_path)
            return output_path
            
        except Exception as e:
            print(f"Error generating PDF: {e}")
            return ""
    
    def send_email_report(self, report_data: Dict[str, Any], 
                         recipient_email: str, 
                         sender_email: str = None,
                         sender_password: str = None) -> bool:
        """Send report via email."""
        try:
            # Generate PDF
            pdf_path = self.generate_pdf_report(report_data)
            if not pdf_path:
                return False
            
            # Email configuration
            if not sender_email or not sender_password:
                print("Email credentials not provided")
                return False
            
            # Create email
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = recipient_email
            msg['Subject'] = f"Interview Report - {report_data['candidate_info'].get('name', 'Candidate')}"
            
            # Email body
            body = f"""
            Interview Report for {report_data['candidate_info'].get('name', 'Candidate')}
            
            Role: {report_data['role']}
            Overall Score: {report_data['session_evaluation'].get('session_score', 'N/A')}/10
            Recommendation: {report_data['session_evaluation'].get('recommendation', 'N/A')}
            
            Please find the detailed report attached.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach PDF
            with open(pdf_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {os.path.basename(pdf_path)}'
            )
            msg.attach(part)
            
            # Send email
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, sender_password)
            text = msg.as_string()
            server.sendmail(sender_email, recipient_email, text)
            server.quit()
            
            # Clean up PDF
            os.remove(pdf_path)
            
            return True
            
        except Exception as e:
            print(f"Error sending email: {e}")
            return False
    
    def _get_fallback_report(self, candidate_info: Dict[str, Any], 
                           session_evaluation: Dict[str, Any], 
                           role: str) -> Dict[str, Any]:
        """Fallback report if generation fails."""
        return {
            "report_id": f"INT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "candidate_info": candidate_info,
            "role": role,
            "company": "Unknown",
            "executive_summary": self._get_default_executive_summary(session_evaluation),
            "session_evaluation": session_evaluation,
            "detailed_analysis": {"error": "Could not generate detailed analysis"},
            "recommendations": {
                "recommendations": "Unable to generate recommendations",
                "hiring_decision": session_evaluation.get("recommendation", "Additional evaluation needed"),
                "confidence_level": "medium"
            },
            "individual_evaluations": []
        }
    
    def _get_default_executive_summary(self, session_evaluation: Dict[str, Any]) -> str:
        """Default executive summary if generation fails."""
        score = session_evaluation.get("session_score", 5)
        
        if score >= 8:
            return "The candidate demonstrated excellent performance throughout the interview with strong technical skills and communication abilities. Recommend for next round."
        elif score >= 6:
            return "The candidate showed good potential with some areas for improvement. Consider for next round with reservations."
        elif score >= 4:
            return "The candidate's performance was adequate but needs significant improvement in several areas. Additional evaluation may be needed."
        else:
            return "The candidate's performance was below expectations. Not recommended for next round."


if __name__ == "__main__":
    # Test the report generator
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        generator = ReportGenerator(api_key)
        
        # Test data
        candidate_info = {"name": "John Doe", "email": "john@example.com"}
        session_evaluation = {
            "session_score": 7.5,
            "total_responses": 5,
            "recommendation": "Good candidate - consider for next round"
        }
        individual_evaluations = []
        
        report = generator.generate_interview_report(
            candidate_info, session_evaluation, individual_evaluations, 
            "Data Scientist", "TechCorp"
        )
        print("Report generated successfully!")
        print(f"Report ID: {report['report_id']}")
    else:
        print("Please set GOOGLE_API_KEY environment variable") 