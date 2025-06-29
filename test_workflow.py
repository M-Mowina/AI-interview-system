#!/usr/bin/env python3
"""
Test script for the AI Interview System workflow
"""

import os
from dotenv import load_dotenv
from src.interview_workflow_simple import workflow

# Load environment variables
load_dotenv()

def test_workflow():
    """Test the complete workflow"""
    print("🤖 Testing AI Interview System Workflow")
    print("=" * 50)
    
    # Test 1: Generate Persona
    print("\n1. Testing Persona Generation...")
    try:
        persona = workflow.generate_persona()
        print(f"✅ Persona generated successfully!")
        print(f"   Preview: {persona[:100]}...")
    except Exception as e:
        print(f"❌ Persona generation failed: {e}")
        return
    
    # Test 2: Get Questions
    print("\n2. Testing Question Loading...")
    try:
        questions = workflow.get_questions()
        print(f"✅ {len(questions)} questions loaded successfully!")
        for i, q in enumerate(questions[:3], 1):
            print(f"   Q{i}: {q[:80]}...")
    except Exception as e:
        print(f"❌ Question loading failed: {e}")
        return
    
    # Test 3: Ask Question
    print("\n3. Testing Question Asking...")
    try:
        question = workflow.ask_question(0)
        print(f"✅ Question retrieved: {question[:80]}...")
    except Exception as e:
        print(f"❌ Question asking failed: {e}")
        return
    
    # Test 4: Evaluate Response
    print("\n4. Testing Response Evaluation...")
    try:
        test_response = "I have experience with machine learning projects including classification and regression tasks. I've worked with Python, scikit-learn, and TensorFlow."
        eval_result = workflow.evaluate_response(test_response)
        print(f"✅ Response evaluated successfully!")
        print(f"   Score: {eval_result['overall_score']}/10")
        print(f"   Feedback: {eval_result['feedback'][:100]}...")
    except Exception as e:
        print(f"❌ Response evaluation failed: {e}")
        return
    
    # Test 5: Generate Report
    print("\n5. Testing Report Generation...")
    try:
        report = workflow.generate_report()
        print(f"✅ Report generated successfully!")
        print(f"   Report ID: {report['report_id']}")
        
        # Get session score from session evaluation
        session_score = report['session_evaluation']['session_score']
        print(f"   Session Score: {session_score}/10")
        print(f"   Executive Summary: {report['executive_summary'][:100]}...")
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! The workflow is working correctly.")
    print("\nYou can now run the Streamlit app with:")
    print("streamlit run streamlit_app.py")

if __name__ == "__main__":
    test_workflow() 