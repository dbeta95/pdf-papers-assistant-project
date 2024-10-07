
"""
This module contains the main Flask application that serves the API endpoints for the chatbot.
"""
import os 
import sys 
import uuid

from flask import Flask, request, jsonify
from flask_cors import CORS
from marshmallow import Schema, fields, ValidationError

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

from src.initialize import create_rag
from src.db import (
    save_conversation, 
    save_feedback
)

class QuestionSchema(Schema):
    question = fields.Str(required=True)
    category = fields.Str(required=False)
    
class FeedbackSchema(Schema):
    conversation_id = fields.Str(required=True)
    feedback = fields.Int(required=True)

app = Flask(__name__)
CORS(app)
rag = create_rag()

@app.route('/question', methods=['POST'])
def handle_question():    
    question_schema = QuestionSchema()
    try: 
        data = question_schema.load(request.get_json())
        question = data.get('question')
        category = data.get('category')
    except ValidationError as err:
        return jsonify(err.messages), 400
        
    if category:
        filter_dict = {"category": category}
        rag.update_parameters(filter_dict=filter_dict)
    else:
        rag.update_parameters(filter_dict={})
    
    # Generate aconversation ID
    conversation_id = str(uuid.uuid4())
    
    # call the RAG function
    answer_data = rag.answer(
        query=question, 
        search="elasticsearch"    
    )
    
    result = {
        "conversation_id": conversation_id,
        "question": question,
        "answer": answer_data['answer'],
             
    }
       
    save_conversation(
        conversation_id=conversation_id,
        question=question,
        answer_data=answer_data        
    )
    
    return jsonify(result)
    
@app.route("/feedback", methods = ['POST'])
def handle_feedback():
    feedback_schema = FeedbackSchema()
    try: 
        data = feedback_schema.load(request.get_json())
        conversation_id = data.get("conversation_id")
        feedback = data.get("feedback")
    except ValidationError as err:
        return jsonify(err.messages), 400   
        
    if feedback not in [1, -1]:
        return jsonify({"error": "Invalid input"}), 400
    
    save_feedback(
        conversation_id=conversation_id,
        feedback=feedback
    )
    
    return jsonify(
        {"mesagge": f"Feedback received for conversation {conversation_id}:{feedback}"}
    )
    
if __name__=="__main__":
    app.run(
        debug=True,
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=os.getenv("APP_PORT", 5000)
    )