import os 
import sys 

src_path = os.path.dirname(__file__)
sys.path.append(src_path)

from flask import Flask, request, jsonify
import uuid

from generation import create_rag
from db import (
    save_conversation, 
    save_feedback
)

app = Flask(__name__)
rag = create_rag()

@app.route('/question', methods=['POST'])
def handle_question():
    data = request.json
    question = data.get('question')
    
    if not question:
        return jsonify({"error", "No question provided"}), 400
    
    # Generate aconversation ID
    conversation_id = str(uuid.uuid4())
    
    # call the RAG function
    answer_data = rag.answer(question)
    
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
    data = request.json
    conversation_id = data.get("conversation_id")
    feedback = data.get("feedback")
    
    if not conversation_id or feedback not in [1, -1]:
        return jsonify({"error": "Invalid input"}), 400
    
    save_feedback(
        conversation_id=conversation_id,
        feedback=feedback
    )
    
    return jsonify(
        {"mesagge": f"Feedback received for conversation {conversation_id}:{feedback}"}
    )
    
if __name__=="__main__":
    app.run(debug=True)