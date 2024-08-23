from flask import Flask, request
import requests
import json
import logging
from flask_cors import CORS, cross_origin
from abc import ABC, abstractmethod
import os
from typing import Dict
from datetime import datetime

from chatbot_implementation import ChatbotImplementation
from chatbot import llm_stream_to_str

def init_logging():
    log_dir = "logs/"
    if os.getenv("LOG_DIR") is not None:
        log_dir = os.getenv("LOG_DIR")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logfile = os.path.join(log_dir, "technical_log.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler()
        ]
    )

init_logging()

app = Flask(__name__)
CORS(app)

chatbot = ChatbotImplementation()

def get_generator():
    request_data = request.get_json()
    generator = chatbot.get_answer(
        request_data["messages"],
        request_data["session_id"],
        request_data["llm_parameters"],
        request_data["chatbot"],
        request.args.get('uid')
    )
    return generator

@app.route("/api/chat", methods=['POST'])
@cross_origin()
def chat():
    generator = get_generator()
    return app.response_class(generator, mimetype='application/json')

@app.route("/api/chat_no_stream", methods=['POST'])
@cross_origin()
def chat_no_stream():

    generator = get_generator()
    response = llm_stream_to_str(generator)
    return response

port = int(os.environ.get('PORT', 5000))
production = int(os.environ.get('PRODUCTION', 0)) == 1
app.run(debug=not production, host='0.0.0.0', port=port)
