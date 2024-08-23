import random
from flask import Flask, request
import requests
import json
import logging
from abc import ABC, abstractmethod
import os
from typing import Dict, List
from datetime import datetime
from itertools import chain
import prompts

class Chatbot(ABC):
    def __init__(self):
        self.rasa_nlu_url = os.getenv("RASA_NLU_URL", "http://localhost:5005/model/parse")
        self.logdir = os.getenv("LOG_DIR", "logs")
        os.makedirs(self.logdir, exist_ok=True)
        self.logfile = None
        self.last_logging_info = None
        self.successful_sessions = []
        self.session_states = {}
        self.max_prompt_length = 15000
        self.llm_url = os.getenv("LLM_URL", "http://mds-gpu-medinym.et.uni-magdeburg.de:9000/generate_stream")
        self.hinting = 0
        self.hint_count = 0
        self.correct_answer = []

    def build_dialog(self, messages: List[Dict]) -> str:
        dlg = []
        for message in messages:
            msg = f"{message['sender']}: {message['message'].strip()}"
            dlg.append(msg)
        dlg.append(f"{messages[0]['sender']}: ")
        logging.info(len(dlg))
        return "\n".join(dlg)
        
    @abstractmethod
    def get_prompt(self, messages, intent, session_id):
        pass

    def nlu(self, user_message: str):
        try:
            response = requests.post(self.rasa_nlu_url, json={"text": user_message})
            nlu_response = response.json()
            return nlu_response["intent"], nlu_response
        except Exception as e:
            logging.error("There was a problem connecting to RASA NLU server")
            logging.exception(e)

    def get_answer(self, messages: List[Dict], session_id: str, llm_parameter: Dict, chatbot_id: str, uid: str):
        if session_id not in self.session_states:
            self.reset_session()

        intent, nlu_response = self.nlu(messages[-1]["message"])
        user_intent = intent

        if self.is_quiz_answer(messages[-1]["message"]):
            return self.handle_quiz_answer(messages, session_id, llm_parameter, nlu_response, chatbot_id, uid)
        elif user_intent["name"] == "ask_quiz" or (self.hinting == 2 and self.hint_count < 4):
            return self.handle_quiz_request(messages, session_id, llm_parameter, nlu_response, chatbot_id, uid)
        elif self.hint_count == 4:
            prompt = prompts.closure_template.format(user_message=messages[-1]["message"])
            success = True
        else:
            prompt, success = self.get_prompt(messages, intent, session_id)

        return self.generate_response(prompt, success, messages, session_id, llm_parameter, nlu_response, chatbot_id, uid, user_intent)

    def reset_session(self):
        self.hinting = 0
        self.hint_count = 0
        self.correct_answer = []

    def is_quiz_answer(self, message):
        return message in ["a", "b", "c"]

    def handle_quiz_answer(self, messages, session_id, llm_parameter, nlu_response, chatbot_id, uid):
        is_correct = self.correct_answer[-1].split(')')[0] == messages[-1]["message"]
        prompt = self.get_quiz_answer_prompt(is_correct)
        return self.generate_quiz_response(prompt, messages, session_id, llm_parameter, nlu_response, chatbot_id, uid)

    def handle_quiz_request(self, messages, session_id, llm_parameter, nlu_response, chatbot_id, uid):
        prompt = "Generate a sentence exactly as the given : 'Hold on a second! Let's spice things up with a quick quiz to test your skills in spotting the argumentation strategies that flat earth believers use. Ready to challenge your understanding? \n\n'"
        return self.generate_quiz_response(prompt, messages, session_id, llm_parameter, nlu_response, chatbot_id, uid)

    def get_quiz_answer_prompt(self, is_correct):
        if is_correct:
            return f"Generate a sentence exactly as the given and add emojies as required : 'Hurray thats right!!! the correct answer is {self.correct_answer[-1]}  and generate a sentence breifly describing what is {self.correct_answer[-1]}then steer the conversation back to flat earth \n\n' "
        else:
            return f"Generate a sentence exactly as the given and put approprite emojies : 'Sorry Your answer is wrong  Correct answer is {self.correct_answer[-1]} and generate a sentence breifly describing the answer in the context of flat earth, then steer the conversation back to flat earth\n\n'"

    def generate_response(self, prompt, success, messages, session_id, llm_parameter, nlu_response, chatbot_id, uid, user_intent):
        logging_info = self.create_logging_info(messages, session_id, llm_parameter, nlu_response, prompt, success, uid)
        answer_generator = self.call_llm(prompt, llm_parameter, logging_info, chatbot_id, user_intent)
        return self.create_generator_chain(success, answer_generator)

    def generate_quiz_response(self, prompt, messages, session_id, llm_parameter, nlu_response, chatbot_id, uid):
        logging_info = self.create_logging_info(messages, session_id, llm_parameter, nlu_response, prompt, True, uid)
        answer_generator = self.call_llm_for_quiz(prompt, llm_parameter, logging_info, chatbot_id, None)
        return self.create_generator_chain(False, answer_generator)

    def create_logging_info(self, messages, session_id, llm_parameter, nlu_response, prompt, success, uid):
        return {
            "messages": messages,
            "session_id": session_id,
            "llm_parameters": llm_parameter,
            "nlu_response": nlu_response,
            "prompt": prompt,
            "success": success,
            "time": datetime.now().isoformat(),
            "uid": uid
        }

    def create_generator_chain(self, success, answer_generator):
        header = {"dialog_success": success}
        header = json.dumps(header)
        header = "header: " + header + "\n\n"
        header = header.encode()
        header_generator = [x for x in [header]]
        return chain(header_generator, answer_generator)

    def write_to_logfile(self, log_str: str, chatbot_id: str):
        if self.logfile is None:
            self.logfile = open(f"{self.logdir}/chat_log.text", "a")
        self.logfile.write(log_str)
        self.logfile.flush()

    def call_llm(self, prompt: str, llm_parameter: Dict, logging_info: Dict, chatbot_id: str, user_intent: str):
        return self._call_llm_generic(prompt, llm_parameter, logging_info, chatbot_id, user_intent, is_quiz=False)

    def call_llm_for_quiz(self, prompt: str, llm_parameter: Dict, logging_info: Dict, chatbot_id: str, user_intent: str):
        return self._call_llm_generic(prompt, llm_parameter, logging_info, chatbot_id, user_intent, is_quiz=True)

    def _call_llm_generic(self, prompt: str, llm_parameter: Dict, logging_info: Dict, chatbot_id: str, user_intent: str, is_quiz: bool):
        data = {
            "inputs": prompt,
            "parameters": llm_parameter
        }

        def generate():
            try:
                session = requests.Session()
                response = session.post(self.llm_url, stream=True, json=data)
            except Exception as e:
                logging.error("There was a problem connecting to the LLM server.")
                logging.exception(e)

            running_text = []

            for chunk in response.iter_content(chunk_size=None):
                chunk_utf8 = chunk.decode("utf-8")[5:]
                for line in chunk_utf8.split("\n"):
                    if len(line.strip()) == 0:
                        continue
                    try:
                        jsono = json.loads(line)
                        next_str = jsono["token"]["text"]
                        running_text.append(next_str)
                    except Exception as e:
                        logging.exception(e)
                        pass
                yield chunk

            running_text = "".join(running_text)
            intent, _ = self.nlu(running_text)
            output_string = self.generate_output_string(intent, is_quiz, user_intent)

            json_structure = self.create_json_structure(output_string)
            json_string = 'data:' + json.dumps(json_structure) + '\n\n'
            yield 'data:{"index":-1,"token":{"id":-1,"text":"\n","logprob":0.0,"special":false},"generated_text":null,"details":null}\n\n'
            yield json_string

            logging_info["llm_response"] = running_text
            logging_info_str = json.dumps(logging_info) + "\n"
            self.write_to_logfile(logging_info_str, chatbot_id)
            self.last_logging_info = logging_info

        return generate()

    def generate_output_string(self, intent, is_quiz, user_intent):
        if is_quiz:
            return self.generate_quiz_output()
        else:
            return self.generate_hint_output(intent)

    def generate_hint_output(self, intent):
        hint_messages = {
            "nefarious_intent": """You can observe nefarious intent in my current response. "Nefarious intent" refers to a malicious or harmful purpose behind someone's actions, often involving deliberate deception or harm.""",
            "cherry_picking_data": """You can observe cherry-picking data in my current response. "Cherry-picking data" refers to selectively presenting only the evidence that supports a particular viewpoint, while ignoring or downplaying evidence that contradicts it.""",
            "contradictory_evidence": "You can observe contradictory evidence explanations in my current response. Flat-Earthers often propose alternative explanations for observations that seem to contradict the flat Earth model. These explanations attempt to reconcile their beliefs with established scientific principles.",
            "overriding_suspicion": "You can observe overriding suspicion tactics in my current response. Flat-Earthers sometimes disregard evidence for a spherical Earth due to skepticism of authority figures, alternative interpretations of history, or a preference for simpler explanations."
        }
        
        if intent["name"] in hint_messages:
            self.hinting += 1
            return f"""<br/> <span style="color:blue;"><b><i> Hint: </i></b></span> <span style="color:green;"><i>{hint_messages[intent["name"]]}</i></span>"""
        return ""

    def generate_quiz_output(self):
        questions = [
            {
                "question": "What term describes the strategy of highlighting only the evidence that supports one's argument while dismissing or ignoring any evidence that contradicts it?",
                "options": ["a) Nefarious Intent", "b) Contradictory Evidence", "c) Cherry Picking"],
                "answer": "c) Cherry Picking"
            },
            {
                "question": "'Different constellations being visible could be explained by a complex light refraction phenomenon in the flat Earth model.' What type of argumentation strategy is being used here?",
                "options": ["a) Cherry Picking", "b) Contradictory Evidence", "c) Nefarious Intent"],
                "answer": "b) Contradictory Evidence"
            },
            {
                "question": "Which argumentation strategy involves having a harmful or deceitful purpose behind one's actions, often involving deliberate misinformation or manipulation?",
                "options": ["a) Contradictory Evidence", "b) Nefarious Intent", "c) Cherry Picking"],
                "answer": "b) Nefarious Intent"
            },
            {
                "question": "'Why haven't any photos or videos been captured from space that definitively show the Earth's curvature?' Which argumentation strategy does this question represent?",
                "options": ["a) Overriding Suspicion", "b) Contradictory Evidence", "c) Cherry Picking"],
                "answer": "a) Overriding Suspicion"
            }
        ]

        self.hinting = 0
        selected_question = questions[self.hint_count]
        self.correct_answer.append(selected_question['answer'])
        self.hint_count += 1

        return f"<br/><b>{selected_question['question']}</b><br/><i>{'<br/>'.join(selected_question['options'])}</i><br/><b>Provide answer to the quiz by typing options a,b or c. </b><br/>"

    def create_json_structure(self, output_string):
        return {
            "index": -1,
            "token": {
                "id": -1,
                "text": output_string,
                "logprob": 0.0,
                "special": False
            },
            "generated_text": None,
            "details": None
        }

    @abstractmethod
    def get_prompt(self, messages, intent, session_id):
        pass

def llm_stream_to_str(generator):
    response = []
    for o in generator:
        s = o.decode("utf-8")[5:]
        try:
            jsono = json.loads(s)
            response.append(jsono["token"]["text"])
        except Exception as e:
            pass
    return "".join(response)