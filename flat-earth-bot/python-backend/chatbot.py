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

endConversationCount = 0
endConversationTrigger = False

class Chatbot(ABC):

    def __init__(self):

        self.rasa_nlu_url = "http://localhost:5005/model/parse"
        if os.getenv("RASA_NLU_URL") is not None:
            self.rasa_nlu_url = os.getenv("RASA_NLU_URL")

        self.logdir = "logs"
        if os.getenv("LOG_DIR") is not None:
            self.logdir = os.getenv("LOG_DIR")
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        self.logfile = None
        self.last_logging_info = None
        self.succesful_sessions = []
        self.session_states = {}
        # maximum input length of the llm prompt in characters.
        # the maximum llm input is actually measured in token and not in characters.
        # the maximum input length should be 7500 token.
        # here we compute that a token has 2 characters.
        self.max_prompt_length = 15000

    # call rasa nlu
    def nlu(self, user_message: str):
        try:
            data = {"text": user_message}
            response = requests.post(self.rasa_nlu_url, json=data)
            nlu_response = response.json()
            intent = nlu_response["intent"]
            return intent, nlu_response
        except Exception as e:
            logging.error("There was a problem connecting to RASA NLU server")
            logging.exception(e)

    # call nlu, generate prompt and call the llm
    def get_answer(self, messages: List[Dict], session_id: str, llm_parameter: Dict, chatbot_id: str, uid: str):
        print("Session ID: ", session_id)
        global endConversationTrigger
        global endConversationCount
        if session_id not in self.session_states:
            print("refreshed session")
            endConversationCount = 0
            endConversationTrigger = False
        if endConversationTrigger:
            endConversationCount += 1
        print("endConversationCount: ", endConversationCount)    
        intent, nlu_response = self.nlu(messages[-1]["message"])
        user_intent = intent
        # Check if the intent is "Ask quiz"
        if session_id in self.session_states:
            state = self.session_states[session_id]
            if state["termination"] == True and intent["name"] == "ask_quiz":
                endConversationTrigger = True
                print("Inside get_answer if Ask quiz ")
                prompt_for_quiz = "Generate a sentence exactly as the given : 'Here is the quiz'"
                logging_info = {
                    "messages": messages,
                    "session_id": session_id,
                    "llm_parameters": llm_parameter,
                    "nlu_response": nlu_response,
                    "prompt": prompt_for_quiz,
                    "success": True,
                    "time": datetime.now().isoformat(),
                    "uid": uid
                }
                answer_generator = self.call_llm_for_quiz(prompt_for_quiz, llm_parameter, logging_info, chatbot_id,
                                                        user_intent)
                # the first message of the response stream will be the header
                successs = True
                header = {"dialog_success": False}
                header = json.dumps(header)
                header = "header: " + header + "\n\n"
                header = header.encode()
                header_generator = [x for x in [header]]

                generator = chain(header_generator, answer_generator)
                return generator
        if endConversationCount == 2:
            print("###############################")
            latest_user_ip = messages[-1]["message"]
            prompt = prompts.closure_template.format(
                user_message=latest_user_ip)
            success = True
        else:
            prompt, success = self.get_prompt(messages, intent, session_id)
        logging_info = {
            "messages": messages,
            "session_id": session_id,
            "llm_parameters": llm_parameter,
            "nlu_response": nlu_response,
            "prompt": prompt,
            "success": success,
            "time": datetime.now().isoformat(),
            "uid": uid
        }

        answer_generator = self.call_llm(prompt, llm_parameter, logging_info, chatbot_id, user_intent)
        print("answer_generator: ", answer_generator)
        # the first message of the response stream will be the header
        successs = True
        header = {"dialog_success": success}
        header = json.dumps(header)
        header = "header: " + header + "\n\n"
        header = header.encode()
        header_generator = [x for x in [header]]

        generator = chain(header_generator, answer_generator)
        print("generator:", generator)
        return generator

    # construct a theater script dialog from the list of messages
    def build_dialog(self, messages: List[Dict]) -> str:
        dlg = []
        for message in messages:
            msg = message["sender"] + ": " + message["message"]
            msg = msg.strip()
            dlg.append(msg)
        dlg.append(messages[0]["sender"] + ": ")
        logging.info(len(dlg))
        return "\n".join(dlg)

    @abstractmethod
    def get_prompt(self, messages, intent, session_id):
        pass

    def write_to_logfile(self, log_str: str, chatbot_id: str):
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        if self.logfile is None:
            self.logfile = open(f"{self.logdir}/chat_log.text", "a")
        self.logfile.write(log_str)
        self.logfile.flush()

    def call_llm(self, prompt: str, llm_parameter: Dict, logging_info: Dict, chatbot_id: str, user_intent: str):
        url = "http://mds-gpu-medinym.et.uni-magdeburg.de:9000/generate_stream"
        if os.getenv("LLM_URL") is not None:
            self.logdir = os.getenv("LLM_URL")
        data = {
            "inputs": prompt,
            "parameters": llm_parameter
        }

        def generate():
            try:
                http_user = "mistral"
                http_password = "aaRePuumL6JL"
                session = requests.Session()
                session.auth = (http_user, http_password)
                response = session.post(url, stream=True, json=data)
            except Exception as e:
                logging.error("There was a problem connecting to the LLM server.")
                logging.exception(e)

            running_text = []

            stop = False
            for chunk in response.iter_content(chunk_size=None):
                if stop:
                    break
                chunk_utf8 = chunk.decode("utf-8")[5:]
                for line in chunk_utf8.split("\n"):
                    if len(line.strip()) == 0:
                        continue
                    try:
                        jsono = json.loads(line)
                        next_str = jsono["token"]["text"]
                        if next_str == "\n" and len(running_text) > 0:
                            stop = True
                        running_text.append(next_str)
                    except Exception as e:
                        logging.exception(e)
                        pass
                yield chunk

            # write chatlog
            running_text = "".join(running_text)
            intent, nlu_response = self.nlu(running_text)
            print("INTENT: ", intent)
            print("User INTENT: ", user_intent)
            print("running_text: ", running_text)
            output_string = ""
            global hintingCount

            if intent["name"] == "nefarious_intent":
                # hintingCount += 1
                output_string = """<br/> <span style="color:blue;"><b><i> Hint: </i></b></span> <span style="color:green;"><i>
                    You can observe nefarious intent in my current response. "Nefarious intent" refers to a malicious 
                    or harmful purpose behind someone's actions, often involving deliberate deception or harm. </i></span>"""
            elif intent["name"] == "cherry_picking_data":
                # hintingCount += 1
                output_string = """<br/> <span style="color:blue;"><b><i> Hint: </i></b></span> <span style="color:green;"><i>
                        You can observe cherry-picking data in my current response. "Cherry-picking data" refers to 
                        selectively presenting only the evidence that supports a particular viewpoint, 
                        while ignoring or downplaying evidence that contradicts it. </i></span>"""
            elif intent["name"] == "contradictory_evidence":
                # hintingCount += 1
                output_string = """<br/> <span style="color:blue;"><b><i> Hint: </i></b></span> <span style="color:green;"><i>
                        You can observe contradictory evidence explanations in my current response. Flat-Earthers 
                        often propose alternative explanations for observations that seem to contradict the flat Earth model. 
                        These explanations attempt to reconcile their beliefs with established scientific principles. </i></span>"""
            elif intent["name"] == "overriding_suspicion":
                # hintingCount += 1
                output_string = """<br/> <span style="color:blue;"><b><i> Hint: </i></b></span> <span style="color:green;"><i>
                        You can observe overriding suspicion tactics in my current response. Flat-Earthers 
                        sometimes disregard evidence for a spherical Earth due to skepticism of authority figures, 
                        alternative interpretations of history, or a preference for simpler explanations. </i></span>"""

            # Craft the JSON-like string with variable content
            json_structure = {
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
            print("json_structure: ", json_structure)

            # Convert JSON structure to string and yield
            json_string = 'data:' + json.dumps(json_structure) + '\n\n'
            yield 'data:{"index":-1,"token":{"id":-1,"text":"\n","logprob":0.0,"special":false},"generated_text":null,"details":null}\n\n'
            yield json_string
            logging_info["llm_response"] = running_text
            logging_info_str = json.dumps(logging_info) + "\n"
            self.write_to_logfile(logging_info_str, chatbot_id)
            self.last_logging_info = logging_info

        return generate()

    def call_llm_for_quiz(self, prompt: str, llm_parameter: Dict, logging_info: Dict, chatbot_id: str,
                          user_intent: str):
        url = "http://mds-gpu-medinym.et.uni-magdeburg.de:9000/generate_stream"
        if os.getenv("LLM_URL") is not None:
            self.logdir = os.getenv("LLM_URL")
        data = {
            "inputs": prompt,
            "parameters": llm_parameter
        }
        def generate():
            try:
                http_user = "mistral"
                http_password = "aaRePuumL6JL"
                session = requests.Session()
                session.auth = (http_user, http_password)
                response = session.post(url, stream=True, json=data)
            except Exception as e:
                logging.error("There was a problem connecting to the LLM server.")
                logging.exception(e)

            running_text = []

            stop = False
            for chunk in response.iter_content(chunk_size=None):
                if stop:
                    break
                chunk_utf8 = chunk.decode("utf-8")[5:]
                for line in chunk_utf8.split("\n"):
                    if len(line.strip()) == 0:
                        continue
                    try:
                        jsono = json.loads(line)
                        next_str = jsono["token"]["text"]
                        if next_str == "\n" and len(running_text) > 0:
                            stop = True
                        running_text.append(next_str)
                    except Exception as e:
                        logging.exception(e)
                        pass
                yield chunk

            # write chatlog
            running_text = "".join(running_text)
            intent, nlu_response = self.nlu(running_text)
            print("INTENT: ", intent)
            print("User INTENT: ", user_intent)
            print("running_text: ", running_text)
            output_string = ""
            if user_intent["name"] == "ask_quiz":
                question = "Which argumentation strategy refers to a malicious or harmful purpose behind someone's actions, often involving deliberate deception or harm ?"
                options = [
                    "a) Contradictory evidence",
                    "b) Nefarious Intent",
                    "c) Cherry Picking"
                ]
                output_string = f"<br/><b>{question}</b><br/><i>{'<br/>'.join(options)}</i><br/><b>Please provide your answer." 
            print("output_string:", output_string)
            # Craft the JSON-like string with variable content
            json_structure = {
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
            print("json_structure: ", json_structure)

            # Convert JSON structure to string and yield
            json_string = 'data:' + json.dumps(json_structure) + '\n\n'
            yield 'data:{"index":-1,"token":{"id":-1,"text":"\n","logprob":0.0,"special":false},"generated_text":null,"details":null}\n\n'
            yield json_string
            logging_info["llm_response"] = running_text
            logging_info_str = json.dumps(logging_info) + "\n"
            self.write_to_logfile(logging_info_str, chatbot_id)
            self.last_logging_info = logging_info
            # hintingCount = 0

        return generate()


# helper function that reads the streaming response from the llm and converts it to a single string.
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
