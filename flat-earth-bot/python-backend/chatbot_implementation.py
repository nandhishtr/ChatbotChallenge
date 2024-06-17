from chatbot import Chatbot
import threading
from textblob import TextBlob
import prompts

lock = threading.Lock()


class ChatbotImplementation(Chatbot):

    def __init__(self):
        Chatbot.__init__(self)
        self.succesful_sessions = []
        self.session_states = {}

    def initialize_session(self, session_id):
        if session_id not in self.session_states:
            self.session_states[session_id] = {
                "curiosity_about_flat_earth":False,
                "user_asks_for_clearer_explanation":False,
                "disagree_flat_earth": False,
                "provided_evidence_against_flat_earth": False,
                "provided_evidence_for_spherical_earth": False,
                "user_identified_argumentation_strategy": False,
            }

    def get_sentiment_analysis_prompt(self, text):

        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0.2:
            sentiment = "positive"
            sentiment_prompt = prompts.positive_sentiment_prompt
        elif analysis.sentiment.polarity < 0:
            sentiment = "negative"
            sentiment_prompt = prompts.negative_sentiment_prompt
        else:
            sentiment = "neutral"
            sentiment_prompt = prompts.neutral_sentiment_prompt
        return sentiment_prompt, sentiment

    def get_intent_prompt(self, intent):
        intent_name = intent["name"]
        if intent_name == "greeting" or "user_shows_appreciation":
            intent_prompt = prompts.greet_intent_prompt
        elif intent_name == "out_of_scope" or "user_asks_personal_questions" or "user_tries_to_change_the_topic":
            intent_prompt = prompts.out_of_scope_prompt
        elif intent_name == "insult_and_abuse_bot" or "user_makes_fun_of_bot":
            intent_prompt = prompts.insult_and_fun_intent_prompt
        elif intent_name == "curiosity_about_flat_earth" or "user_asks_for_clearer_explanation" or "user_acknowledges_or_agrees_with_flat_earth_beliefs":
            intent_prompt = prompts.curiosity_intent_prompt
        elif intent_name == "disagree_flat_earth":
            intent_prompt = prompts.argumentation_intent_prompt
        elif intent_name == "provided_evidence_against_flat_earth":
            intent_prompt = prompts.provided_evidence_against_flat_earth
        elif intent_name == "provided_evidence_for_spherical_earth":
            intent_prompt = prompts.provided_evidence_for_spherical_earth
        else:
            intent_prompt = prompts.argumentation_intent_prompt
        return intent_prompt, intent_name

    # def update_session_state(self, intent, session_id):
    #     if session_id in self.session_states:
    #         state = self.session_states[session_id]
    #         if intent["name"] == "provided_evidence_against_flat_earth":
    #             state["provided_evidence_against_flat_earth"] = True
    #         elif intent["name"] == "provided_evidence_for_spherical_earth":
    #             state["provided_evidence_for_spherical_earth"] = True
    #         elif intent["name"] == "user_identified_argumentation_strategy":
    #             state["user_identified_argumentation_strategy"] = True
    #     print("********************************")
    #     print("Current state : ", state)
    #     print("********************************")
    def update_session_state(self, intent, session_id):
        if session_id in self.session_states:
            state = self.session_states[session_id]
            if intent["name"] == "provided_evidence_against_flat_earth":
                state["provided_evidence_against_flat_earth"] = True
            elif intent["name"] == "provided_evidence_for_spherical_earth":
                state["provided_evidence_for_spherical_earth"] = True
            elif intent["name"] == "curiosity_about_flat_earth":
                state["curiosity_about_flat_earth"] = True
            elif intent["name"] == "user_asks_for_clearer_explanation":
                state["user_asks_for_clearer_explanation"] = True
            elif intent["name"] == "disagree_flat_earth":
                state["disagree_flat_earth"] = True
            elif intent["name"] == "user_identified_argumentation_strategy":
                if state["provided_evidence_against_flat_earth"] and state["provided_evidence_for_spherical_earth"] and \
                        state["curiosity_about_flat_earth"] and state["user_asks_for_clearer_explanation"] and state[
                    "disagree_flat_earth"]:
                    state["user_identified_argumentation_strategy"] = True
                else:
                    state["user_identified_argumentation_strategy"] = False

        print("********************************")
        print("Current state : ", state)
        print("********************************")

    def is_session_successful(self, session_id):
        if session_id in self.session_states:
            state = self.session_states[session_id]
            return all(state.values())
        return False

    def get_prompt(self, messages, intent, session_id):
        self.initialize_session(session_id)
        print("*********************")
        print("intent", intent)
        latest_user_ip = messages[-1]["message"]
        sentiment_prompt, sentiment = self.get_sentiment_analysis_prompt(latest_user_ip)
        intent_prompt, intent_name = self.get_intent_prompt(intent)
        print("*********************")
        print(f"Sentiment: {sentiment} , Intent: {intent_name} ", "latest input :", latest_user_ip)
        print("*********************")
        self.update_session_state(intent, session_id)
        session_is_successful = self.is_session_successful(session_id)

        if session_is_successful:
            # generate the prompt that the user succeeded
            prompt = prompts.prompt_template_success.format(
                user_message=latest_user_ip)  # + self.build_dialog(messages)
            print("Prompt to llm:", prompt)

        else:

            prompt = prompts.prompt_template_persona + sentiment_prompt + intent_prompt + self.build_dialog(messages)
            # print("*****************************************************")
            # print("prompt template is :", prompt_template)
            print("*****************************************************")
            print("Prompt to llm:", prompt)
            print("*****************************************************")

        return prompt, session_is_successful
