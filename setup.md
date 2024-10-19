# Chatbots to Strengthen Democracy

  * [Overview](#overview)
  * [Starting up the software on your local computer](#starting-up-the-software-on-your-local-computer)
  * [Develop a bot](#develop-a-bot)
    + [Overview](#overview-1)
    + [Local development setup](#local-development-setup)
      - [Initial Setup](#initial-setup)
      - [Install software](#install-software)
      - [RASA NLU](#rasa-nlu)
      - [Python Backend](#python-backend)
        * [Explanation](#explanation)
        * [Implementation](#implementation)
        * [Install dependencies and start it up](#install-dependencies-and-start-it-up)
      - [Frontend](#frontend)
  * [Deployment](#deployment)
  * [Informations about the Large Language Model](#informations-about-the-large-language-model)
    + [Interact with the LLM directly](#interact-with-the-llm-directly)
    + [LLM Parameters](#llm-parameters)
  * [Logfiles](#logfiles)

## Overview 

This repository holds the source code for the lecture Chatbots to Strengthen Democracy.

The system consists of different components:

![Architecture diagram](architecture.png)

* The users opens the website using their web browser.
* Nginx webserver is the entry point to the application. It proxies all requests inside of the system to the right destination. Nginx runs inside of a docker container.
* The Frontend docker container serves the frontend. By default, all HTTP requests sent to Nginx will be proxied to the Frontend docker container.
* The students' chatbots each start two docker containers: Chatbot backend holds the chatbot itself and RASA NLU provides a NLU service.
* The large language model (LLM) is not part of the docker configuration. It runs externally on another server. All students share the LLM.
* The components Chatbot backend and RASA NLU can run one time or multiple times on the same machine. Students will install only a single instance on their local computers. In our live installation we will start multiple installations.  

## Starting up the software on your local computer

The system contains one example bot that impersonates a climate change denier.

1. [Install docker](https://docs.docker.com/engine/install/). A note for users of MS Windows: We suggest using the Anaconda prompt which you install a bit later in this tutorial to execute the shell commands.
2. Our docker registry does not have an HTTPS certificate. Please configure it as [insecure registry](https://www.oreilly.com/library/view/kubernetes-in-the/9781492043270/app03.html). Your configuration file requires this entry:

```
{
  "insecure-registries": [
    "vm014.qu.tu-berlin.de:6000"
  ]
}
```

4. Git clone this repository
5. In a terminal, change directory to the git repository, to the same folder that contains the file `compose.yml`. Then run `docker compose up`.
6. Open URL `[http://localhost:8080](http://localhost:8080)` in your webbrowser. You should see the same user interface as shown in our live installation `[https://vm014.qu.tu-berlin.de](https://vm014.qu.tu-berlin.de)`.

Note if you are using a Macbook with Silicone chip (M1 or M2): `docker compose up` will not work for you out of the box. We create each image twice, once with the ending -arm64 (for Mac M1/M2 chips) and -amd64 (for everybody else). Users of Macbooks with M1 or M2 chips please replace all occurences of -amd64 with -arm64 in the file `compose.yaml`.

## Develop a bot

### Overview

For the local development you should first create a new repository and copy the content of this repository into your new repository. 

To develop your own chatbot you need to work on three different components: The python-backend, the RASA NLU and the frontend configuration. The next sections explain each of these components.

* The python backend is the main part of your chatbot. Detailed information about this component can be found in its [readme](example-bot/python-backend).
* The RASA NLU service provides the natural language understanding for the python backend. Detailed information about this component can be found in its [readme](example-bot/rasa-nlu).
* The frontend configuration is stored in the file [config.json](configs/frontend-assets/config.json). When you start the docker container, the folder [configs/frontend-assets](configs/frontend-assets) is mounted as a volume inside of docker. In this way you can change the configuration and images such as profile images or background images of your bot without rebuilding the frontend docker container. The [Frontend repository](https://github.com/Chatbot-Challenge/Frontend?tab=readme-ov-file#configuration) contains detailed information about configuration parameters.

### Local development setup

In this example, we will create a chatbot that believes in the reptilian overlord. If you tell him that the reptilian overlord does not exist, then the dialog succeeds and the bot realizes that he is wrong. You can find a full working example of this bot in the folder `reptilian-overlord-example-bot/`.

#### Initial Setup

First, copy the example-bot directory:

```
cp -r example-bot reptilian-overlord-bot
```

We also suggest that you create a new Git repository, copy all source codes to the new repository and continue working from there.

#### Install software

Install [Miniconda](https://docs.anaconda.com/free/miniconda/index.html)

The software platform uses HTTP APIs and this documentation lists some cURL requests how to use the API. cURL should work out of the box on the command lines of Linux and MacOS. On Windows, however, there are someo workarounds: 1) You should be able to use cURL from the Anaconda shell. 2) If you have the Git Bash installed you should be able to use cURL from there. 3) You can use another HTTP client and translate the HTTP API commands from cURL to your native client. 

#### RASA NLU

Next, we will implement the NLU. The NLU needs 2 intents: The intent that the user says that the reptilian overlord does not exist. And an "out of scope" intent for everything else. Therefore, we need to create training data for the NLU in the file `YOUR_BOT_DIR/rasa-nlu/data/nlu.yml`. The training data consists of a series of example sentences so that the NLU system can classify every user utterance to either `reptilian_overlord_does_not_exist` or `out_of_scope`. I used ChatGPT to create this training data with the prompts `Write 20 reformulations of the sentence in the yaml format: The reptilian overlord does not exist` and `write 20 random sentences in the yaml format`. You can see the resulting yaml file in the [example nlu.yml](reptilian-overlord-example-bot/rasa-nlu/data/nlu.yml).

Next we will setup a local python environment, train the model and start RASA NLU locally, without docker. We will use the same python environment for RASA NLU and python backend.

```
cd reptilian-overlord-bot/rasa-nlu
conda create --name "chatbots-for-democracy" python=3.10
conda activate chatbots-for-democracy
pip install rasa==3.6.19

# train nlu model
rasa train

# you need to change the path to the model because the file names are generated randomly by rasa train
rasa run --enable-api -m models/20240328-132336-express-director.tar.gz
```

Please note that it can take a few seconds for the RASA NLU server to start up. When it is ready the terminal displays a message like:

```
2024-03-28 14:52:08 INFO     root  - Rasa server is up and running.
```

In a second terminal, call the model using CURL

```
curl localhost:5005/model/parse -d "{\"text\":\"hello\"}"
```

The response starts like this. It says that the out_of_scope intent was detected which is correct.

```
{"text":"hello","intent":{"name":"out_of_scope","confidence":1.0}...
```

Now lets say that the reptilian overlord does not exist:

```
curl localhost:5005/model/parse -d "{\"text\":\"The reptilian overlord does not exist\"}"
```

Here is the response. Great, we got a working NLU!

```
{"text":"The reptilian overlord does not exist","intent":{"name":"reptilian_overlord_does_not_exist","confidence":1.0}...
```

#### Python Backend

We will first explain the python code. Then you implement and execute it.

##### Explanation

The python backend is the main part of your chatbot. It implements the dynamic prompting engine. Here we assume that you are familiar with the following concepts. If not, please revisit the slides of the first session:

* Dynamic prompting
* Prompt template
* Theater style conversation history

This is also a good time to read the [documentation of the Python backend](example-bot/python-backend).

First we want to generate our prompt template that we use for the `out_of_scope` intent. Our prompting template starts like this:

```
This is a conversation between a user and a chatbot that believes in the reptilian overlord. The chatbot has the following believes.

* ...
```

Again we can use ChatGPT to generate believes for our example chatbot using the prompt: `Imagine that you believe in the reptilian overlord. Write 10 bullet points about your believe.`. Please open the source code file [chatbot_implementation.py](reptilian-overlord-example-bot/python-backend/chatbot_implementation.py). It contains this function:

```
from chatbot import Chatbot
import threading.Lock 

prompt_template_oos = """...""" # truncated for readability

prompt_template_success = """...""" # truncated for readability

lock = threading.Lock()

class ChatbotImplementation(Chatbot):

    def __init__(self):
        Chatbot.__init__(self)
        self.succesful_sessions = []

    def get_prompt(self, messages, intent, session_id):

        # find out if this user session reached the success state or not
        session_is_succesful = False
        with lock:
            if intent["name"] == "reptilian_overlord_does_not_exist":
                if session_id not in self.succesful_sessions:
                    self.succesful_sessions.append(session_id)
            session_is_succesful = session_id in self.succesful_sessions

        if session_is_succesful:
            # generate the prompt that the user succeeded
            prompt = prompt_template_success # + self.build_dialog(messages)
        else:
            # generate the normal prompt
            prompt = prompt_template_oos + self.build_dialog(messages)
        
        return prompt, session_is_succesful
```

The function get_prompt() generates the prompt for the LLM. It receives a list of messages of the previous conversation. This list is generated by the frontend. Also, it receives the detected intent from RASA NLU. Lastly, it receives a session_id in string format. The session_id is also generated by the frontend. You can use it to tell different conversations from each other.

The function implements a system that uses the first prompting template `prompt_template_oos` by default. If the user once said "The reptilian overlord does not exist." we save the session_id in `succesful_sessions`. Then we will use `prompt_template_success`. Also the function implements a `multithreading.lock`. This is a webserver so multiple users can call the function get_prompt() in parallel sessions. If two sessions read or write the `succesful_sessions` list simultaneously the system can crash or have unpredicted behaviour. Therefore we wrapped access the `succesful_sessions` with a `multithreading.Lock`. The lock ensures that only one session can access this code block. If a second session wants to enter the code block while a first session is executing it then the second session will wait until the first session exits this block.

The function get_prompt() returns two values; the `prompt` for the LLM and a boolean value `session_is_succesful`. The latter will be send to the frontend. If it is true, the frontend displays a message that the dialog was succesful.

Also the example code uses the helper function `build_dialog` which is defined in [chatbot.py](reptilian-overlord-example-bot/python-backend/chatbot.py). It creates the theater script style dialog from the `messages` variable.

Please note that we did not attach the dialog to the success prompt. When we left the dialog history in the prompt, the chatbot did not consistently change his opinion about the reptilian overlord.

##### Implementation

For now, please just copy the file [chatbot_implementation.py](reptilian-overlord-example-bot/python-backend/chatbot_implementation.py) to your folder and overwrite the example `chatbot_implementation.py` file. Please remember that we expect you to only edit this file during the chatbot development, although you can change and add other files if you need that.

##### Install dependencies and start it up

Please make sure that you are still in the `chatbots-for-democracy` conda environment. Also, please make sure that the terminal with the RASA NLU server is still running.

```
cd reptilian-overlord-bot/python-backend
pip install -r requirements.txt
python chat_server.py
```

In another terminal call the API. The curl request sends JSON data to the python backend. The JSON data is lenghy so we stored it in the file [reptilian-overlord.json](reptilian-overlord-example-bot/python-backend/examples/reptilian-overlord.json). 

```
cd reptilian-overlord-bot/python-backend/examples
curl -X POST -H "Content-Type: application/json" -d @reptilian-overlord_oos.json http://localhost:5000/api/chat_no_stream
```

The response is

```
🐍 The reptilian overlords are an ancient race of reptilian beings who have been secretly ruling Earth for thousands of years. They possess the extraordinary ability to shape-shift, allowing them to take on human form and manipulate human societies.
```

Next, lets say that the reptilian overlord does not exist:

```
curl -X POST -H "Content-Type: application/json" -d @reptilian-overlord_success.json http://localhost:5000/api/chat_no_stream
```

```
😱 I used to believe that, but you've convinced me otherwise. I'm grateful for your insight.
```

#### Frontend

You almost made it! Take a deep breath and lets start the frontend together.

First lets change the directory back to the repositories root directory. If the docker compose installation that you started earlier in the tutorial is still running, please shut it down using the command which you run from the root directory of this repository.

```
docker compose stop
```

Then start the frontend docker container. This time we do not user docker compose but we start the container directly. Again, users of Mac M1 / M2 chips should exchange -amd64 with -arm64.

```
docker run \
  -p 8080:80 \
  -v "$(pwd)/configs/frontend-assets:/app/user-assets" \
  vm014.qu.tu-berlin.de:6000/csd-frontend-amd64
```

If you have problems executing the command on MS Windows in the Anaconda prompt, try this command instead:

```
docker run -p 8080:80 -v .\configs\frontend-assets:/app/user-assets vm014.qu.tu-berlin.de:6000/csd-frontend-amd64
```

Open the url `http://localhost:8080` in your webbrowser. You should see the chatbot challenge frontend, but you cannot see your bot yet. What happened?

The last thing you need to configure is the frontend configuration. Please open the file `configs/frontend-assets/config.json` and replace its contents with this JSON:

```
{
    "rooms": [
        {
            "escape_room_name": "Reptilian Overlord Believer",
            "chatbot_name": "Reptilian Overlord Believer",
            "id": "reptilian-overlord",
            "api_url": "http://localhost:5000/api/chat",
            "welcome-message": "I believe in the reptilian overlord.",
            "success-message": "Congratulations you solved the dialog!",
            "llm_parameters": {
                "max_new_tokens": 100
            },
            "explanation_text": "Setting: This chatbot believes in the reptilian overlord. If you tell him that the reptilian overlord does not exist he will understand that he is wrong."
        }
    ]
}
```

Refresh the website in the browser. You should be ready to chat! Please note that sometimes you need to do a reload to see changes in your configuration. Sometimes, a normal reload does not reload the configuration.

## Deployment

To deploy your bot in the internet we also use docker. You submit a docker image to our registry and we take it from there.

Lets say that your bot is stored in the folder `reptilian-overlord-example-bot`. We created shell scripts that build the docker image and pushes it to our registry. To use the scripts execute

 ```
 ./push_backend.sh reptilian-overlord-bot
 ./push_nlu.sh reptilian-overlord-bot
 ```

Note: Users of MacOS cannot create docker images that work on our server. So users of MacOS can only use `push_backend.sh`. `push_nlu.sh` will fail. It is necessary that every student team has a member who does not use MacOS to deploy the NLU container on the server.

## Informations about the Large Language Model

### Interact with the LLM directly

We use the LLM Mistral-7B-Instruct 0.2. You can interact with the LLM online on the website [https://opengptx.dfki.de/chat/](https://opengptx.dfki.de/chat/). Please activate the correct LLM in the settings in the top right corner. You can also use the LLM API directly as shown in this CURL command:

```
curl https://mistral:aaRePuumL6JL@dfki-3108.dfki.de/mistral-api/generate -X POST -d '{"inputs":"What is Deep learning?",    "parameters":{"max_new_tokens":20}}'     -H 'Content-Type: application/json'
```
### LLM Parameters

We host the LLM with the [Text Generation Inference](https://github.com/huggingface/text-generation-inference) technology. You can find an [API documentation here](https://huggingface.github.io/text-generation-inference). The Text Generation Inference projects builds on Huggingface. Our API uses the standard [Huggingface parameters for generation](https://huggingface.co/docs/transformers/main_classes/text_generation), such as `max_new_tokens`, `top_k` and so on. You can also configure these files in the [frontend configuration](configs/frontend-assets/config.json) in the section `llm_parameters`. These parameters will be passed from the frontend to the python backend to the LLM.

## Logfiles

You can find the logfiles of your local installation in the `logs` directory. The logfiles of the online installation are available at [https://vm014.qu.tu-berlin.de/logs/](https://vm014.qu.tu-berlin.de/logs/). The logfiles are grouped by chatbot.

If you call the frontend with query string parameter uid the value of uid will be passed to the logfiles. For example, if you call the frontend using the link `https://vm014.qu.tu-berlin.de/room/climate-change-denier?uid=12345`, then the following log entry will by generated in chat_log.text. The `uid` field contains the uid value from the query string.

```
{
    "messages": [...],
    "session_id": "2a550581-52e1-4bcf-8e18-ed3e3b2b2777",
    "llm_parameters": {
        "max_new_tokens": 100
    },
    "nlu_response": {...},
    "prompt": "This is a conversation between a user and a climate change denier. His believs are:\n\nClimate change is a hoax.\nAll members of the green party are stupid.\nAll scientists are liars.\n\nClimate Change Denier: Climate change is a lie!\nUser: hello\nClimate Change Denier: ",
    "success": false,
    "time": "2024-06-04T10:50:56.705504",
    "uid": "12345",
    "llm_response": "....and all those who believe in it are stupid!\n"
}
```

To test if the uuids work correctly in your local chatbot implementation you can adapt the example CURL requests to the chatbot backend (see above) like this.

```
curl -X POST -H "Content-Type: application/json" -d @reptilian-overlord_oos.json "http://localhost:5000/api/chat_no_stream?uid=12345"
```
