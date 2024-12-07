import os
import sys
from dotenv import load_dotenv

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

from prompts import RequestData

load_dotenv()


class TextModel:
    def __init__(self):
        self.model_file = os.getenv("LLM_MODEL_PATH")
        
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        self.model = Llama(
            model_path=self.model_file,
            n_ctx=4096, # Uncomment to increase the context window,
            verbose=False,
        )

        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    
    def run(self, request: RequestData) -> str:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        
        self.output = self.model(
            request.user,
            max_tokens=100, # Generate up to 32 tokens, set to None to generate up to the end of the context window
            stop=["Q:"], # Stop generating just before the model would generate a new question
            echo=False, # Do not echo the prompt back in the output
        )

        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        return self.output["choices"][0]["text"]


class VisionModel:
    def __init__(self):
        self.model_file = os.getenv("VLM_MODEL_PATH")
        self.clip_model_file = os.getenv("CLIP_MODEL_PATH")
        
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        chat_handler = Llava15ChatHandler(clip_model_path=self.clip_model_file)
        self.model = Llama(
            model_path=self.model_file,
            chat_handler=chat_handler,
            n_ctx=4096, # Uncomment to increase the context window,
            echo=False,
            verbose=False,
        )

        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def run(self, request: RequestData) -> str:
        
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        self.response = self.model.create_chat_completion(
            messages = [
                {"role": "system", "content": request.system},
                {
                    "role": "user",
                    "content": [
                        {"type" : "text", "text": request.user},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{request.image_b64}"} 
                    ]
                }
            ]
        )

        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        return self.response["choices"][0]["message"]["content"]
