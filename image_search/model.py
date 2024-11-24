import base64
import os
import sys
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler


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
    
    def run(self, prompt: str) -> str:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        
        self.output = self.model(
            prompt,
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

        self.system = "You are an assistant who perfectly describes images."
        self.user_prompt = "Describe the following image in a paragraph."

    def image_to_base64(self, img_path: str) -> str:
        """Converts a local image file to base64 string."""
        with Image.open(img_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def run(self, img_path: str) -> str:
        img_base64 = self.image_to_base64(img_path)
        
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        self.response = self.model.create_chat_completion(
            messages = [
                {"role": "system", "content": self.system},
                {
                    "role": "user",
                    "content": [
                        {"type" : "text", "text": self.user_prompt},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_base64}"} 
                    ]
                }
            ]
        )

        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        return self.response["choices"][0]["message"]["content"]


class PromptGenerator:
    def __init__(self, image_desc: str):
        self.image_desc = image_desc

    def title_generation_prompt(self) -> str:
        return (
            "Prompt:\Create a concise, descriptive title for an image based on "
            "the following detailed description. Only provide the title—no "
            "additional text or explanation. Keep the title matter-of-fact. For example:\n\n"
            "[EXAMPLE]\n\n"
            "IMAGE DESCRIPTION:\n"
            "The image features a wooden pathway in a lush green field, surrounded "
            "by a beautiful landscape. The pathway is lined with grass.\n\n"
            "YOUR RESPONSE:\n"
            "Title: Wooden Pathway"
            "\n\n[/EXAMPLE]\n\n"
            f"\n\nIMAGE DESCRIPTION:\n{self.image_desc}\n\n"
            "YOUR RESPONSE:\n"
        )
    
    def summary_generation_prompt(self) -> str:
        return (
            "Prompt:\Create a concise, descriptive summary based on "
            "the following detailed description. Only provide the summary—no "
            "additional text or explanation. Keep the text matter-of-fact. For example:\n\n"
            "[EXAMPLE]\n\n"
            "IMAGE DESCRIPTION:\n"
            "The image features a wooden walkway or bridge, surrounded by a lush "
            "green field with tall grass. The pathway is located in the middle of "
            "the field, providing a connection between different areas. The sky above "
            "the field is blue, and the overall atmosphere appears serene and peaceful. "
            "The scene is reminiscent of a park or a countryside setting, where one can "
            "enjoy a leisurely walk or a quiet moment of reflection.'.\n\n"
            "YOUR RESPONSE:\n"
            "Summary: The image features a wooden pathway in a lush green field, surrounded "
            "by a beautiful landscape. The pathway is lined with grass"
            "\n\n[/EXAMPLE]\n\n"
            f"\n\nIMAGE DESCRIPTION:\n{self.image_desc}\n\n"
            "YOUR RESPONSE:\n"
        )

    def keyword_generation_prompt(self) -> str:
        return (
            "Prompt:\Create a concise, descriptive list of keywords based on "
            "the following detailed description. Only provide the keywords—no "
            "additional text or explanation. For example:\n\n"
            "[EXAMPLE]\n\n"
            "IMAGE DESCRIPTION:\n"
            "The image features a wooden walkway or bridge, surrounded by a lush "
            "green field with tall grass. The pathway is located in the middle of "
            "the field, providing a connection between different areas. The sky above "
            "the field is blue, and the overall atmosphere appears serene and peaceful. "
            "The scene is reminiscent of a park or a countryside setting, where one can "
            "enjoy a leisurely walk or a quiet moment of reflection.'.\n\n"
            "YOUR RESPONSE:\n"
            "Keywords: Grass, nature, pathway"
            "\n\n[/EXAMPLE]\n\n"
            f"\n\nIMAGE DESCRIPTION:\n{self.image_desc}\n\n"
            "YOUR RESPONSE:\n"
        )


    def classification_prompt(self) -> str:
        return (
            "Prompt: Select one of the following options to classify the image based on "
            "it's description:\n"
            " 1. Natural/landscape\n"
            " 2. Picture of building or structure\n"
            " 3. Selfie\n"
            " 4. Friends and Family Picture\n"
            " 5. None of the above\n\n"
            "[EXAMPLE]\n\n"
            "IMAGE DESCRIPTION:\n"
            "The image features a wooden walkway or bridge, surrounded by a lush "
            "green field with tall grass. The pathway is located in the middle of "
            "the field, providing a connection between different areas. The sky above "
            "the field is blue, and the overall atmosphere appears serene and peaceful. "
            "The scene is reminiscent of a park or a countryside setting, where one can "
            "enjoy a leisurely walk or a quiet moment of reflection.'.\n\n"
            "YOUR RESPONSE:\n"
            "1"
            "\n\n[/EXAMPLE]\n\n"
            f"\n\nIMAGE DESCRIPTION:\n{self.image_desc}\n\n"
            "YOUR RESPONSE:\n"
        )