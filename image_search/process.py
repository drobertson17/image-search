import os
import re
import uuid
from dotenv import load_dotenv

from tqdm import tqdm

import prompts
from bedrock import BedrockLlamaMultiModeVLM, BedrockLlamaTextLLM
from db import ImageDBService
from image import get_predominant_color
from local_models import TextModel, VisionModel


load_dotenv()


class ProcessImages:
    def __init__(self, local_mode: bool = True):
        self.directory = os.getenv("DIR_TO_PROCESS")
        self.db = ImageDBService()
        if local_mode:
            self.vlm = VisionModel()
            self.llm = TextModel()
        else:
            self.vlm = BedrockLlamaMultiModeVLM()
            self.llm_1 = TextModel()
            self.llm_2 = BedrockLlamaTextLLM()
        self.valid_file_types = ["jpg", "jpeg", "png"]

    def build_file_list(self):
        self.files = []
        for root, _, files in os.walk(self.directory):
            for file in files:
                if file.split(".")[-1].lower() in self.valid_file_types:
                    self.files.append(os.path.join(root, file))

    def get_classification(self, llm_response: str) -> str:
        col_mapping = {
            "1": "natural_landscape",
            "2": "building_structure",
            "3": "selfie",
            "4": "friends_or_family"
        }

        match = re.search(r'\d', llm_response)

        if not match:
            # No match found
            return
        
        if match.group(0) != "5":
            # Positively classified as "no match"
            return col_mapping[match.group(0)]


    def clean_response(self, response: str) -> str:
        response = response.split(": ")[-1].strip()
        return response.split("\n")[0]


    def process_image(self, image_path: str) -> dict:
        request = prompts.RequestData(
            system=prompts.IMAGE_DESC_SYSTEM_PROMPT,
            user=prompts.IMAGE_DESC_USER_PROMPT,
            image_path=image_path
        )
        img_desc = self.vlm.run(request)
        
        prompt = prompts.PromptGenerator(img_desc)
        title = self.llm_2.run(prompt.title_generation_prompt())
        summary = self.llm_2.run(prompt.summary_generation_prompt())
        keywords = self.llm_1.run(prompt.keyword_generation_prompt())
        classification_response = self.llm_1.run(prompt.classification_prompt())

        results = {
            "id": str(uuid.uuid4()),
            "full_path": image_path,
            "base_dir": "/".join(image_path.split("/")[:-1]),
            "file_name": image_path.split("/")[-1],
            "file_type": image_path.split(".")[-1],
            "long_desc": img_desc,
            "short_desc": self.clean_response(summary),
            "keywords": self.clean_response(keywords),
            "image_classification": self.clean_response(keywords).split(",")[0],
            "title": self.clean_response(title),
            "predominant_color": get_predominant_color(image_path),
        }

        classification = self.get_classification(classification_response)
        if classification is not None:
            results[classification] = True

        return results

    def process(self):
        self.build_file_list()
        for image in tqdm(self.files):
            if self.db.image_exists(image):
                continue
            img_desc = self.process_image(image)
            self.db.add_image_description(img_desc)


if __name__ == "__main__":
    process = ProcessImages(local_mode=False)
    process.process()
    print()