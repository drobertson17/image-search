import os
import re
import uuid
from dotenv import load_dotenv

from tqdm import tqdm

from model import TextModel, VisionModel, PromptGenerator
from db import ImageDBService


load_dotenv()


class ProcessImages:
    def __init__(self):
        self.directory = os.getenv("DIR_TO_PROCESS")
        self.db = ImageDBService()
        self.vlm = VisionModel()
        self.llm = TextModel()
        self.valid_file_types = ["jpg", "jpeg"]

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
        if match:
            return col_mapping[match.group(0)]


    def clean_response(self, response: str) -> str:
        response = response.split(": ")[-1].strip()
        return response.split("\n")[0]


    def process_image(self, img_path: str) -> dict:
        img_desc = self.vlm.run(img_path)
        
        prompt = PromptGenerator(img_desc)
        title = self.llm.run(prompt.title_generation_prompt())
        summary = self.llm.run(prompt.summary_generation_prompt())
        keywords = self.llm.run(prompt.keyword_generation_prompt())
        classification_response = self.llm.run(prompt.classification_prompt())

        results = {
            "id": str(uuid.uuid4()),
            "full_path": img_path,
            "base_dir": "/".join(img_path.split("/")[:-1]),
            "file_name": img_path.split("/")[-1],
            "file_type": img_path.split(".")[-1],
            "long_desc": img_desc,
            "short_desc": self.clean_response(summary),
            "keywords": self.clean_response(keywords),
            "title": self.clean_response(title),
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
    process = ProcessImages()
    process.process()
    print()