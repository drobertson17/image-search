import os
import re
import uuid
from dotenv import load_dotenv

from tqdm import tqdm

import prompts
from bedrock import BedrockLlamaMultiModeVLM, BedrockLlamaTextLLM
from db import ImageDBService
from image import ImageProcessor
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
            self.llm = BedrockLlamaTextLLM()
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

    def process_image(self, image_path: str) -> tuple[dict, str]:
        img = ImageProcessor(image_path)

        results = {
            "id": str(uuid.uuid4()),
            "full_path": image_path,
            "base_dir": "/".join(image_path.split("/")[:-1]),
            "file_name": image_path.split("/")[-1],
            "file_type": image_path.split(".")[-1],
            "predominant_color": img.get_predominant_color(),
            "exif_data": img.exif_data
        }

        return results, img.vlm_image
    
    def describe_image(self, vlm_image: str) -> dict:
        request = prompts.RequestData(
            system=prompts.IMAGE_DESC_SYSTEM_PROMPT,
            user=prompts.IMAGE_DESC_USER_PROMPT,
            image_b64=vlm_image
        )
        img_desc = self.vlm.run(request)
        
        summary_prompt = prompts.summarize_description_prompt(img_desc)
        summary_response = self.llm.run(summary_prompt)
        summary = prompts.post_process_summary(summary_response)

        results = {
            "long_desc": img_desc,
            "short_desc": summary["summary"],
            "keywords": summary["keywords"],
            "image_classification": summary["keywords"].split(",")[0],
            "title": summary["title"],
        }

        classification = self.get_classification(summary["classification"])
        if classification is not None:
            results[classification] = True

        return results

    def process(self):
        self.build_file_list()
        for image in tqdm(self.files):
            if self.db.image_exists(image):
                continue
            try:
                results, vlm_image = self.process_image(image)
            except:
                print(f"skipping: {image}")
                continue
            results.update(self.describe_image(vlm_image))
            self.db.add_image_description(results)


if __name__ == "__main__":
    process = ProcessImages(local_mode=False)
    process.process()
    print()