import re
from dataclasses import dataclass

from rapidfuzz import process, fuzz

@dataclass
class RequestData:
    user: str
    system: str = "You are a helpful assistant."
    image_b64: str = None


IMAGE_DESC_SYSTEM_PROMPT = "You are an assistant who perfectly describes images."
IMAGE_DESC_USER_PROMPT = "Describe the following image in a paragraph."


def summarize_description_prompt(description: str) -> str:
    return RequestData(user=
        "Prompt: "
        "Create a title, summary, list of keywords and classification for a description "
        "of an image. Classify the image based on its description into one of the following:\n"
        " 1. Natural/landscape\n"
        " 2. Picture of building or structure\n"
        " 3. Selfie\n"
        " 4. Friends and Family Picture\n"
        " 5. None of the above\n\n"
        "For example:\n\n"
        "[EXAMPLE]\n\n"
        "IMAGE DESCRIPTION:\n"
        "The image features a wooden pathway in a lush green field, surrounded "
        "by a beautiful landscape. The pathway is lined with grass.\n\n"
        "YOUR RESPONSE:\n"
        "Title: Wooden Pathway\n"
        "Summary: The image features a wooden pathway in a lush green field, surrounded "
        "by a beautiful landscape. The pathway is lined with grass\n"
        "Keywords: Grass, nature, pathway\n"
        "Classification: 1"
        "\n\n[/EXAMPLE]\n\n"
        f"\n\nIMAGE DESCRIPTION:\n{description}\n\n"
        "YOUR RESPONSE:\n"
    )
    
def post_process_summary(summary: str) -> dict:
    resp_parts = re.split(r":\s*|\n", summary)
    match_config = {
        "scorer": fuzz.ratio,
        "score_cutoff": 80,
    }

    def get_match(header: str) -> str:
        match = process.extractOne(header, resp_parts, **match_config)
        
        if match:
            match_index = match[-1]
            return resp_parts[match_index + 1].strip()
        
        return ""

    return {
        "title": get_match("title"),
        "summary": get_match("summary"),
        "keywords": get_match("keywords"),
        "classification": get_match("classification"),
    }
