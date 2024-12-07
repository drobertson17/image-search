import json
import logging
from dataclasses import dataclass
from textwrap import dedent
from typing import Callable

import boto3
from botocore.exceptions import ClientError, UnauthorizedSSOTokenError

from image_search.prompts import RequestData


@dataclass
class UseTracker:
    cost_per_token: float
    total_cost: float = 0.0
    total_tokens: int = 0


def llama_text_prompt(system: str, user: str) -> str:
    return dedent(f"""
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>{system}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>{user}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """)


def llama_image_prompt(system: str, user: str) -> str:
    return dedent(f"""
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>{system}<|eot_id|>
        <|start_header_id|>user<|end_header_id|><|image|>{user}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """)


class BedrockLlama:
    def __init__(
            self,
            model_id: str,
            prompt_fcn: Callable[[str, str], str],
            cost_per_token: float,
            logger_name: str
        ):
        session = boto3.Session(profile_name='default')
        self.bedrock = session.client("bedrock-runtime")
        self.model_id = model_id
        self.prompt_fcn = prompt_fcn
        self.usage = UseTracker(cost_per_token = cost_per_token)
        self.logger = logging.getLogger(logger_name)
        logging.basicConfig(level=logging.INFO, filename=f"{logger_name}.log")

    def update_use_tracking(self, model_response: dict):
        total_tokens = (model_response["prompt_token_count"] + 
                        model_response["generation_token_count"])
        total_cost = self.usage.cost_per_token * total_tokens
        self.usage.total_tokens += total_tokens
        self.usage.total_cost += total_cost
    
    def prepare_request(self, request_data: RequestData) -> dict:
        return {}

    def run(self, request_data: RequestData) -> str:
        request = self.prepare_request(request_data)

        try:
            # Invoke the model with the request.
            response = self.bedrock.invoke_model(
                modelId=self.model_id, body=request
            )

        except UnauthorizedSSOTokenError:
             raise Exception("Expired Token.")
    
        except (ClientError, Exception) as e:
            self.logger.error(
                f"ERROR: Can't invoke '{self.model_id}'. Reason: {e}"
            )
            return ""
        
        model_response = json.loads(response["body"].read())
        self.update_use_tracking(model_response)

        return model_response["generation"]
    

class BedrockLlamaTextLLM(BedrockLlama):
    def __init__(
            self,
            model_id: str = "us.meta.llama3-2-11b-instruct-v1:0",
            prompt_fcn: Callable[[str, str], str] = llama_text_prompt,
            cost_per_token: float = 0.00015/1000,
            logger_name: str = "llama_text",
        ):
        super().__init__(
            model_id=model_id,
            prompt_fcn=prompt_fcn,
            cost_per_token=cost_per_token,
            logger_name=logger_name
        )
        
    def prepare_request(self, request_data: RequestData) -> dict:
        native_request = {
            "prompt": self.prompt_fcn(
                request_data.system,
                request_data.user
            ),
            "max_gen_len": 512,
            "temperature": 0.5,
        }

        return json.dumps(native_request)


class BedrockLlamaMultiModeVLM(BedrockLlama):
    def __init__(
            self,
            model_id: str = "us.meta.llama3-2-11b-instruct-v1:0",
            prompt_fcn: Callable[[str, str], str] = llama_image_prompt,
            cost_per_token: float = 0.00016/1000,
            logger_name: str = "llama_image",
        ):
        super().__init__(
            model_id=model_id,
            prompt_fcn=prompt_fcn,
            cost_per_token=cost_per_token,
            logger_name=logger_name
        )
        
    def prepare_request(self, request_data: RequestData) -> dict:
        native_request = {
            "prompt": self.prompt_fcn(
                request_data.system,
                request_data.user
            ),
            "max_gen_len": 512,
            "temperature": 0.5,
            "images": [request_data.image_b64],
        }

        return json.dumps(native_request)
