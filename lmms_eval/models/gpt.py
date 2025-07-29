import json
import os
import time
from typing import List, Tuple

import requests as url_requests
from accelerate import Accelerator, DistributedType
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

API_TYPE = os.getenv("API_TYPE", "openai")
NUM_SECONDS_TO_SLEEP = 30
from loguru import logger as eval_logger
import weave

class ResponseWithDocIds(str):
    """Custom string class that can store doc_ids while being compatible with string operations"""
    def __new__(cls, content, doc_ids=None):
        obj = str.__new__(cls, content)
        obj.doc_ids = doc_ids or []
        return obj

@weave.op()
def request_server(self, payload):
    response = url_requests.post(
        self.api_url,
        headers=self.headers,
        json=payload,
        timeout=self.timeout
    )
    
    # Check if the request was successful
    if response.status_code != 200:
        raise Exception(f"HTTP {response.status_code}: {response.text} (URL: {self.api_url})")
    
    try:
        return response.json()
    except Exception as e:
        raise Exception(f"Failed to parse JSON response: {e}. Response text: {response.text}")

@register_model("gpt")
class GPT(lmms):
    def __init__(
        self,
        model_version: str = "gpt-4o-mini",
        timeout: int = 120,
        continual_mode: bool = False,
        response_persistent_folder: str = None,
        api_url: str = None,
        api_key: str = None,
        api_type: str = "openai",
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.timeout = timeout
        self.continual_mode = continual_mode

        # 設置 API 相關配置
        self.api_type = api_type.lower()
        self.api_url = api_url or os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")

        # 根據 API 類型設置 headers
        if self.api_type == "azure":
            self.headers = {
                "api-key": self.api_key,
                "Content-Type": "application/json",
            }
        else:  # 默認使用 openai 風格的 header
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

        if self.continual_mode:
            if response_persistent_folder is None:
                raise ValueError("Continual mode requires a persistent path for the response. Please provide a valid path.")

            os.makedirs(response_persistent_folder, exist_ok=True)
            self.response_persistent_folder = response_persistent_folder
            self.response_persistent_file = os.path.join(self.response_persistent_folder, f"{self.model_version}_response.json")

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
            else:
                self.response_cache = {}
                self.cache_mode = "start"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device

    @weave.op()
    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, _, doc_id, task, split in [reg.args for reg in requests]:
            if self.continual_mode is True and self.cache_mode == "resume":
                doc_uuid = f"{task}___{split}___{doc_id}"
                if doc_uuid in self.response_cache:
                    cached_response = self.response_cache[doc_uuid]
                    if cached_response:
                        # Handle cached response that might be string or dict with doc_ids
                        if isinstance(cached_response, dict) and "doc_ids" in cached_response:
                            cached_with_doc_ids = ResponseWithDocIds(cached_response["text"], cached_response["doc_ids"])
                            res.append(cached_with_doc_ids)
                        else:
                            res.append(cached_response)  # Legacy string format or ResponseWithDocIds
                        pbar.update(1)
                        continue

            payload = {
                "model": self.model_version if API_TYPE == "openai" else None,
                "messages": [{"role": "user", "content": contexts}],
            }

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if gen_kwargs["max_new_tokens"] > 16384:
                gen_kwargs["max_new_tokens"] = 16384
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0

            payload["max_tokens"] = gen_kwargs["max_new_tokens"]
            payload["temperature"] = gen_kwargs["temperature"]

            retrieved_doc_ids = []
            for attempt in range(5):
                try:
                    response_data = request_server(self, payload)
                    
                    # Check if response has expected structure
                    if not isinstance(response_data, dict):
                        raise ValueError(f"Invalid response format: expected dict, got {type(response_data)}")
                    
                    if "choices" not in response_data:
                        raise ValueError(f"Response missing 'choices' field. Response: {response_data}")
                    
                    if not response_data["choices"] or len(response_data["choices"]) == 0:
                        raise ValueError(f"Empty choices in response: {response_data}")
                    
                    response_text = response_data["choices"][0]["message"]["content"].strip()
                    
                    # Capture doc_ids from response if available
                    if "doc_ids" in response_data:
                        retrieved_doc_ids = response_data["doc_ids"]
                    break

                except Exception as e:
                    error_msg = response_data if 'response_data' in locals() else "No response data"
                    
                    eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}.\nResponse: {error_msg}")
                    if attempt < 4:  # Changed from <= 5 to < 4 for correct retry logic
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:
                        eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}.\nResponse: {error_msg}")
                        response_text = ""

            # Create response string with embedded doc_ids
            if retrieved_doc_ids:
                response_with_metadata = ResponseWithDocIds(response_text, retrieved_doc_ids)
                res.append(response_with_metadata)
            else:
                res.append(response_text)
            pbar.update(1)

            if self.continual_mode is True:
                doc_uuid = f"{task}___{split}___{doc_id}"
                # Cache the response - store as dict for JSON serialization
                if retrieved_doc_ids:
                    self.response_cache[doc_uuid] = {"text": response_text, "doc_ids": retrieved_doc_ids}
                else:
                    self.response_cache[doc_uuid] = response_text
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for GPT")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("GPT not support loglikelihood")
