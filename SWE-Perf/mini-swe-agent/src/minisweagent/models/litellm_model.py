import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import time

from evoml_services.clients.thanos.client import ThanosSettings
from artemis_client.vision.client import VisionClient, VisionSettings
from vision_models import LLMInferenceRequest, LLMAskInferenceRequest, LLMConversationMessage
 
thanos_settings = ThanosSettings.with_env_prefix("thanos", _env_file=".env")
settings = VisionSettings.with_env_prefix("vision", _env_file=".env")
client = VisionClient(settings, thanos_settings)

import litellm
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS

logger = logging.getLogger("litellm_model")


@dataclass
class LitellmModelConfig:
    model_name: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")


class LitellmModel:
    def __init__(self, **kwargs):
        self.config = LitellmModelConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                litellm.exceptions.UnsupportedParamsError,
                litellm.exceptions.NotFoundError,
                litellm.exceptions.PermissionDeniedError,
                litellm.exceptions.ContextWindowExceededError,
                litellm.exceptions.APIError,
                litellm.exceptions.AuthenticationError,
                KeyboardInterrupt,
            )
        ),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            return litellm.completion(
                model=self.config.model_name, messages=messages, **(self.config.model_kwargs | kwargs)
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    # def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
    #     response = self._query(messages, **kwargs)
    #     cost = litellm.cost_calculator.completion_cost(response)
    #     self.n_calls += 1
    #     self.cost += cost
    #     GLOBAL_MODEL_STATS.add(cost)
    #     return {
    #         "content": response.choices[0].message.content or "",  # type: ignore
    #     }

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        ### VISION ###############################
        max_retries = 5
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                converted_messages = [
                    LLMConversationMessage(
                        role=message['role'], 
                        content=message['content'] if isinstance(message['content'], str) else message['content'][0]['text']
                    ) for message in messages
                ]
                
                # Map model names to vision client format
                model_type = self.config.model_name
                if model_type == "gemini-2.5-pro":
                    model_type = "gemini-v25-pro"  # Correct format
                elif model_type == "claude-3-5-sonnet-20241022":
                    model_type = "claude-v35-sonnet"  # Correct format
                elif model_type == "claude-sonnet-4-20250514":
                    model_type = "claude-v4-sonnet"  # Correct format
                
                response = client.ask(
                    LLMAskInferenceRequest(
                        messages=converted_messages,
                        model_type=model_type,
                        with_history=True
                    )
                )
                
                cost = response.task_info.total_cost
                ##########################################
                
                self.n_calls += 1
                self.cost += cost
                GLOBAL_MODEL_STATS.add(cost)
                return {"content": response.messages[-1].content or ""}
                
            except Exception as e:
                error_msg = str(e)
                if "Too many connections" in error_msg and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff: 2, 4, 8, 16, 32 seconds
                    logger.warning(f"Vision client connection error (attempt {attempt + 1}/{max_retries}), retrying in {delay}s: {error_msg}")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Vision client error: {e}")
                    raise e

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}
