"""Minimal structured-output LLM abstraction and OpenAI-compatible client."""
import json
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from src.config import AgentSettings

StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)


class StructuredLLM(Protocol):
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_model: type[StructuredOutput],
    ) -> StructuredOutput:
        """Generate and validate one structured model response."""


class LLMResponseError(RuntimeError):
    """Raised when an LLM request fails or returns invalid structured output."""


class OpenAIStructuredLLM:
    """OpenAI-compatible structured-output client with lazy initialization."""

    def __init__(self, settings: AgentSettings):
        if not settings.api_key:
            raise ValueError("OPENAI_API_KEY is required for live LLM inference.")
        from openai import OpenAI

        self.settings = settings
        self.client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
        )

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_model: type[StructuredOutput],
    ) -> StructuredOutput:
        try:
            response = self.client.chat.completions.create(
                model=self.settings.llm_model,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": output_model.__name__,
                        "strict": True,
                        "schema": output_model.model_json_schema(),
                    },
                },
            )
            content = response.choices[0].message.content
            if not content:
                raise LLMResponseError("LLM returned an empty response.")
            return output_model.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError, IndexError, AttributeError) as exc:
            raise LLMResponseError(f"Invalid structured LLM response: {exc}") from exc
        except LLMResponseError:
            raise
        except Exception as exc:
            raise LLMResponseError(f"LLM request failed: {exc}") from exc
