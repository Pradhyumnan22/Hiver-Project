"""Minimal structured-output LLM abstraction and OpenAI-compatible client."""
from typing import Protocol, TypeVar
from threading import Lock

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
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.requests = 0
        self._usage_lock = Lock()
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
            response = self.client.chat.completions.parse(
                model=self.settings.llm_model,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=output_model,
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise LLMResponseError("LLM returned an empty response.")
            with self._usage_lock:
                self.requests += 1
                if response.usage is not None:
                    self.prompt_tokens += int(response.usage.prompt_tokens or 0)
                    self.completion_tokens += int(response.usage.completion_tokens or 0)
            return output_model.model_validate(parsed)
        except (ValidationError, IndexError, AttributeError) as exc:
            raise LLMResponseError(f"Invalid structured LLM response: {exc}") from exc
        except LLMResponseError:
            raise
        except Exception as exc:
            raise LLMResponseError(f"LLM request failed: {exc}") from exc

    @property
    def usage(self) -> dict[str, int]:
        return {
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }
