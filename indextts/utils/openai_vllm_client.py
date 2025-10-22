from typing import Any, AsyncGenerator, Dict, Iterable, Mapping, Optional

import httpx
import numpy as np
import torch

from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams


def _to_serializable(obj: Any) -> Any:
    """Convert tensors and ndarrays into JSON-serializable lists."""

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _to_serializable(value) for key, value in obj.items()}
    return obj


def _serialize_prompt(prompt: Dict[str, Any]) -> Dict[str, Any]:
    """Create a JSON-serializable copy of the prompt payload."""

    serialized: Dict[str, Any] = {}
    for key, value in prompt.items():
        if key == "prompt_token_ids" and isinstance(value, Iterable):
            serialized[key] = [int(v) for v in value]
        elif key in {"multi_modal_data", "mm_processor_kwargs", "multi_modal_uuids"}:
            serialized[key] = _to_serializable(value)
        else:
            serialized[key] = value
    return serialized


class OpenAIVLLMClient:
    """Minimal AsyncLLM-compatible client that forwards requests to an OpenAI API."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must be provided when using the OpenAI client")
        if not model:
            raise ValueError("model must be provided when using the OpenAI client")

        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            headers.update(dict(extra_headers))
        self._headers = headers

    def generate(
        self,
        prompt: Dict[str, Any],
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[Any] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Proxy the generation call to an OpenAI-compatible completions endpoint."""

        serialized_prompt = _serialize_prompt(prompt)
        payload = {
            "model": self._model,
            "prompt": serialized_prompt,
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "top_k": sampling_params.top_k,
            "repetition_penalty": sampling_params.repetition_penalty,
            "max_tokens": sampling_params.max_tokens,
            "stop_token_ids": sampling_params.stop_token_ids,
            "n": 1,
            "stream": False,
            "return_token_ids": True,
            "request_id": request_id,
        }
        payload = {key: value for key, value in payload.items() if value is not None}

        async def _generator() -> AsyncGenerator[RequestOutput, None]:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/v1/completions",
                    json=payload,
                    headers=self._headers,
                )
                response.raise_for_status()
                data = response.json()

            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError("OpenAI response does not contain any choices")

            outputs = []
            for idx, choice in enumerate(choices):
                token_ids = choice.get("token_ids") or []
                text = choice.get("text", "")
                outputs.append(
                    CompletionOutput(
                        index=idx,
                        text=text,
                        token_ids=token_ids,
                        cumulative_logprob=None,
                        logprobs=None,
                        finish_reason=choice.get("finish_reason"),
                        stop_reason=choice.get("stop_reason"),
                    )
                )

            prompt_token_ids = choices[0].get("prompt_token_ids")
            if prompt_token_ids is None:
                prompt_token_ids = serialized_prompt.get("prompt_token_ids")

            request_output = RequestOutput(
                request_id=request_id,
                prompt=serialized_prompt.get("prompt"),
                prompt_token_ids=prompt_token_ids,
                prompt_logprobs=None,
                outputs=outputs,
                finished=True,
            )

            yield request_output

        return _generator()

    # The following methods are not needed by the current callers but are
    # provided to maintain compatibility with the AsyncLLM interface where
    # possible.

    @property
    def is_running(self) -> bool:
        return True

    @property
    def is_stopped(self) -> bool:
        return False

    @property
    def errored(self) -> bool:
        return False

    @property
    def dead_error(self) -> BaseException:
        return RuntimeError("OpenAI client encountered an unrecoverable error")

    async def abort(self, request_id: Any) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError

    async def get_vllm_config(self) -> Any:  # pragma: no cover - placeholder
        raise NotImplementedError

    async def get_model_config(self) -> Any:  # pragma: no cover - placeholder
        raise NotImplementedError

    async def get_decoding_config(self) -> Any:  # pragma: no cover - placeholder
        raise NotImplementedError

    async def get_input_preprocessor(self) -> Any:  # pragma: no cover - placeholder
        raise NotImplementedError

    async def get_tokenizer(self, lora_request: Optional[Any] = None) -> Any:  # pragma: no cover - placeholder
        raise NotImplementedError

    async def is_tracing_enabled(self) -> bool:  # pragma: no cover - placeholder
        raise NotImplementedError

    async def do_log_stats(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError

    async def check_health(self) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError

    async def start_profile(self) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError

    async def stop_profile(self) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError

    async def reset_mm_cache(self) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError

    async def reset_prefix_cache(self, device: Optional[Any] = None) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError

    async def sleep(self, level: int = 1) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError

    async def wake_up(self, tags: Optional[list[str]] = None) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError

    async def is_sleeping(self) -> bool:  # pragma: no cover - placeholder
        raise NotImplementedError

    async def add_lora(self, lora_request: Any) -> bool:  # pragma: no cover - placeholder
        raise NotImplementedError

