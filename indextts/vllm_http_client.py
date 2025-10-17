"""HTTP-based client for delegating vLLM generation to a remote service."""

import types
from typing import Any, AsyncGenerator, Dict, Iterable, Optional

import httpx
import torch
from msgspec.structs import asdict


def _serialize_multi_modal_data(multi_modal_data: Dict[str, Any]) -> Dict[str, Any]:
    serialized: Dict[str, Any] = {}
    for key, value in multi_modal_data.items():
        if key == "audio" and isinstance(value, dict):
            audio_payload: Dict[str, Any] = {}
            for audio_key, audio_value in value.items():
                if audio_key == "audio_embeds":
                    embeds: list[list[list[float]]] = []
                    for embed in audio_value:
                        if isinstance(embed, torch.Tensor):
                            embeds.append(embed.cpu().tolist())
                        else:
                            embeds.append(torch.as_tensor(embed).tolist())
                    audio_payload[audio_key] = embeds
                else:
                    audio_payload[audio_key] = audio_value
            serialized[key] = audio_payload
        else:
            serialized[key] = value
    return serialized


def _serialize_prompt(prompt: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, value in prompt.items():
        if key == "multi_modal_data" and isinstance(value, dict):
            payload[key] = _serialize_multi_modal_data(value)
        else:
            payload[key] = value
    return payload


def _serialize_sampling_params(sampling_params: Any) -> Dict[str, Any]:
    params = asdict(sampling_params)
    filtered: Dict[str, Any] = {}
    for key, value in params.items():
        if key.startswith("_") or key in {"output_kind", "output_text_buffer_length"}:
            continue
        if isinstance(value, set):
            filtered[key] = list(value)
        else:
            filtered[key] = value
    return filtered


class VLLMHTTPClient:
    """Thin wrapper that mimics the ``AsyncLLM`` streaming API over HTTP."""

    def __init__(self, base_url: str, timeout: float = 300.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout)
        self._closed = False

    async def close(self) -> None:
        if not self._closed:
            await self._client.aclose()
            self._closed = True

    async def __aenter__(self) -> "VLLMHTTPClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        await self.close()

    def generate(
        self,
        prompt: Dict[str, Any],
        sampling_params: Any,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[Any, None]:
        payload = {
            "prompt": _serialize_prompt(prompt),
            "sampling_params": _serialize_sampling_params(sampling_params),
            "request_id": request_id,
        }

        async def _iterator() -> AsyncGenerator[Any, None]:
            response = await self._client.post("/internal/vllm/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            token_ids: Iterable[int] = data["token_ids"]
            output = types.SimpleNamespace(
                outputs=[types.SimpleNamespace(token_ids=list(token_ids))]
            )
            yield output

        return _iterator()

    async def check_health(self) -> None:
        response = await self._client.get("/health")
        response.raise_for_status()

    # Compatibility helpers -------------------------------------------------
    async def shutdown(self) -> None:
        await self.close()

    async def reset_mm_cache(self) -> None:  # pragma: no cover - optional endpoint
        try:
            await self._client.post("/internal/vllm/reset_mm_cache")
        except httpx.HTTPError:
            # Endpoint may not be implemented on the server; ignore.
            pass


__all__ = ["VLLMHTTPClient"]
