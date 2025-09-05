from __future__ import annotations

import json
from typing import AsyncGenerator, Dict, Optional

import httpx

from .config import settings


class OllamaClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")

    async def ensure_model(self, model: str) -> None:
        # Try tags; if not present, trigger a pull (idempotent)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                r.raise_for_status()
                tags = r.json().get("models", [])
                names = {m.get("name") for m in tags}
                if model in names:
                    return
        except Exception:
            pass
        # Pull model
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
            async with client.stream("POST", f"{self.base_url}/api/pull", json={"name": model}) as r:
                r.raise_for_status()
                async for _ in r.aiter_lines():
                    # Ignore progress lines
                    pass

    async def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.2,
        system: Optional[str] = None,
        max_tokens: int = 150,
        timeout: float = 45.0,
    ) -> str:
        await self.ensure_model(model)
        url = f"{self.base_url}/api/generate"
        payload: Dict = {
            "model": model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 2048,  # Smaller context for speed
            },
            "stream": False,
        }
        if system:
            payload["system"] = system
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            try:
                r = await client.post(url, json=payload)
                r.raise_for_status()
                data = r.json()
                return data.get("response", "")
            except httpx.TimeoutException:
                return "Response timed out. The model may be overloaded. Please try again."
            except Exception as e:
                return f"Error generating response: {str(e)}"

    async def stream(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.2,
        system: Optional[str] = None,
        max_tokens: int = 150,
        timeout: float = 30.0,
    ) -> AsyncGenerator[str, None]:
        await self.ensure_model(model)
        url = f"{self.base_url}/api/generate"
        payload: Dict = {
            "model": model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 2048,  # Smaller context for speed
            },
            "stream": True,
        }
        if system:
            payload["system"] = system
        
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
                async with client.stream("POST", url, json=payload) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            token = obj.get("response")
                            if token:
                                yield token
                        except Exception:
                            continue
        except httpx.TimeoutException:
            yield "Response timed out. The model may be overloaded. Please try again."
        except Exception as e:
            yield f"Error generating response: {str(e)}"


ollama = OllamaClient()
