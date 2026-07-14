import httpx

from app.config import require_model, settings
from app.generators.base import SYSTEM_PROMPT, Generator


class AnthropicGenerator(Generator):
    provider = "anthropic"

    @property
    def model(self) -> str:
        return require_model("ANTHROPIC_MODEL", settings.ANTHROPIC_MODEL)

    @property
    def temperature(self) -> float:
        return settings.TEMPERATURE

    async def _call(self, prompt: str) -> str:
        # 배포 환경변수에 딸려 들어간 공백·줄바꿈·따옴표를 제거한다.
        # (키는 멀쩡해도 헤더에 '\n'·따옴표가 섞이면 401 invalid x-api-key)
        api_key = settings.ANTHROPIC_API_KEY.strip().strip('"').strip("'").strip()
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY 가 비어 있다.")
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": settings.MAX_TOKENS,
            "temperature": self.temperature,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        }
        url = "https://api.anthropic.com/v1/messages"
        async with httpx.AsyncClient(timeout=settings.TIMEOUT_S) as client:
            r = await client.post(url, headers=headers, json=payload)
            # 최신 모델은 temperature 를 받지 않는다("deprecated for this model").
            # 그 경우 temperature 를 빼고 한 번 더 호출한다 — 모델을 바꿔도 안 깨진다.
            if r.status_code == 400 and "temperature" in r.text:
                payload.pop("temperature", None)
                r = await client.post(url, headers=headers, json=payload)
            if r.status_code >= 400:
                raise RuntimeError(f"Anthropic {r.status_code}: {r.text[:300]}")
            data = r.json()
            return "".join(
                b.get("text", "")
                for b in data.get("content", [])
                if b.get("type") == "text"
            ).strip()
