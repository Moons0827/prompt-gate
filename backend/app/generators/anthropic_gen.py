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
        async with httpx.AsyncClient(timeout=settings.TIMEOUT_S) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": settings.MAX_TOKENS,
                    "temperature": self.temperature,
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            if r.status_code >= 400:
                raise RuntimeError(f"Anthropic {r.status_code}: {r.text[:300]}")
            data = r.json()
            return "".join(
                b.get("text", "")
                for b in data.get("content", [])
                if b.get("type") == "text"
            ).strip()
