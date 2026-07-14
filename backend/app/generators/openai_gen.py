import httpx

from app.config import require_model, settings
from app.generators.base import SYSTEM_PROMPT, Generator


class OpenAIGenerator(Generator):
    provider = "openai"

    @property
    def model(self) -> str:
        return require_model("OPENAI_MODEL", settings.OPENAI_MODEL)

    @property
    def temperature(self) -> float:
        return settings.TEMPERATURE

    async def _call(self, prompt: str) -> str:
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY 가 비어 있다.")
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        # 모델 계열에 따라 토큰 파라미터 이름이 다르다. 하나가 막히면 다른 것으로 재시도한다.
        async with httpx.AsyncClient(timeout=settings.TIMEOUT_S) as client:
            for token_key in ("max_completion_tokens", "max_tokens"):
                payload = dict(body)
                payload[token_key] = settings.MAX_TOKENS
                payload["temperature"] = self.temperature
                r = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                if r.status_code < 400:
                    return r.json()["choices"][0]["message"]["content"].strip()
                msg = r.text[:300]
                # 파라미터 이름/temperature 미지원이면 다음 조합으로 재시도
                if "max_tokens" in msg or "temperature" in msg:
                    body.pop("temperature", None)
                    continue
                raise RuntimeError(f"OpenAI {r.status_code}: {msg}")
        raise RuntimeError("OpenAI: 토큰 파라미터 조합을 찾지 못했다. 모델 ID를 확인하라.")
