from app.generators.base import Generator


class NullGenerator(Generator):
    """디버그 전용. 통제 조건이 아니다.

    통제 학급도 반드시 실제 AI 응답을 받는다. 처치와 통제의 차이는
    교사가 유해 내용만 차단하느냐, 교육적으로 되돌리느냐 하나뿐이다.
    생성을 꺼 버리면 그것은 '통제'가 아니라 '다른 수업'이 된다.
    """

    provider = "null"

    @property
    def model(self) -> str:
        return "null-debug"

    @property
    def temperature(self) -> float:
        return 0.0

    async def _call(self, prompt: str) -> str:
        return f"[NULL GENERATOR — 디버그용] 받은 프롬프트 {len(prompt)}자."
