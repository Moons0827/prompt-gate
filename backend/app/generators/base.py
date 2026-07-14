"""생성기 인터페이스 · 시스템 프롬프트 · 되묻기 가드.

  ⚠ 모델이 되물으면 연구 설계가 무너진다.

  학생이 "도서관에 사람이 안 와. 방법 알려 줘"를 보냈는데 AI가
  "몇 명이 오나요? 어느 학년인가요?"라고 되물으면 — AI가 되돌림을 대신해 버린다.
  통제 학급(안전 검토만) 학생도 AI에게서 교육적 피드백을 받게 되고,
  처치와 통제의 차이가 사라진다.

  두 겹으로 막는다.
    (1) SYSTEM_PROMPT 로 되묻기·조언·사실 날조를 금지
    (2) guard() 로 탐지 -> 1회 재시도 -> 그래도 되물으면 '기록'

  차단하지 않는다. 기록한다. 되묻기가 몇 %에서 일어났는지를 논문에 써야 한다.

시스템 프롬프트는 처치의 일부다. 논문 부록에 그대로 싣는다.
해시를 매 호출마다 남긴다 -- 도중에 바뀌면 그 시점이 데이터에 드러난다.
"""

from __future__ import annotations

import hashlib
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# ══════════════════════════════════════════════════════════════
# 시스템 프롬프트 (처치의 일부 — 논문 부록에 그대로 싣는다)
# ══════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """너는 초등학교 5·6학년 학생이 학교 문제를 해결하도록 돕는 도우미다.

반드시 지킬 것:

1. 되묻지 마라. 학생에게 추가 정보를 요구하지 마라.
   정보가 부족해도 되묻지 말고, 주어진 것만으로 최선을 다해 답하라.
   "몇 명인가요?", "어느 학년인가요?", "더 자세히 알려 주세요" 같은 말을 하지 마라.

2. 없는 사실을 지어내지 마라. 학생이 주지 않은 학교 사정(인원수, 시간, 규칙 등)을
   추측해서 채워 넣지 마라. 정보가 없으면 일반적인 수준에서만 답하라.

3. 학생의 질문이 부족하다고 지적하지 마라. 조언하지 마라. 가르치려 들지 마라.
   "이렇게 물어보면 더 좋아요" 같은 말을 하지 마라. 그것은 선생님의 몫이다.

4. 요청받은 것만 만들어라. 초등학생이 읽을 수 있는 쉬운 말로 쓰되,
   구체적인 예를 들어 이해하기 쉽게 충분히 설명하라. 한두 문장으로 끝내지 마라.

5. 마크다운 기호를 쓰지 마라. 제목(#), 굵게(**), 구분선(---), 목록 기호(-, *) 없이
   그냥 일반 문장과 문단으로만 써라.

6. 위험하거나 남을 해치는 내용은 답하지 마라.
"""

SYSTEM_PROMPT_HASH = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]

RETRY_SUFFIX = "\n\n(중요: 되묻지 마라. 질문하지 마라. 지금 주어진 것만으로 바로 결과물을 만들어라.)"


# ══════════════════════════════════════════════════════════════
# 되묻기 가드
# ══════════════════════════════════════════════════════════════
#
# 오탐을 반드시 피해야 한다. 이 시스템의 산출물은 대개 '포스터 문구'인데,
# 포스터 문구는 수사의문문 투성이다 -- "책, 심심하지 않나요?", "오늘 뭐 읽지?"
# 이런 것을 되묻기로 잡으면 처치 오염률이 부풀려져 논문이 틀린다.
#
#   HARD : 명시적 정보 요구. 이것 하나만 있어도 되묻기다.
#   SOFT : 학생만 답할 수 있는 사실을 묻는 의문문.
#          단독으로는 판정하지 않는다. 응답이 '질문 위주'일 때만 되묻기로 본다.
#
_HARD = [
    r"(알려|말해|말씀해|설명해|적어|공유해|보내|입력해)\s*주(세요|시겠|실 수|시면|십시오)",
    r"(정보|자료|숫자|내용|맥락)[^\n]{0,8}(더|추가로|좀)[^\n]{0,8}(필요|주세요|알려)",
    r"(더\s*자세히|더\s*구체적으로)[^\n]{0,10}(알려|말해|적어|설명)",
    r"질문이\s*(있|하나)",
    r"확인[^\n]{0,4}주(세요|시)",
]
_HARD_RE = re.compile("|".join(_HARD))

_SOFT = [
    r"몇\s*(명|개|번|퍼센트|프로|%|킬로|kg)",
    r"(어느|어떤|몇)\s*학년",
    r"(어느|어떤)\s*(반|학교|요일)",
    r"얼마나\s*(많|자주|오래)",
]
_SOFT_RE = re.compile("|".join(_SOFT))

_ADVICE = [
    r"(이렇게|다음처럼|아래처럼)\s*(물어|질문)",
    r"질문을\s*(고치|바꾸|다시|수정|보완)",
    r"더\s*좋은\s*(질문|프롬프트)",
    r"(정보|맥락)를\s*추가하(면|세요)",
    r"(빠진|부족한)\s*(것|정보|내용)",
]
_ADVICE_RE = re.compile("|".join(_ADVICE))

_SENT_SPLIT = re.compile(r"[.!?。！？\n]+")


@dataclass
class GuardResult:
    asked_back: bool = False
    gave_advice: bool = False
    hard_hit: bool = False
    soft_hit: bool = False
    question_ratio: float = 0.0
    note: str = ""
    excerpt: str = ""

    @property
    def contaminated(self) -> bool:
        return self.asked_back or self.gave_advice


def _question_ratio(text: str) -> float:
    sents = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    if not sents:
        return 0.0
    qs = len(re.findall(r"[^.!?\n]*[?？]", text))
    return min(1.0, qs / max(1, len(sents)))


def guard(text: str) -> GuardResult:
    """모델이 되돌림을 대신했는지 탐지한다. 차단이 아니라 기록이 목적이다."""
    t = (text or "").strip()
    if not t:
        return GuardResult(note="빈 응답")

    hard = bool(_HARD_RE.search(t))
    soft = bool(_SOFT_RE.search(t)) and ("?" in t or "？" in t)
    qr = _question_ratio(t)

    # SOFT 는 응답이 질문 위주일 때만 되묻기로 본다.
    # 포스터 문구 속 수사의문문을 오탐하지 않기 위함.
    asked = hard or (soft and qr >= 0.5)
    advice = bool(_ADVICE_RE.search(t))

    if hard:
        note = "모델이 정보를 명시적으로 요구했다 — AI가 되돌림을 대신했다."
    elif asked:
        note = "응답이 질문 위주다 — 되묻기로 판단한다."
    elif advice:
        note = "모델이 질문을 고치라고 조언했다 — 교사 역할을 침범했다."
    elif soft:
        note = "의문문이 있으나 질문 위주가 아니다 — 수사의문문으로 보아 통과. (사후 확인 대상)"
    else:
        note = ""

    return GuardResult(
        asked_back=asked, gave_advice=advice, hard_hit=hard, soft_hit=soft,
        question_ratio=round(qr, 2), note=note, excerpt=t[:160],
    )


# ══════════════════════════════════════════════════════════════
# 생성기 인터페이스
# ══════════════════════════════════════════════════════════════
@dataclass
class GenResult:
    text: str = ""
    provider: str = ""
    model: str = ""
    temperature: float = 0.0
    system_prompt_hash: str = SYSTEM_PROMPT_HASH
    latency_ms: int = 0
    guard: GuardResult = field(default_factory=GuardResult)
    retried: bool = False
    retry_fixed: bool = False
    ok: bool = True
    error: str = ""


class Generator(ABC):
    provider: str = "base"

    @abstractmethod
    async def _call(self, prompt: str) -> str: ...

    @property
    @abstractmethod
    def model(self) -> str: ...

    @property
    @abstractmethod
    def temperature(self) -> float: ...

    def _mk(self, t0: float, **kw) -> GenResult:
        return GenResult(
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            **kw,
        )

    async def generate(self, prompt: str) -> GenResult:
        t0 = time.perf_counter()

        try:
            text = await self._call(prompt)
        except Exception as e:  # noqa: BLE001
            return self._mk(t0, ok=False, error=f"{type(e).__name__}: {e}")

        g = guard(text)
        if not g.asked_back:
            return self._mk(t0, text=text, guard=g)

        # 되물었다. 한 번만 더 강하게 요청한다.
        try:
            text2 = await self._call(prompt + RETRY_SUFFIX)
        except Exception as e:  # noqa: BLE001
            return self._mk(
                t0, text=text, guard=g, retried=True,
                error=f"retry failed: {type(e).__name__}: {e}",
            )

        g2 = guard(text2)
        return self._mk(
            t0, text=text2, guard=g2, retried=True, retry_fixed=(not g2.asked_back),
        )
