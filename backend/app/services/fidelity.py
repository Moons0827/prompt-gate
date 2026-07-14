"""처치 충실도(treatment fidelity) 서비스.

세 가지를 지킨다.

1. 불변식 — 교사는 학생 프롬프트를 대신 고칠 수 없다.
   sent_prompt == student_prompt 여야 한다. 어기면 기록한다.

2. 까닭의 형식 — 태그만으로는 부족하다.
   Damen 외(2021, QJEP 74(6), 1054-1069)에서 자기중심 편향이 줄어든 것은
   서사적 피드백을 받은 집단뿐이었고, 정확성 피드백 집단에서는 줄지 않았다.
   태그("[대상]이 부족합니다")는 정확성 신호다. 그것만 주면 학생은
   "네 가지를 넣어야 통과한다"는 규칙을 외울 뿐, 관점 전환은 일어나지 않는다.
   → 교사가 쓴 까닭을 서사형/정확성형으로 분류하여 기록한다.
   → 학급별 서사형 비율이 곧 처치가 제대로 전달되었는지의 지표다.

3. 모델이 되돌림을 대신했는가 — generators/base.py의 guard() 참조.
"""

from __future__ import annotations

import re

from app.models import ReasonType

# ─────────────────────────────────────────────────────────────
# 까닭 유형 분류기
# ─────────────────────────────────────────────────────────────
# 서사형 : 청자에게 무슨 일이 일어나는지를 이야기한다.
#   "1학년이 이 포스터를 보면 '잔반율'에서 멈출 거예요."
#   "AI는 우리 반이 나물을 몇 번 남겼는지 몰라요. 이대로 보내면
#    아무 학교에나 붙일 수 있는 포스터가 올 거예요."
_NARRATIVE = [
    r"(보면|읽으면|받으면|들으면|보고|읽고)",         # 청자의 행위
    r"(할 거|올 거|될 거|나올 거|생길 거|모를 거)",   # 결과 예측
    r"(몰라요|모릅니다|모를|못 알아|처음 들|헷갈)",   # 청자의 지식 상태
    r"(1학년|2학년|3학년|4학년|5학년|6학년|친구|선생님|학부모|옆 반|동생|형|누나|AI|에이아이)",
    r"(이대로|지금 이대로|그대로 보내)",
]
_NARRATIVE_RE = [re.compile(p) for p in _NARRATIVE]

# 정확성형 : 무엇이 부족한지만 말한다.
_ACCURACY = [
    r"(부족|없습니다|없어요|빠졌|빠져|안 적|안 썼|누락)",
    r"\[(상황|대상|조건|목적)\][^\n]{0,6}(부족|없|빠)",
    r"(추가하|넣으세요|보완하|다시 쓰)",
]
_ACCURACY_RE = [re.compile(p) for p in _ACCURACY]


def classify_reason(reason: str) -> tuple[ReasonType, float]:
    """교사가 쓴 까닭을 분류한다.

    반환: (유형, 서사성 점수 0~1)

    휴리스틱이다. 목적은 차단이 아니라 기록과 넛지다.
    최종 판정은 사후 인간 코딩으로 한다 — 이 점수는 그 표본추출의 우선순위를 정하는 데 쓴다.
    """
    text = (reason or "").strip()
    if not text:
        return ReasonType.UNKNOWN, 0.0

    n_hits = sum(1 for r in _NARRATIVE_RE if r.search(text))
    a_hits = sum(1 for r in _ACCURACY_RE if r.search(text))

    # 길이도 신호다. 서사는 짧을 수 없다.
    length_bonus = 0.2 if len(text) >= 25 else 0.0

    score = min(1.0, n_hits / len(_NARRATIVE_RE) + length_bonus)

    if n_hits >= 2 and len(text) >= 20:
        return ReasonType.NARRATIVE, score
    if a_hits > 0 and n_hits < 2:
        return ReasonType.ACCURACY, score
    return ReasonType.UNKNOWN, score


NUDGE = (
    "까닭이 '무엇이 부족한지'만 말하고 있습니다. "
    "그 사람이 이 질문을 받으면 무슨 일이 일어나는지 이야기로 써 주세요.\n"
    "  ✗ \"[대상]이 부족합니다.\"\n"
    "  ○ \"1학년이 이 포스터를 보면 '잔반율'에서 멈출 거예요. "
    "그 아이는 그 말을 들어 본 적이 없거든요.\"\n"
    "(고친 문장을 주지는 마세요. 그것은 학생의 몫입니다.)"
)


# ─────────────────────────────────────────────────────────────
# 불변식
# ─────────────────────────────────────────────────────────────
def check_invariant(student_prompt: str, sent_prompt: str) -> bool:
    """True면 위반. 교사가 학생 프롬프트를 대신 고쳤다는 뜻이다."""
    return student_prompt.strip() != (sent_prompt or "").strip()


# ─────────────────────────────────────────────────────────────
# 편집 거리 (되돌림이 실제로 무엇을 바꿨는가)
# ─────────────────────────────────────────────────────────────
def edit_distance(a: str, b: str) -> int:
    a, b = a or "", b or ""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
