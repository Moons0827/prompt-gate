"""모델 검증 — 검사지 과제 5의 프롬프트 4개를 실제 모델에 넣어 본다.

이 스크립트가 두 가지를 한 번에 끝낸다.

  1. 모델 선택      — 되묻지 않는 모델을 고른다. 브랜드가 아니라 이것이 1순위다.
  2. 과제 5 확정    — AI 응답 예시가 연구자가 지어낸 것이 아니라 실제 출력이 된다.

실행:
    GENERATOR=anthropic ANTHROPIC_MODEL=... python verify_model.py
    GENERATOR=openai    OPENAI_MODEL=...    python verify_model.py

결과는 verify_<provider>_<model>.json 에 저장된다.
논문에 그대로 싣는다.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys

from app.config import settings
from app.generators.base import SYSTEM_PROMPT, SYSTEM_PROMPT_HASH
from app.generators.registry import get_generator

# ── 검사지 v7 과제 5 (A형: 도서관) ────────────────────────────
FORM_A = {
    "form": "A(도서관)",
    "facts": ["점심시간", "12시 40분", "8명", "3명", "대출증"],
    "cannot": ["여는 시간", "아침", "새 책", "사서"],
    "prompts": {
        "㉠ 맥락 0": "도서관 포스터 문구를 써 줘.",
        "㉡ 부가 질문만": "도서관 포스터 문구를 예쁘고 재미있게, 세 줄로, 이모티콘 넣어서 써 줘.",
        "㉢ 맥락 완비": (
            "우리 학교 도서관은 점심시간(12시 40분~1시 20분)에만 열어. "
            "도서관에 안 가는 우리 반 친구들은 그 시간에 여는 줄도 몰라. "
            "그 친구들이 오고 싶어지도록 포스터 문구를 써 줘."
        ),
        "㉣ 조건 위반": "우리 학교 도서관에 사람이 안 와. 도서관을 아침에도 열도록 계획을 세워 줘.",
    },
}

# ── B형: 운동장 ──────────────────────────────────────────────
FORM_B = {
    "form": "B(운동장)",
    "facts": ["점심시간", "12시 40분", "12명", "4명", "자유 구역"],
    "cannot": ["점심시간", "늘리", "넓히", "축구를 못"],
    "prompts": {
        "㉠ 맥락 0": "운동장 포스터 문구를 써 줘.",
        "㉡ 부가 질문만": "운동장 포스터 문구를 예쁘고 재미있게, 세 줄로, 이모티콘 넣어서 써 줘.",
        "㉢ 맥락 완비": (
            "우리 학교 운동장은 점심시간(12시 40분~1시 20분)에만 나갈 수 있어. "
            "운동장에 안 나가는 우리 반 친구들은 운동장이 축구하는 사람만 쓰는 곳인 줄 알아. "
            "축구를 안 해도 쓸 수 있는 '자유 구역'이 있다는 걸 알려 주는 포스터 문구를 써 줘."
        ),
        "㉣ 조건 위반": "우리 학교 운동장에 사람이 안 나와. 점심시간을 30분 더 늘리는 계획을 세워 줘.",
    },
}

EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\u2600-\u27BF\u2190-\u21FF\u2B00-\u2BFF]"
)


def has_school_fact(text: str, facts: list[str]) -> bool:
    return any(f in text for f in facts)


def check(key: str, text: str, form: dict) -> tuple[bool, str]:
    """각 프롬프트가 검사지 설계대로 동작했는가."""
    fact = has_school_fact(text, form["facts"])
    if key.startswith("㉠"):
        # 맥락 0 → 일반론. 우리 학교 사실이 나오면 안 된다(지어낸 것이므로).
        return (not fact), "일반론이어야 한다. 학교 사실이 나오면 모델이 지어낸 것이다."
    if key.startswith("㉡"):
        # 부가 질문만 → 형식은 지키되 내용은 빈다.
        lines = len([l for l in text.splitlines() if l.strip()])
        emo = bool(EMOJI.search(text))
        ok = (not fact) and (emo or lines >= 2)
        return ok, "형식(이모티콘·줄수)은 지키되 학교 정보는 없어야 한다."
    if key.startswith("㉢"):
        # 맥락 완비 → 우리가 준 사실이 반영되어야 한다.
        return fact, "우리가 준 사실(여는 시간 등)이 답에 나타나야 한다."
    if key.startswith("㉣"):
        # 조건 위반 → 실행 불가능한 계획을 성실히 내야 한다. 거절하면 안 된다.
        refused = any(w in text for w in ["어려", "할 수 없", "불가능", "권장하지"])
        planlike = any(w in text for w in ["단계", "1.", "①", "먼저", "계획"])
        return (planlike and not refused), "거절하지 말고 계획을 내야 한다."
    return True, ""


async def main() -> int:
    gen = get_generator()
    print(f"provider={gen.provider}  model={gen.model}  temp={gen.temperature}")
    print(f"system_prompt_hash={SYSTEM_PROMPT_HASH}\n")

    out = {
        "provider": gen.provider,
        "model": gen.model,
        "temperature": gen.temperature,
        "system_prompt_hash": SYSTEM_PROMPT_HASH,
        "system_prompt": SYSTEM_PROMPT,
        "forms": [],
    }

    asked_back_total = 0
    design_fail = 0
    n = 0

    for form in (FORM_A, FORM_B):
        rows = []
        print(f"── {form['form']} " + "─" * 50)
        for key, prompt in form["prompts"].items():
            r = await gen.generate(prompt)
            n += 1
            if not r.ok:
                print(f"  {key:14} ERROR  {r.error}")
                rows.append({"key": key, "prompt": prompt, "error": r.error})
                design_fail += 1
                continue

            ok, why = check(key, r.text, form)
            if r.guard.asked_back:
                asked_back_total += 1
            if not ok:
                design_fail += 1

            flag = []
            if r.guard.asked_back:
                flag.append("되묻기")
            if r.guard.gave_advice:
                flag.append("조언")
            if r.retried:
                flag.append("재시도" + ("→해소" if r.retry_fixed else "→실패"))
            if not ok:
                flag.append("설계불일치")

            print(f"  {key:14} {'OK' if (ok and not r.guard.contaminated) else '✗ ' + '/'.join(flag)}")
            print(f"    {r.text[:100].replace(chr(10), ' / ')}")
            if not ok:
                print(f"    → {why}")

            rows.append({
                "key": key,
                "prompt": prompt,
                "response": r.text,
                "asked_back": r.guard.asked_back,
                "gave_advice": r.guard.gave_advice,
                "question_ratio": r.guard.question_ratio,
                "retried": r.retried,
                "retry_fixed": r.retry_fixed,
                "design_ok": ok,
                "design_note": "" if ok else why,
                "latency_ms": r.latency_ms,
            })
        out["forms"].append({"form": form["form"], "rows": rows})
        print()

    out["summary"] = {
        "n": n,
        "asked_back": asked_back_total,
        "asked_back_rate": round(asked_back_total / n, 3) if n else None,
        "design_mismatch": design_fail,
        "verdict": (
            "사용 가능" if asked_back_total == 0 and design_fail == 0
            else "사용 불가 — 아래 참조"
        ),
    }

    print("═" * 62)
    print(f"되묻기 {asked_back_total}/{n}   설계 불일치 {design_fail}/{n}")
    if asked_back_total:
        print("⚠ 되묻는 모델은 쓸 수 없다. AI가 되돌림을 대신하면 처치와 통제의 차이가 사라진다.")
        print("  시스템 프롬프트를 강화하거나 다른 모델을 쓴다.")
    if design_fail:
        print("⚠ 설계 불일치 — 과제 5의 정답 연결이 성립하지 않는다.")
        print("  특히 ㉡가 학교 사실을 지어내면 '부가 질문은 내용을 만들지 못한다'는 전제가 깨진다.")
    if not asked_back_total and not design_fail:
        print("✓ 이 모델을 쓸 수 있다. 응답 4개를 검사지 과제 5에 그대로 넣는다.")

    fn = f"verify_{gen.provider}_{gen.model.replace('/', '_')}.json"
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {fn}")
    return 0 if (not asked_back_total and not design_fail) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
