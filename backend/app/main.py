"""prompt-gate — 되돌림 루프 시스템.

  학생 프롬프트 → 교사 게이트 → (통과) AI 전송 / (되돌림) 학생에게 반환

핵심 규칙 (README와 동일, 코드로 강제한다):
  1. 교사는 학생 프롬프트를 대신 고칠 수 없다. 통과 / 되돌림 두 가지뿐.
  2. 되돌릴 때 붙이는 것은 태그 + 까닭뿐. 고친 문장을 주지 않는다.
  3. 까닭은 청자의 관점에서 쓴다 (Damen 외, 2021).
  4. 통제 조건도 AI 응답을 받는다. 차이는 교사가 유해 내용만 차단한다는 점 하나.
  5. 모델이 되물으면 처치가 오염된다 — 탐지하여 기록한다.
"""

from __future__ import annotations

import asyncio
import json
import os
import re

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.generators.registry import get_generator
from app.models import (
    ActivityOption,
    AIResponse,
    Base,
    ClassSetting,
    Classroom,
    Condition,
    Event,
    JudgeItem,
    Judgment,
    PeerReview,
    PromptVersion,
    ReasonType,
    RetraceTag,
    Review,
    Status,
    Submission,
    SurveyResponse,
    Team,
    TeacherAnswer,
    TeamNote,
    TransferPrompt,
)
from app.services.fidelity import (
    NUDGE,
    check_invariant,
    classify_reason,
    edit_distance,
)

# ─────────────────────────────────────────────────────────────
# 차시 → 기능(모드) 설정. 로그인에서 고른 차시가 기능을 고정한다.
#   loop     : 쓰기 → 교사 게이트 → AI (2~5차시 공통 골격)
#   judge    : AI 답 5개 판정표 (1차시, 학생이 프롬프트를 쓰지 않음)
#   retrace  : 자기 질문 역추적 · 자기 태그 (6차시, 교사·AI 없음)
#   peer     : 다른 조 질문에 태그+까닭 (7차시, 되돌림 주체=동료)
#   transfer : 전이 과제 저장만 (8차시, 되돌림·AI·피드백 없음)
# ─────────────────────────────────────────────────────────────
def _judge_gen_prompt(question: str) -> str:
    """조가 쓴 질문에 대해 서로 다른 답 5개를 정해진 형식으로 받도록 감싼다."""
    return (
        "다음 질문에 대해 서로 확실히 다른 관점과 방법으로, 방향이 겹치지 않는 답 "
        "다섯 가지를 만들어 줘.\n"
        "형식 규칙(반드시 지켜라):\n"
        "- 각 답은 '@@1@@ ' 처럼 번호 마커로 시작한다(@@1@@ 부터 @@5@@ 까지).\n"
        "- 한 답은 두세 문장으로, 구체적인 예를 들어 자세히 설명한다.\n"
        "- 제목·머리말·맺음말·구분선(---)·마크다운 기호(#, *)는 절대 쓰지 마라.\n"
        "예시:\n@@1@@ 첫 번째 방법은 이렇게 한다. 예를 들어 …\n"
        "@@2@@ 두 번째 방법은 …\n\n질문: " + question.strip()
    )

# 되돌림 태그 레지스트리. 차시마다 어떤 태그를 쓸지 SESSIONS[n]["tags"]로 고른다.
TAG_LABELS: dict[str, tuple[str, str]] = {
    "tag_situation": ("[상황]", "우리 학교에서 무슨 일이 있었는지 (숫자·사실)"),
    "tag_audience":  ("[대상]", "누가 읽을지, 그 사람이 무엇을 모르는지"),
    "tag_condition": ("[조건]", "우리가 할 수 있는 일 / 할 수 없는 일"),
    "tag_purpose":   ("[목적]", "AI에게 무엇을 만들어 달라는 것인지"),
    "tag_role":      ("[역할]", "AI에게 어떤 역할(누구)이 되어 답하라고 했는지"),
    "tag_example":   ("[예시]", "원하는 답의 예시를 보여 주었는지"),
}
BASE_TAGS = ["tag_situation", "tag_audience", "tag_condition", "tag_purpose"]

SESSIONS: dict[int, dict] = {
    1: {"mode": "judge", "title": "AI는 우리 학교를 모른다",
        "question": "AI가 우리 학교 급식 문제를 해결해 줄 수 있을까?",
        "intro": "세계에는 먹을 것이 부족해 굶주리는 사람들이 많습니다. 하지만 우리 학교에서는 "
                 "먹지 않고 버리는 급식이 생기고 있습니다. 우리는 기아 문제에 관심을 가지고, 먹을 "
                 "만큼만 급식을 받아 남김없이 먹는 '급식 잔반 줄이기 프로젝트'를 실천하려고 합니다. "
                 "우리가 할 수 있는 방법을 찾아보기 위해 AI와 함께 이 프로젝트를 진행합니다.",
        "teacher_compare": True},   # 활동3: 교사의 상세 프롬프트 답과 비교
    2: {"mode": "loop", "title": "급식을 가장 많이 남기는 친구",
        "question": "급식을 가장 많이 남기는 친구는 누구일까?",
        "intro": "우리는 AI에게 우리의 정보를 주지 않고 질문을 하면 원하는 답변을 얻기 어렵다는 것을 "
                 "배웠습니다. 그렇다면 어떤 정보를 줘야 AI에게 원하는 답변을 받을 수 있을까요? "
                 "그냥 정보를 많이 제공하면 될까요?",
        "placeholder": "우리 반 급식 잔반을 3일 동안 조사하려고 해요. "
                       "어떻게 조사하면 좋을지 물어보세요. (우리 반 상황을 알려 주는 것을 잊지 마세요)",
        "answers": 5,      # 통과 시 AI가 5가지로 답한다
        "options": True},  # 활동3: 통과 답 5개를 O/X 적합 판정 + 최종 선정
    3: {"mode": "data", "title": "데이터 한 스푼",
        "question": "우리가 조사한 잔반 데이터를 넣으면 AI 답이 어떻게 달라질까?",
        "intro": "잔반 조사 데이터의 유형(숫자·이유별·친구의 말)과 관계없는 정보를 구분하고, "
                 "꼭 필요한 데이터를 골라 질문에 넣어 AI 답이 어떻게 달라지는지 비교합니다."},
    4: {"mode": "loop", "title": "우리 아이디어 vs AI 아이디어",
        "question": "우리 아이디어와 AI 아이디어, 어느 쪽이 나을까?",
        "placeholder": "우리가 '할 수 있는 일 / 없는 일'을 조건으로 넣어 아이디어를 물어보세요."},
    5: {"mode": "loop", "title": "같은 학교인데 왜 못 알아들을까",
        "question": "같은 학교에 다니는데도, 왜 어떤 사람은 우리 포스터를 못 알아들을까?",
        "placeholder": "그 사람이 무엇을 모르는지 + 우리 반 자료를 넣어 포스터 문구를 부탁하세요.",
        "selfcheck": [
            "그 사람이 무엇을 모르는지 적었는가?",
            "우리 반 자료(숫자)를 넣었는가?",
            "무엇을 하게 하고 싶은지 적었는가?",
        ]},
    6: {"mode": "retrace", "title": "스티커가 왜 안 붙었을까",
        "question": "스티커가 왜 그만큼밖에 안 붙었을까? 우리 질문을 다시 보자."},
    7: {"mode": "peer", "title": "남의 질문에서 빠진 것 찾기",
        "question": "다른 모둠의 질문에서 빠진 것을 찾아 주자."},
    8: {"mode": "transfer", "title": "새 문제에 혼자",
        "question": "새로운 문제에 혼자서 질문해 보자. (우리 학교 전기 절약)"},
}
LOOP_SESSIONS = {n for n, c in SESSIONS.items() if c["mode"] == "loop"}

# 3차시 — 조사 데이터 카드(활동1) · 비교 질문 3개(활동2)
DATA3_CARDS = [
    {"id": 1, "text": "우리 반 학생은 24명이다.", "type": "숫자"},
    {"id": 2, "text": "오늘 9명이 급식을 남겼다.", "type": "숫자"},
    {"id": 3, "text": "9명 중 6명은 양이 많아서 남겼다.", "type": "이유별"},
    {"id": 4, "text": "2명은 좋아하지 않는 음식이어서 남겼다.", "type": "이유별"},
    {"id": 5, "text": "1명은 먹을 시간이 부족해서 남겼다.", "type": "이유별"},
    {"id": 6, "text": "한 학생이 '밥을 조금만 받고 싶어요.'라고 말했다.", "type": "친구의 말"},
    {"id": 7, "text": "오늘 날씨는 흐리다.", "type": "관계없음"},
    {"id": 8, "text": "우리 반은 체육을 좋아한다.", "type": "관계없음"},
    {"id": 9, "text": "교실 뒤에 사물함이 있다.", "type": "관계없음"},
]
DATA3_QUESTIONS = [
    {"key": "가", "label": "질문 가 · 조사 내용 없음",
     "q": "우리 반의 급식 잔반을 줄이는 방법을 알려 줘."},
    {"key": "나", "label": "질문 나 · 숫자만 있음",
     "q": "오늘 우리 반 24명 중 9명이 급식을 남겼어. 잔반을 줄이는 방법을 알려 줘."},
    {"key": "다", "label": "질문 다 · 숫자와 원인이 있음",
     "q": "오늘 우리 반 24명 중 9명이 급식을 남겼고, 그중 6명은 양이 많아서 남겼다고 답했어. "
          "이 조사 결과를 바탕으로 우리 반의 급식 잔반을 줄이는 방법을 알려 줘."},
]
# 3차시 도입 '오늘 급식 조사'의 '왜' 선택지
DATA3_WHY = ["양이 많아서", "좋아하는 음식이 아니어서", "먹을 시간이 부족해서", "기타"]

# 태그를 쓰는 차시(되돌림 있는 차시)에 기본 4태그를 붙인다.
for _n, _cfg in SESSIONS.items():
    if _cfg["mode"] in ("loop", "retrace", "peer"):
        _cfg.setdefault("tags", list(BASE_TAGS))
# 2차시(역할 부여 차시)는 6태그 — 목적·상황·대상·조건·역할·예시
SESSIONS[2]["tags"] = [
    "tag_purpose", "tag_situation", "tag_audience", "tag_condition", "tag_role", "tag_example",
]

engine = create_engine(
    settings.DB_URL, connect_args={"check_same_thread": False}
)
with engine.connect() as conn:
    conn.exec_driver_sql("PRAGMA journal_mode=WAL")
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(engine)

app = FastAPI(title="prompt-gate")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def db() -> Session:
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


def log(s: Session, kind: str, submission_id: int | None = None, **payload) -> None:
    s.add(
        Event(
            kind=kind,
            submission_id=submission_id,
            payload=json.dumps(payload, ensure_ascii=False),
        )
    )


# ─────────────────────────────────────────────────────────────
# 조 (로그인·제출·대화의 단위 — 조원 전체가 공유한다)
# ─────────────────────────────────────────────────────────────
@app.get("/api/config/sessions")
def session_config():
    """차시별 기능(모드) 설정. 프런트가 이걸로 차시별 화면·태그를 고른다."""
    out = []
    for n, cfg in sorted(SESSIONS.items()):
        d = {"no": n, **cfg}
        if "tags" in cfg:  # 태그 키를 라벨·설명이 붙은 객체로 펼친다
            d["tags"] = [
                {"key": k, "label": TAG_LABELS[k][0], "desc": TAG_LABELS[k][1]} for k in cfg["tags"]
            ]
        out.append(d)
    return {"sessions": out}


@app.get("/api/classrooms")
def classrooms(s: Session = Depends(db)):
    """조별 로그인 화면용. 학급과 그 안의 조 목록을 준다."""
    out = []
    for cr in s.scalars(select(Classroom).order_by(Classroom.id)).all():
        out.append(
            {
                "classroom_id": cr.id,
                "school": cr.school,
                "name": cr.name,
                "condition": cr.condition.value,
                # 교사가 연 차시. 0 = 전체 열림(제한 없음). 학생은 이 차시만 할 수 있다.
                "active_session": int(_cs_get(s, cr.id, "active_session") or 0),
                "teams": [
                    {"team_id": t.id, "number": t.number}
                    for t in sorted(cr.teams, key=lambda t: t.number)
                ],
            }
        )
    return out


class ActiveSessionIn(BaseModel):
    session_no: int = Field(ge=0, le=8)   # 0 = 전체 열기


@app.post("/api/teacher/active-session/{classroom_id}")
def set_active_session(classroom_id: int, body: ActiveSessionIn, s: Session = Depends(db)):
    """교사가 학생에게 열 차시를 정한다(0=전체). 학생은 이 차시만 접속·활동한다."""
    cr = s.get(Classroom, classroom_id)
    if not cr:
        raise HTTPException(404, "없는 학급")
    _cs_set(s, classroom_id, "active_session", str(body.session_no))
    log(s, "set_active_session", classroom=classroom_id, value=body.session_no)
    s.commit()
    return {"ok": True, "active_session": body.session_no}


class SubmitIn(BaseModel):
    team_id: int
    session_no: int = Field(ge=1, le=8)
    prompt: str = Field(min_length=1)


@app.post("/api/team/submit")
def submit(body: SubmitIn, s: Session = Depends(db)):
    """조의 새 프롬프트 제출, 또는 되돌아온 것을 고쳐서 재제출.

    조 단위 루프이므로 이 게이트는 조 전체에 걸린다 — 한 조원의 질문이
    교사 검토 중이면 같은 조의 다른 조원도 새 질문을 낼 수 없다.
    """
    if body.session_no not in LOOP_SESSIONS:
        raise HTTPException(
            400,
            f"{body.session_no}차시는 되돌림 루프를 쓰지 않습니다. "
            "이 차시의 전용 활동 화면을 쓰세요.",
        )
    sub = s.scalar(
        select(Submission)
        .where(
            Submission.team_id == body.team_id,
            Submission.session_no == body.session_no,
        )
        .order_by(Submission.id.desc())
    )
    if sub is None or sub.status in (Status.PASSED, Status.BLOCKED):
        sub = Submission(team_id=body.team_id, session_no=body.session_no)
        s.add(sub)
        s.flush()
        v_no = 1
        dist = 0
    else:
        if sub.status != Status.RETURNED:
            raise HTTPException(409, "아직 교사 검토 중입니다.")
        prev = sub.versions[-1]
        v_no = prev.version + 1
        dist = edit_distance(prev.student_prompt, body.prompt)
        sub.status = Status.PENDING

    v = PromptVersion(
        submission_id=sub.id,
        version=v_no,
        student_prompt=body.prompt.strip(),
        edit_distance=dist,
    )
    s.add(v)
    log(s, "submit", sub.id, version=v_no, edit_distance=dist, chars=len(body.prompt))
    s.commit()
    return {"submission_id": sub.id, "version": v_no, "status": sub.status.value}


def _tags_of(r: Review) -> list[str]:
    return [
        t
        for t, on in [
            ("[상황]", r.tag_situation),
            ("[대상]", r.tag_audience),
            ("[조건]", r.tag_condition),
            ("[목적]", r.tag_purpose),
            ("[역할]", r.tag_role),
            ("[예시]", r.tag_example),
        ]
        if on
    ]


@app.get("/api/team/{team_id}/session/{session_no}")
def team_state(team_id: int, session_no: int, s: Session = Depends(db)):
    """조 공유 화면. 같은 조원이 이 조의 모든 프롬프트·되돌림·AI 답을 다 본다.

    threads : 이 조가 이 차시에 만든 되돌림 루프 전부(오래된 순).
    active  : 지금 편집 가능한 루프(pending/returned). 없으면 새 질문을 낸다.
    """
    team = s.get(Team, team_id)
    if not team:
        raise HTTPException(404, "없는 조")

    subs = s.scalars(
        select(Submission)
        .where(Submission.team_id == team_id, Submission.session_no == session_no)
        .order_by(Submission.id)
    ).all()

    threads = []
    active_id, active_status = None, None
    for sub in subs:
        versions = []
        for v in sub.versions:
            item = {
                "version": v.version,
                "prompt": v.student_prompt,
                "returned": False,
                "tags": [],
                "reason": "",
                "ai_text": v.response.text if v.response else "",
            }
            if v.review and v.review.decision == Status.RETURNED:
                item["returned"] = True
                item["tags"] = _tags_of(v.review)
                item["reason"] = v.review.reason
            versions.append(item)
        threads.append(
            {
                "submission_id": sub.id,
                "status": sub.status.value,
                "return_count": sub.return_count,
                "versions": versions,
            }
        )
        if sub.status in (Status.PENDING, Status.RETURNED):
            active_id, active_status = sub.id, sub.status.value

    return {
        "team_no": team.number,
        "condition": team.classroom.condition.value,
        "threads": threads,
        "active_id": active_id,
        "active_status": active_status,
    }


# ─────────────────────────────────────────────────────────────
# 교사
# ─────────────────────────────────────────────────────────────
@app.get("/api/teacher/queue/{classroom_id}")
def queue(classroom_id: int, s: Session = Depends(db)):
    rows = s.execute(
        select(Submission, PromptVersion, Team, Classroom)
        .join(Team, Team.id == Submission.team_id)
        .join(Classroom, Classroom.id == Team.classroom_id)
        .join(PromptVersion, PromptVersion.submission_id == Submission.id)
        .where(Classroom.id == classroom_id, Submission.status == Status.PENDING)
        .order_by(Submission.id, PromptVersion.version.desc())
    ).all()
    seen, out = set(), []
    for sub, v, tm, cr in rows:
        if sub.id in seen:
            continue
        seen.add(sub.id)
        out.append(
            {
                "submission_id": sub.id,
                "version_id": v.id,
                "team_no": tm.number,
                "condition": cr.condition.value,
                "session_no": sub.session_no,
                "return_count": sub.return_count,
                "version": v.version,
                "prompt": v.student_prompt,
            }
        )
    return out


class ReviewIn(BaseModel):
    version_id: int
    decision: str  # passed | returned | blocked
    teacher_id: str = ""
    tag_situation: bool = False
    tag_audience: bool = False
    tag_condition: bool = False
    tag_purpose: bool = False
    tag_role: bool = False       # [역할] — 2차시에서만 쓴다
    tag_example: bool = False    # [예시] — 2차시에서만 쓴다
    reason: str = ""
    # 교사가 프롬프트를 대신 고치려 시도하면 여기 값이 온다 — 거부하고 기록한다.
    edited_prompt: str | None = None


@app.post("/api/teacher/review")
async def review(body: ReviewIn, s: Session = Depends(db)):
    v = s.get(PromptVersion, body.version_id)
    if not v:
        raise HTTPException(404, "없는 버전")
    sub = v.submission
    classroom = s.get(Team, sub.team_id).classroom
    cond = classroom.condition

    # ── 불변식 1 : 교사는 대신 고칠 수 없다
    if body.edited_prompt is not None and check_invariant(
        v.student_prompt, body.edited_prompt
    ):
        v.invariant_violated = True
        log(
            s,
            "INVARIANT_VIOLATION",
            sub.id,
            note="교사가 학생 프롬프트를 대신 고치려 했다",
            attempted=body.edited_prompt,
        )
        s.commit()
        raise HTTPException(
            403,
            "교사는 학생의 프롬프트를 대신 고칠 수 없습니다. "
            "통과 또는 되돌림만 가능합니다. (시도가 기록되었습니다)",
        )

    try:
        dec = Status(body.decision)
    except ValueError:
        raise HTTPException(400, "decision은 passed / returned / blocked 중 하나여야 합니다.")

    # ── 통제 조건 : 통과 / 차단(유해)만 가능
    if cond == Condition.CONTROL and dec == Status.RETURNED:
        raise HTTPException(
            400,
            "통제 조건에서는 교육적 되돌림을 할 수 없습니다. "
            "통과 또는 유해 차단만 가능합니다.",
        )

    # ── 되돌림 횟수 제한(교사 설정, 0=무제한)
    if dec == Status.RETURNED and classroom.max_returns and sub.return_count >= classroom.max_returns:
        raise HTTPException(
            400,
            f"이 조는 되돌림을 이미 {sub.return_count}번 했습니다"
            f"(최대 {classroom.max_returns}번). 이제 통과시켜 주세요.",
        )

    rtype, rscore = ReasonType.UNKNOWN, 0.0
    if dec == Status.RETURNED:
        if not any(
            [
                body.tag_situation,
                body.tag_audience,
                body.tag_condition,
                body.tag_purpose,
                body.tag_role,
                body.tag_example,
            ]
        ):
            raise HTTPException(400, "되돌리려면 태그를 하나 이상 붙여야 합니다.")
        # 까닭은 선택 — 태그만으로도 되돌릴 수 있다. 쓰면 서사성으로 분류한다.
        if body.reason.strip():
            rtype, rscore = classify_reason(body.reason)

    rev = Review(
        version_id=v.id,
        decision=dec,
        tag_situation=body.tag_situation,
        tag_audience=body.tag_audience,
        tag_condition=body.tag_condition,
        tag_purpose=body.tag_purpose,
        tag_role=body.tag_role,
        tag_example=body.tag_example,
        reason=body.reason.strip(),
        reason_type=rtype,
        reason_score=rscore,
        teacher_id=body.teacher_id,
    )
    s.add(rev)
    sub.status = dec

    result = {"decision": dec.value}

    if dec == Status.RETURNED:
        sub.return_count += 1
        result["reason_type"] = rtype.value
        if rtype == ReasonType.ACCURACY:
            # 차단하지 않는다. 넛지만 한다. 그리고 기록한다.
            result["nudge"] = NUDGE
        log(
            s,
            "return",
            sub.id,
            return_count=sub.return_count,
            reason_type=rtype.value,
            reason_score=round(rscore, 2),
            tags=[
                t
                for t, on in [
                    ("상황", body.tag_situation),
                    ("대상", body.tag_audience),
                    ("조건", body.tag_condition),
                    ("목적", body.tag_purpose),
                    ("역할", body.tag_role),
                    ("예시", body.tag_example),
                ]
                if on
            ],
        )
        s.commit()
        return result

    if dec == Status.BLOCKED:
        log(s, "blocked", sub.id, reason=body.reason[:200])
        s.commit()
        return result

    # ── 통과 : 학생 프롬프트를 그대로 전송한다 (sent_prompt는 항상 원문)
    v.sent_prompt = v.student_prompt
    v.invariant_violated = check_invariant(v.student_prompt, v.sent_prompt)

    # 일부 차시는 최종 답을 5가지로 받는다(SESSIONS[n]["answers"]). 원문은 그대로 두고,
    # 생성 호출에만 '다섯 가지로 답하라'는 형식 지시를 얹는다.
    n_answers = SESSIONS.get(sub.session_no, {}).get("answers", 1)
    gen_prompt = _judge_gen_prompt(v.sent_prompt) if n_answers >= 5 else v.sent_prompt

    gen = get_generator()
    g = await gen.generate(gen_prompt)

    if not g.ok:
        # 생성 실패를 조용히 넘기지 않는다. 학생에게 빈 답을 보여 주면 안 된다.
        log(s, "GENERATION_ERROR", sub.id, error=g.error, model=g.model)
        sub.status = Status.PENDING          # 통과를 되돌린다
        v.sent_prompt = None
        s.commit()
        raise HTTPException(502, f"AI 호출 실패: {g.error}")

    resp = AIResponse(
        version_id=v.id,
        text=g.text,
        provider=g.provider,
        model=g.model,
        temperature=g.temperature,
        system_prompt_hash=g.system_prompt_hash,
        latency_ms=g.latency_ms,
        asked_back=g.guard.asked_back,
        gave_advice=g.guard.gave_advice,
        retried=g.retried,
        retry_fixed=g.retry_fixed,
        error=g.error,
    )
    s.add(resp)

    # 활동3(2차시 등): 통과한 답을 항목으로 쪼개 조가 O/X 적합 판정하게 한다.
    # 새 통과가 들어오면 그 조·차시의 이전 옵션을 지우고 최신 답으로 다시 만든다.
    if SESSIONS.get(sub.session_no, {}).get("options"):
        for old in s.scalars(
            select(ActivityOption).where(
                ActivityOption.team_id == sub.team_id,
                ActivityOption.session_no == sub.session_no,
            )
        ).all():
            s.delete(old)
        for i, txt in enumerate(_split_five(g.text), 1):
            s.add(ActivityOption(
                team_id=sub.team_id, session_no=sub.session_no, idx=i, text=txt,
            ))

    log(
        s,
        "passed",
        sub.id,
        model=g.model,
        provider=g.provider,
        temperature=g.temperature,
        sys_hash=g.system_prompt_hash,
        latency_ms=g.latency_ms,
        retried=g.retried,
        error=g.error,
    )

    # ⚠ 모델이 되돌림을 대신했다 — 처치 오염
    if g.guard.contaminated:
        log(
            s,
            "FIDELITY_ALERT_MODEL_ASKED_BACK",
            sub.id,
            note=g.guard.note,
            asked_back=g.guard.asked_back,
            gave_advice=g.guard.gave_advice,
            model=g.model,
            excerpt=g.text[:200],
        )
        result["fidelity_alert"] = g.guard.note

    result["ai_text"] = g.text
    s.commit()
    return result


# ─────────────────────────────────────────────────────────────
# 1차시 — AI 답 5개 판정표 (judge). 학생은 프롬프트를 쓰지 않는다.
# ─────────────────────────────────────────────────────────────
def _split_five(text: str) -> list[str]:
    """모델 응답을 최대 5개 답으로 쪼갠다.

    답이 길어지면 모델이 마크다운(#, ---, **)을 섞어 내보낸다. 그것을 걷어내고
    우리가 지정한 '@@N@@' 마커 기준으로 나눈다. 마커가 없으면 번호(1.)로 폴백."""
    t = (text or "").replace("**", "").replace("__", "")
    t = re.sub(r"(?m)^\s*#{1,6}\s*", "", t)      # 헤더(#) 기호 제거
    t = re.sub(r"(?m)^\s*-{3,}\s*$", "", t)       # 구분선(---) 줄 제거

    def clean(p: str) -> str:
        return re.sub(r"[ \t]*\n[ \t]*", " ", p).strip(" \t\n-–—")

    if re.search(r"@@\s*\d+\s*@@", t):
        # 마커 뒤 내용만 뽑는다 — 마커 앞 머리말은 통째로 무시된다.
        parts = re.findall(r"@@\s*\d+\s*@@\s*(.*?)(?=@@\s*\d+\s*@@|$)", t, re.S)
    else:
        parts = re.split(r"(?m)^\s*\d+[\.\)]\s*", t)
        if len(parts) >= 2:
            parts = parts[1:]   # 첫 번호(1.) 앞 머리말·제목은 버린다
    parts = [clean(p) for p in parts]
    parts = [p for p in parts if p]
    return parts[:5] if parts else [t.strip()]


class JudgeGenIn(BaseModel):
    team_id: int
    prompt: str = Field(min_length=1)   # 조가 AI에게 물어볼 질문


# 조별 생성 잠금 — 같은 조원이 '물어보기'를 동시에 눌러도 답이 중복(10개) 생성되지 않게.
_judge_locks: dict[int, asyncio.Lock] = {}


def _judge_items_out(items):
    return {"question": items[0].question if items else "",
            "items": [{"index": it.idx, "text": it.text} for it in items]}


@app.post("/api/team/judge/generate")
async def judge_generate(body: JudgeGenIn, s: Session = Depends(db)):
    """조가 쓴 질문을 AI에 보내 답 5개를 만든다(조 공유). 이미 있으면 그대로 준다.

    같은 조원이 동시에 눌러도 조당 한 번만 생성한다(잠금 + 재확인)."""
    team = s.get(Team, body.team_id)
    if not team:
        raise HTTPException(404, "없는 조")

    def existing():
        return s.scalars(
            select(JudgeItem).where(JudgeItem.team_id == body.team_id).order_by(JudgeItem.idx)
        ).all()

    have = existing()
    if have:
        return _judge_items_out(have)

    lock = _judge_locks.setdefault(body.team_id, asyncio.Lock())
    async with lock:
        # 잠금 안에서 다시 확인 — 먼저 들어온 요청이 이미 만들었을 수 있다.
        have = existing()
        if have:
            return _judge_items_out(have)

        question = body.prompt.strip()
        gen = get_generator()
        g = await gen.generate(_judge_gen_prompt(question))
        if not g.ok:
            raise HTTPException(502, f"AI 호출 실패: {g.error}")
        items = _split_five(g.text)
        for i, t in enumerate(items, 1):
            s.add(JudgeItem(
                team_id=body.team_id, question=question, idx=i, text=t,
                provider=g.provider, model=g.model, system_prompt_hash=g.system_prompt_hash,
            ))
        log(s, "judge_generate", model=g.model, provider=g.provider,
            count=len(items), chars=len(question))
        s.commit()
        return {"question": question,
                "items": [{"index": i, "text": t} for i, t in enumerate(items, 1)]}


@app.get("/api/teacher/settings/{classroom_id}")
def teacher_settings_get(classroom_id: int, s: Session = Depends(db)):
    cr = s.get(Classroom, classroom_id)
    if not cr:
        raise HTTPException(404, "없는 학급")
    return {"max_returns": cr.max_returns}


class SettingsIn(BaseModel):
    max_returns: int = Field(ge=0, le=20)


@app.post("/api/teacher/settings/{classroom_id}")
def teacher_settings_set(classroom_id: int, body: SettingsIn, s: Session = Depends(db)):
    """되돌림 최대 횟수 설정(0=무제한)."""
    cr = s.get(Classroom, classroom_id)
    if not cr:
        raise HTTPException(404, "없는 학급")
    cr.max_returns = body.max_returns
    log(s, "set_max_returns", value=body.max_returns)
    s.commit()
    return {"ok": True, "max_returns": cr.max_returns}


@app.get("/api/teacher/judge-questions/{classroom_id}")
def teacher_judge_questions(classroom_id: int, s: Session = Depends(db)):
    """1차시: 각 조가 AI에게 물어본 질문을 교사가 확인한다(답 개수도 함께)."""
    rows = s.execute(
        select(Team.id, Team.number, JudgeItem.question, func.count(JudgeItem.id))
        .join(JudgeItem, JudgeItem.team_id == Team.id)
        .where(Team.classroom_id == classroom_id)
        .group_by(Team.id)
        .order_by(Team.number)
    ).all()
    return [{"team_id": tid, "team_no": n, "question": q, "answers": cnt}
            for tid, n, q, cnt in rows]


@app.get("/api/team/{team_id}/judge")
def team_judge_get(team_id: int, s: Session = Depends(db)):
    """조 판정 화면: 조가 쓴 질문 + 그 질문에 대한 AI 답 5개 + 이 조가 이미 매긴 판정."""
    team = s.get(Team, team_id)
    if not team:
        raise HTTPException(404, "없는 조")
    items = s.scalars(
        select(JudgeItem).where(JudgeItem.team_id == team_id).order_by(JudgeItem.idx)
    ).all()
    saved = {
        j.item_index: j
        for j in s.scalars(select(Judgment).where(Judgment.team_id == team_id)).all()
    }
    # 활동3: 교사의 상세 프롬프트 답(학급 공유) — '전송(published)'된 것만 학생에게 보인다.
    ta = s.scalar(select(TeacherAnswer).where(
        TeacherAnswer.classroom_id == team.classroom_id, TeacherAnswer.published.is_(True)))
    compare = s.scalar(
        select(TeamNote).where(
            TeamNote.team_id == team_id, TeamNote.session_no == 1, TeamNote.key == "compare"
        )
    )
    return {
        "question": items[0].question if items else "",
        "ready": len(items) > 0,       # 조가 아직 질문을 안 보냈으면 false
        "items": [
            {
                "index": it.idx, "text": it.text,
                "verdict": saved[it.idx].verdict if it.idx in saved else None,
                "reason": saved[it.idx].reason if it.idx in saved else "",
            }
            for it in items
        ],
        "teacher_answer": ({"prompt": ta.prompt, "text": ta.text} if ta else None),
        "compare": (compare.text if compare else ""),
    }


# ── 1차시 활동3: 교사의 자세한 프롬프트 답(학급 공유) ──────────────────
class TeacherAnswerIn(BaseModel):
    classroom_id: int
    prompt: str = Field(min_length=1)


@app.post("/api/teacher/judge/teacher-answer")
async def teacher_answer_gen(body: TeacherAnswerIn, s: Session = Depends(db)):
    """교사가 자세한 프롬프트로 답을 만든다(아직 학생에겐 안 보임 — published=False).

    교사가 확인한 뒤 '전송'을 눌러야 학생에게 공개된다. 다시 누르면 새로 만든다."""
    cr = s.get(Classroom, body.classroom_id)
    if not cr:
        raise HTTPException(404, "없는 학급")
    gen = get_generator()
    g = await gen.generate(body.prompt.strip())
    if not g.ok:
        raise HTTPException(502, f"AI 호출 실패: {g.error}")
    for old in s.scalars(
        select(TeacherAnswer).where(TeacherAnswer.classroom_id == body.classroom_id)
    ).all():
        s.delete(old)
    s.add(TeacherAnswer(
        classroom_id=body.classroom_id, prompt=body.prompt.strip(), text=g.text,
        published=False,
        provider=g.provider, model=g.model, system_prompt_hash=g.system_prompt_hash,
    ))
    log(s, "teacher_answer", model=g.model, provider=g.provider)
    s.commit()
    return {"prompt": body.prompt.strip(), "text": g.text, "published": False}


@app.post("/api/teacher/judge/teacher-answer/{classroom_id}/publish")
def teacher_answer_publish(classroom_id: int, s: Session = Depends(db)):
    """만들어 둔 교사 답을 학생에게 전송(공개)한다."""
    ta = s.scalar(select(TeacherAnswer).where(TeacherAnswer.classroom_id == classroom_id))
    if not ta:
        raise HTTPException(404, "먼저 AI 답을 만들어 주세요.")
    ta.published = True
    log(s, "teacher_answer_publish")
    s.commit()
    return {"ok": True, "published": True}


@app.get("/api/teacher/judge/teacher-answer/{classroom_id}")
def teacher_answer_status(classroom_id: int, s: Session = Depends(db)):
    """교사 화면용: 현재 만들어 둔 교사 답과 전송 여부."""
    ta = s.scalar(select(TeacherAnswer).where(TeacherAnswer.classroom_id == classroom_id))
    if not ta:
        return {"exists": False}
    return {"exists": True, "prompt": ta.prompt, "text": ta.text, "published": ta.published}


class TextIn(BaseModel):
    text: str


@app.post("/api/teacher/judge/teacher-answer/{classroom_id}/text")
def teacher_answer_edit_text(classroom_id: int, body: TextIn, s: Session = Depends(db)):
    """교사가 활동3 AI 답을 직접 수정한다(전송 여부는 그대로)."""
    ta = s.scalar(select(TeacherAnswer).where(TeacherAnswer.classroom_id == classroom_id))
    if not ta:
        raise HTTPException(404, "먼저 AI 답을 만들어 주세요.")
    ta.text = body.text.strip()
    log(s, "teacher_answer_edit")
    s.commit()
    return {"ok": True, "published": ta.published}


class JudgeItemEditIn(BaseModel):
    team_id: int
    idx: int
    text: str


@app.post("/api/teacher/judge/item")
def teacher_edit_judge_item(body: JudgeItemEditIn, s: Session = Depends(db)):
    """교사가 1차시 AI 답(5개 중 하나)을 수정한다. 학생 화면에 바로 반영된다."""
    it = s.scalar(select(JudgeItem).where(
        JudgeItem.team_id == body.team_id, JudgeItem.idx == body.idx))
    if not it:
        raise HTTPException(404, "없는 답")
    it.text = body.text.strip()
    log(s, "teacher_edit_judge_item", team=body.team_id, idx=body.idx)
    s.commit()
    return {"ok": True}


# ── 조 자유 서술 메모 (1차시 비교, 2차시 최종선정 등) ──────────────────
class NoteIn(BaseModel):
    team_id: int
    session_no: int
    key: str = Field(min_length=1, max_length=20)
    text: str = ""


@app.post("/api/team/note")
def team_note(body: NoteIn, s: Session = Depends(db)):
    row = s.scalar(
        select(TeamNote).where(
            TeamNote.team_id == body.team_id,
            TeamNote.session_no == body.session_no,
            TeamNote.key == body.key,
        )
    )
    if row:
        row.text = body.text.strip()
    else:
        s.add(TeamNote(
            team_id=body.team_id, session_no=body.session_no,
            key=body.key, text=body.text.strip(),
        ))
    s.commit()
    return {"ok": True}


# ── 3차시 — 데이터 카드 · 3질문 비교 · 데이터 넣어 고쳐쓰기 ──────────────
def _n3_get(s: Session, team_id: int, key: str) -> str:
    row = s.scalar(select(TeamNote).where(
        TeamNote.team_id == team_id, TeamNote.session_no == 3, TeamNote.key == key))
    return row.text if row else ""


def _n3_set(s: Session, team_id: int, key: str, text: str) -> None:
    row = s.scalar(select(TeamNote).where(
        TeamNote.team_id == team_id, TeamNote.session_no == 3, TeamNote.key == key))
    if row:
        row.text = text
    else:
        s.add(TeamNote(team_id=team_id, session_no=3, key=key, text=text))


_data3_locks: dict[int, asyncio.Lock] = {}

# 교사가 정보 카드를 입력하지 않았을 때 보여 줄 기본 예시(과정안 예시).
DATA3_DEFAULT_CARDS_TEXT = "\n".join(f"[{c['type']}] {c['text']}" for c in DATA3_CARDS)


def _parse_cards(text: str) -> list[dict]:
    """한 줄에 카드 하나. '[유형] 내용' 형식이면 유형 배지를 붙인다."""
    cards = []
    for i, line in enumerate((text or "").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\[(.+?)\]\s*(.+)$", line)
        if m:
            cards.append({"id": i, "type": m.group(1).strip(), "text": m.group(2).strip()})
        else:
            cards.append({"id": i, "type": "", "text": line})
    return cards


def _cs_get(s: Session, classroom_id: int, key: str) -> str:
    row = s.scalar(select(ClassSetting).where(
        ClassSetting.classroom_id == classroom_id, ClassSetting.key == key))
    return row.text if row else ""


def _cs_set(s: Session, classroom_id: int, key: str, text: str) -> None:
    row = s.scalar(select(ClassSetting).where(
        ClassSetting.classroom_id == classroom_id, ClassSetting.key == key))
    if row:
        row.text = text
    else:
        s.add(ClassSetting(classroom_id=classroom_id, key=key, text=text))


def _survey_agg(s: Session, classroom_id: int) -> dict:
    rows = s.scalars(select(SurveyResponse).where(
        SurveyResponse.classroom_id == classroom_id)).all()
    reasons: dict[str, int] = {}
    notes = []
    for r in rows:
        try:
            rd = json.loads(r.reasons or "{}")
        except Exception:
            rd = {}
        for k, v in rd.items():
            reasons[k] = reasons.get(k, 0) + max(0, int(v or 0))
        if (r.etc_note or "").strip():
            notes.append(r.etc_note.strip())
    reasons_sorted = sorted([(k, v) for k, v in reasons.items() if v > 0], key=lambda x: -x[1])
    return {
        "teams": len(rows),
        "members": sum(r.members for r in rows),
        "left": sum(r.left_count for r in rows),
        "reasons": [{"why": w, "count": c} for w, c in reasons_sorted],
        "etc_notes": notes,
    }


def _my_survey(s: Session, team_id: int) -> dict:
    row = s.scalar(select(SurveyResponse).where(SurveyResponse.team_id == team_id))
    if not row:
        return {"members": 0, "reasons": {}, "etc_note": ""}
    try:
        rd = json.loads(row.reasons or "{}")
    except Exception:
        rd = {}
    return {"members": row.members, "reasons": rd, "etc_note": row.etc_note or ""}


def _survey_cards(s: Session, classroom_id: int) -> list[dict]:
    """조별 조사를 합산해 '우리 반 정보 카드'로 자동 정리한다."""
    a = _survey_agg(s, classroom_id)
    if a["teams"] == 0 or (a["members"] == 0 and a["left"] == 0):
        return []
    cards = [{"id": 1, "type": "숫자",
              "text": f"오늘 우리 반 {a['members']}명 중 {a['left']}명이 급식을 남겼다."}]
    idx = 2
    for r in a["reasons"]:
        why = "다른 이유로" if r["why"] == "기타" else r["why"]
        cards.append({"id": idx, "type": "이유별", "text": f"{r['count']}명은 {why} 남겼다."})
        idx += 1
    for note in a.get("etc_notes", [])[:3]:
        cards.append({"id": idx, "type": "친구의 말", "text": f"한 조가 '{note}'라고 적었다."})
        idx += 1
    return cards


def _class_cards(s: Session, classroom_id: int) -> tuple[list[dict], str]:
    """(카드, 출처). 교사가 전달/입력한 카드가 있으면 그것, 없으면 예시."""
    raw = _cs_get(s, classroom_id, "data3_cards")
    if raw.strip():
        return _parse_cards(raw), "class"     # 교사가 전달(또는 입력)한 우리 반 카드
    return DATA3_CARDS, "default"             # 아직 전달 전 — 예시 카드


@app.get("/api/teacher/data3/cards/{classroom_id}")
def data3_cards_get(classroom_id: int, s: Session = Depends(db)):
    """교사용: 이 학급의 3차시 정보 카드 입력값(설문 결과)을 준다."""
    cr = s.get(Classroom, classroom_id)
    if not cr:
        raise HTTPException(404, "없는 학급")
    raw = _cs_get(s, classroom_id, "data3_cards")
    return {"text": raw, "default": DATA3_DEFAULT_CARDS_TEXT}


class ClassCardsIn(BaseModel):
    text: str = ""


@app.post("/api/teacher/data3/cards/{classroom_id}")
def data3_cards_set(classroom_id: int, body: ClassCardsIn, s: Session = Depends(db)):
    """교사용: 설문 결과를 정보 카드로 저장한다(한 줄에 카드 하나, 학급 공유)."""
    cr = s.get(Classroom, classroom_id)
    if not cr:
        raise HTTPException(404, "없는 학급")
    _cs_set(s, classroom_id, "data3_cards", body.text)
    log(s, "data3_class_cards", classroom=classroom_id)
    s.commit()
    return {"ok": True, "cards": _parse_cards(body.text)}


class SurveyIn(BaseModel):
    team_id: int
    members: int = Field(ge=0, le=60)
    reasons: dict[str, int] = {}   # {이유: 인원}
    etc_note: str = ""             # '기타' 의견(교사가 받음)


@app.post("/api/team/survey")
def survey_submit(body: SurveyIn, s: Session = Depends(db)):
    """3차시 도입: 우리 조 조사 결과를 낸다(조당 하나, 덮어쓴다). 반 전체로 합산."""
    team = s.get(Team, body.team_id)
    if not team:
        raise HTTPException(404, "없는 조")
    reasons = {k: max(0, int(v or 0)) for k, v in body.reasons.items() if int(v or 0) > 0}
    left = sum(reasons.values())
    row = s.scalar(select(SurveyResponse).where(SurveyResponse.team_id == body.team_id))
    if row:
        row.members, row.left_count = body.members, left
        row.reasons = json.dumps(reasons, ensure_ascii=False)
        row.etc_note = body.etc_note.strip()
    else:
        s.add(SurveyResponse(
            classroom_id=team.classroom_id, team_id=body.team_id,
            members=body.members, left_count=left,
            reasons=json.dumps(reasons, ensure_ascii=False),
            etc_note=body.etc_note.strip(),
        ))
    log(s, "survey", team=body.team_id, members=body.members, left=left)
    s.commit()
    return {"ok": True, "survey": _survey_agg(s, team.classroom_id)}


@app.get("/api/teacher/data3/survey/{classroom_id}")
def data3_survey_teacher(classroom_id: int, s: Session = Depends(db)):
    """교사용: 조별 조사 결과 + 반 전체 집계 + 기타 의견 + 현재 전달된 카드."""
    cr = s.get(Classroom, classroom_id)
    if not cr:
        raise HTTPException(404, "없는 학급")
    rows = s.execute(
        select(Team.number, SurveyResponse)
        .join(SurveyResponse, SurveyResponse.team_id == Team.id)
        .where(SurveyResponse.classroom_id == classroom_id)
        .order_by(Team.number)
    ).all()
    per_team = []
    for num, r in rows:
        try:
            rd = json.loads(r.reasons or "{}")
        except Exception:
            rd = {}
        per_team.append({"team_no": num, "members": r.members, "left": r.left_count,
                         "reasons": rd, "etc_note": r.etc_note or ""})
    cards, source = _class_cards(s, classroom_id)
    return {
        "survey": _survey_agg(s, classroom_id),
        "per_team": per_team,
        "delivered": source == "class",   # 이미 학생에게 전달됐는지
    }


@app.post("/api/teacher/data3/survey-to-cards/{classroom_id}")
def data3_survey_to_cards(classroom_id: int, s: Session = Depends(db)):
    """교사용: 조사 집계를 '우리 반 정보 카드'로 만들어 학생에게 전달한다."""
    cr = s.get(Classroom, classroom_id)
    if not cr:
        raise HTTPException(404, "없는 학급")
    cards = _survey_cards(s, classroom_id)
    if not cards:
        raise HTTPException(400, "아직 조사 결과가 없습니다. 조들이 조사를 먼저 저장해야 합니다.")
    text = "\n".join(f"[{c['type']}] {c['text']}" for c in cards)
    _cs_set(s, classroom_id, "data3_cards", text)
    log(s, "data3_survey_deliver", classroom=classroom_id, cards=len(cards))
    s.commit()
    return {"ok": True, "text": text, "cards": cards}


@app.get("/api/team/{team_id}/data3")
def data3_state(team_id: int, s: Session = Depends(db)):
    team = s.get(Team, team_id)
    if not team:
        raise HTTPException(404, "없는 조")
    sel = _n3_get(s, team_id, "cards")
    cards, source = _class_cards(s, team.classroom_id)
    return {
        "survey": _survey_agg(s, team.classroom_id),
        "my_survey": _my_survey(s, team_id),
        "why_options": DATA3_WHY,
        "cards": cards,
        "cards_source": source,
        "questions": DATA3_QUESTIONS,
        "selected": [int(x) for x in sel.split(",") if x.strip().isdigit()],
        "cards_reason": _n3_get(s, team_id, "cards_reason"),
        "answers": {q["key"]: _n3_get(s, team_id, f"cmp_{q['key']}") for q in DATA3_QUESTIONS},
        "compare_note": _n3_get(s, team_id, "compare_note"),
        "trace": {k: _n3_get(s, team_id, f"trace_{k}") for k in ("box", "line", "tri")},
        "rewrite_q": _n3_get(s, team_id, "rewrite_q"),
        "rewrite_ans": _n3_get(s, team_id, "rewrite_ans"),
        "rewrite_reason": _n3_get(s, team_id, "rewrite_reason"),
        "summary": [_n3_get(s, team_id, f"sum{i}") for i in (1, 2, 3)],
    }


class Data3CardsIn(BaseModel):
    team_id: int
    selected: list[int] = []
    reason: str = ""


@app.post("/api/team/data3/cards")
def data3_cards(body: Data3CardsIn, s: Session = Depends(db)):
    """활동1: 고른 조사 데이터 카드 + 고른 까닭 저장."""
    _n3_set(s, body.team_id, "cards", ",".join(str(x) for x in body.selected))
    _n3_set(s, body.team_id, "cards_reason", body.reason.strip())
    log(s, "data3_cards", team=body.team_id, count=len(body.selected))
    s.commit()
    return {"ok": True}


@app.post("/api/team/data3/compare/{team_id}")
async def data3_compare(team_id: int, s: Session = Depends(db)):
    """활동2: 세 질문(가·나·다)에 대한 AI 답을 만든다(조 공유, 캐시). 동시 클릭 방지."""
    team = s.get(Team, team_id)
    if not team:
        raise HTTPException(404, "없는 조")
    lock = _data3_locks.setdefault(team_id, asyncio.Lock())
    async with lock:
        gen = get_generator()
        out = {}
        for q in DATA3_QUESTIONS:
            key = f"cmp_{q['key']}"
            existing = _n3_get(s, team_id, key)
            if existing:
                out[q["key"]] = existing
                continue
            g = await gen.generate(q["q"])
            if not g.ok:
                raise HTTPException(502, f"AI 호출 실패: {g.error}")
            _n3_set(s, team_id, key, g.text.strip())
            out[q["key"]] = g.text.strip()
        log(s, "data3_compare", team=team_id)
        s.commit()
    return {"answers": out}


class Data3RewriteIn(BaseModel):
    team_id: int
    prompt: str = Field(min_length=1)


@app.post("/api/team/data3/rewrite")
async def data3_rewrite(body: Data3RewriteIn, s: Session = Depends(db)):
    """활동3: 데이터를 넣어 고쳐 쓴 질문을 AI에 보내 답을 받는다(다시 보내면 새로)."""
    team = s.get(Team, body.team_id)
    if not team:
        raise HTTPException(404, "없는 조")
    gen = get_generator()
    g = await gen.generate(body.prompt.strip())
    if not g.ok:
        raise HTTPException(502, f"AI 호출 실패: {g.error}")
    _n3_set(s, body.team_id, "rewrite_q", body.prompt.strip())
    _n3_set(s, body.team_id, "rewrite_ans", g.text.strip())
    log(s, "data3_rewrite", team=body.team_id, chars=len(body.prompt))
    s.commit()
    return {"answer": g.text.strip()}


# ── 2차시 활동3: 통과 답 O/X 적합 판정 + 최종 선정 ────────────────────
@app.get("/api/team/{team_id}/activity/{session_no}")
def activity_get(team_id: int, session_no: int, s: Session = Depends(db)):
    """통과 답에서 만든 옵션 + O/X 판정 + 최종 선정 메모."""
    opts = s.scalars(
        select(ActivityOption)
        .where(ActivityOption.team_id == team_id, ActivityOption.session_no == session_no)
        .order_by(ActivityOption.idx)
    ).all()
    final = s.scalar(
        select(TeamNote).where(
            TeamNote.team_id == team_id, TeamNote.session_no == session_no, TeamNote.key == "final"
        )
    )
    return {
        "options": [
            {"idx": o.idx, "text": o.text, "fit": o.fit, "reason": o.reason} for o in opts
        ],
        "final": (final.text if final else ""),
    }


class OptionIn(BaseModel):
    team_id: int
    session_no: int
    idx: int
    fit: bool | None = None    # O=True / X=False / 미정=None
    reason: str = ""


@app.post("/api/team/activity/option")
def activity_option(body: OptionIn, s: Session = Depends(db)):
    """한 옵션의 O/X 적합 판정과 이유를 저장한다."""
    o = s.scalar(
        select(ActivityOption).where(
            ActivityOption.team_id == body.team_id,
            ActivityOption.session_no == body.session_no,
            ActivityOption.idx == body.idx,
        )
    )
    if not o:
        raise HTTPException(404, "없는 옵션")
    if body.fit is False and len(body.reason.strip()) < 2:
        raise HTTPException(400, "X(적합하지 않음)로 분류하면 이유를 써 주세요.")
    o.fit, o.reason = body.fit, body.reason.strip()
    s.commit()
    return {"ok": True}


class JudgeVerdict(BaseModel):
    item_index: int
    verdict: int = Field(ge=1, le=3)   # 1 바로쓸수있다 / 2 안맞다 / 3 못한다
    reason: str = ""


class JudgeIn(BaseModel):
    team_id: int
    verdicts: list[JudgeVerdict]


@app.post("/api/team/judge")
def team_judge(body: JudgeIn, s: Session = Depends(db)):
    """조의 판정을 저장(같은 항목은 덮어쓴다). ②·③ 판정에는 까닭이 필요하다."""
    for jv in body.verdicts:
        if jv.verdict in (2, 3) and len(jv.reason.strip()) < 2:
            raise HTTPException(400, f"{jv.item_index}번: ②·③ 판정에는 까닭을 써 주세요.")
    for jv in body.verdicts:
        row = s.scalar(
            select(Judgment).where(
                Judgment.team_id == body.team_id, Judgment.item_index == jv.item_index
            )
        )
        if row:
            row.verdict, row.reason = jv.verdict, jv.reason.strip()
        else:
            s.add(Judgment(
                team_id=body.team_id, item_index=jv.item_index,
                verdict=jv.verdict, reason=jv.reason.strip(),
            ))
    s.commit()
    return {"saved": len(body.verdicts)}


# ─────────────────────────────────────────────────────────────
# 6차시 — 역추적 (retrace). 자기 5차시 질문을 스스로 되짚는다. AI·교사 없음.
# ─────────────────────────────────────────────────────────────
def _passed_session5(s: Session, team_id: int) -> list[PromptVersion]:
    subs = s.scalars(
        select(Submission).where(Submission.team_id == team_id, Submission.session_no == 5)
    ).all()
    return [v for sub in subs for v in sub.versions if v.sent_prompt]


@app.get("/api/team/{team_id}/retrace")
def retrace_get(team_id: int, s: Session = Depends(db)):
    team = s.get(Team, team_id)
    if not team:
        raise HTTPException(404, "없는 조")
    sources = [{"version_id": v.id, "prompt": v.sent_prompt} for v in _passed_session5(s, team_id)]
    saved = [
        {
            "source_version_id": r.source_version_id,
            "tags": _retrace_tags(r),
            "note": r.retrace_note,
        }
        for r in s.scalars(
            select(RetraceTag).where(RetraceTag.team_id == team_id).order_by(RetraceTag.id)
        ).all()
    ]
    return {"sources": sources, "saved": saved}


def _retrace_tags(r) -> list[str]:
    return [
        t for t, on in [
            ("[상황]", r.tag_situation), ("[대상]", r.tag_audience),
            ("[조건]", r.tag_condition), ("[목적]", r.tag_purpose),
        ] if on
    ]


class RetraceIn(BaseModel):
    team_id: int
    source_version_id: int | None = None
    tag_situation: bool = False
    tag_audience: bool = False
    tag_condition: bool = False
    tag_purpose: bool = False
    retrace_note: str = ""


@app.post("/api/team/retrace")
def retrace_post(body: RetraceIn, s: Session = Depends(db)):
    if not any([body.tag_situation, body.tag_audience, body.tag_condition, body.tag_purpose]):
        raise HTTPException(400, "빠졌다고 생각하는 것에 태그를 하나 이상 붙여 주세요.")
    if len(body.retrace_note.strip()) < 5:
        raise HTTPException(400, "결과와 질문을 잇는 역추적을 한 줄로 써 주세요. (예: 1학년 복도에서만 스티커가 적었다 → …)")
    s.add(RetraceTag(
        team_id=body.team_id, source_version_id=body.source_version_id,
        tag_situation=body.tag_situation, tag_audience=body.tag_audience,
        tag_condition=body.tag_condition, tag_purpose=body.tag_purpose,
        retrace_note=body.retrace_note.strip(),
    ))
    s.commit()
    return {"ok": True}


# ─────────────────────────────────────────────────────────────
# 7차시 — 동료 판별 (peer). 다른 조 질문에 태그+까닭을 붙여 되돌린다.
# ─────────────────────────────────────────────────────────────
def _latest_prompt(s: Session, team_id: int) -> PromptVersion | None:
    """그 조의 최신 프롬프트 판(5차시 우선, 없으면 아무 차시)."""
    subs = s.scalars(
        select(Submission).where(Submission.team_id == team_id).order_by(Submission.id.desc())
    ).all()
    s5 = [sub for sub in subs if sub.session_no == 5]
    for sub in (s5 + subs):
        if sub.versions:
            return sub.versions[-1]
    return None


@app.get("/api/team/{team_id}/peer")
def peer_get(team_id: int, s: Session = Depends(db)):
    team = s.get(Team, team_id)
    if not team:
        raise HTTPException(404, "없는 조")
    others = s.scalars(
        select(Team).where(Team.classroom_id == team.classroom_id, Team.id != team_id)
        .order_by(Team.number)
    ).all()
    done = {
        pr.target_version_id
        for pr in s.scalars(select(PeerReview).where(PeerReview.reviewer_team_id == team_id)).all()
    }
    targets = []
    for ot in others:
        v = _latest_prompt(s, ot.id)
        if not v:
            continue
        targets.append({
            "team_no": ot.number, "version_id": v.id,
            "prompt": v.student_prompt, "reviewed": v.id in done,
        })
        if len(targets) >= 3:
            break
    my_vids = {v.id for sub in
               s.scalars(select(Submission).where(Submission.team_id == team_id)).all()
               for v in sub.versions}
    received = []
    if my_vids:
        for pr in s.scalars(
            select(PeerReview).where(PeerReview.target_version_id.in_(my_vids)).order_by(PeerReview.id)
        ).all():
            reviewer = s.get(Team, pr.reviewer_team_id)
            received.append({
                "reviewer_no": reviewer.number if reviewer else None,
                "tags": _retrace_tags(pr), "reason": pr.reason,
            })
    return {"targets": targets, "received": received}


class PeerReviewIn(BaseModel):
    reviewer_team_id: int
    target_version_id: int
    tag_situation: bool = False
    tag_audience: bool = False
    tag_condition: bool = False
    tag_purpose: bool = False
    reason: str = ""


@app.post("/api/team/peer-review")
def peer_review_post(body: PeerReviewIn, s: Session = Depends(db)):
    v = s.get(PromptVersion, body.target_version_id)
    if not v:
        raise HTTPException(404, "없는 질문")
    if not any([body.tag_situation, body.tag_audience, body.tag_condition, body.tag_purpose]):
        raise HTTPException(400, "빠진 것에 태그를 하나 이상 붙여 주세요.")
    if len(body.reason.strip()) < 10:
        raise HTTPException(400, "까닭을 써 주세요. " + NUDGE)
    rtype, rscore = classify_reason(body.reason)
    s.add(PeerReview(
        reviewer_team_id=body.reviewer_team_id, target_version_id=body.target_version_id,
        tag_situation=body.tag_situation, tag_audience=body.tag_audience,
        tag_condition=body.tag_condition, tag_purpose=body.tag_purpose,
        reason=body.reason.strip(), reason_type=rtype, reason_score=rscore,
    ))
    s.commit()
    return {"reason_type": rtype.value}


# ─────────────────────────────────────────────────────────────
# 8차시 — 전이 (transfer). 되돌림·AI·피드백 없음. 저장만 한다.
# ─────────────────────────────────────────────────────────────
class TransferIn(BaseModel):
    team_id: int
    prompt: str = Field(min_length=1)


@app.post("/api/team/transfer")
def transfer_post(body: TransferIn, s: Session = Depends(db)):
    s.add(TransferPrompt(team_id=body.team_id, prompt=body.prompt.strip()))
    s.commit()
    return {"ok": True}


@app.get("/api/team/{team_id}/transfer")
def transfer_get(team_id: int, s: Session = Depends(db)):
    team = s.get(Team, team_id)
    if not team:
        raise HTTPException(404, "없는 조")
    rows = s.scalars(
        select(TransferPrompt).where(TransferPrompt.team_id == team_id).order_by(TransferPrompt.id)
    ).all()
    return {"prompts": [{"prompt": r.prompt} for r in rows]}


# ─────────────────────────────────────────────────────────────
# 연구자 — 처치 충실도 내보내기
# ─────────────────────────────────────────────────────────────
@app.get("/api/admin/fidelity")
def fidelity(s: Session = Depends(db)):
    """논문에 쓸 처치 충실도 지표. 학급별로 낸다."""
    out = []
    for cr in s.scalars(select(Classroom)).all():
        subs = s.scalars(
            select(Submission)
            .join(Team, Team.id == Submission.team_id)
            .where(Team.classroom_id == cr.id)
        ).all()
        reviews = [
            v.review
            for sub in subs
            for v in sub.versions
            if v.review and v.review.decision == Status.RETURNED
        ]
        narr = sum(1 for r in reviews if r.reason_type == ReasonType.NARRATIVE)
        acc = sum(1 for r in reviews if r.reason_type == ReasonType.ACCURACY)
        responses = [
            v.response for sub in subs for v in sub.versions if v.response
        ]
        asked = sum(1 for r in responses if r.asked_back or r.gave_advice)
        viol = sum(
            1 for sub in subs for v in sub.versions if v.invariant_violated
        )
        models = sorted({r.model for r in responses})
        hashes = sorted({r.system_prompt_hash for r in responses})
        out.append(
            {
                "classroom": f"{cr.school} {cr.name}".strip() + "학급",
                "condition": cr.condition.value,
                "submissions": len(subs),
                "returns": len(reviews),
                "mean_return_count": (
                    round(sum(x.return_count for x in subs) / len(subs), 2)
                    if subs
                    else 0
                ),
                # ★ 처치가 제대로 전달되었는가
                "narrative_reason_rate": (
                    round(narr / len(reviews), 2) if reviews else None
                ),
                "accuracy_reason_count": acc,
                # ★ 모델이 되돌림을 대신했는가 (0이어야 한다)
                "model_asked_back_rate": (
                    round(asked / len(responses), 3) if responses else None
                ),
                # ★ 교사가 대신 고쳤는가 (0이어야 한다)
                "invariant_violations": viol,
                # ★ 재현성 — 기간 중 모델이 바뀌지 않았는가
                "models_used": models,
                "system_prompt_hashes": hashes,
            }
        )
    return out


@app.get("/api/admin/config")
def config():
    from app.generators.base import SYSTEM_PROMPT_HASH

    gen = get_generator()
    return {
        "generator": settings.GENERATOR,
        "model": gen.model,
        "temperature": gen.temperature,
        "system_prompt_hash": SYSTEM_PROMPT_HASH,
        "warning": (
            "NullGenerator는 통제 조건이 아니다. 실제 수업에서는 실모델을 쓴다."
            if settings.GENERATOR == "null"
            else ""
        ),
    }


@app.get("/api/admin/ping")
async def admin_ping():
    """실제 AI를 아주 짧게 호출해 키가 먹히는지만 확인한다(DB 저장 없음).

    ok:true 면 키·모델 정상. ok:false 면 error 에 원인(예: 401 invalid x-api-key)."""
    gen = get_generator()
    g = await gen.generate("안녕")
    return {"ok": g.ok, "provider": g.provider, "model": g.model,
            "error": (g.error or "")[:300], "sample": (g.text or "")[:80]}


class ResetIn(BaseModel):
    code: str


@app.post("/api/admin/reset")
def admin_reset(body: ResetIn, s: Session = Depends(db)):
    """활동 데이터(제출·판정·대화·이벤트 등)를 전부 지운다. 학급·조는 남긴다.

    공개 주소이므로 코드가 맞아야만 실행한다(RESET_CODE).
    """
    if body.code != settings.RESET_CODE:
        raise HTTPException(403, "초기화 코드가 틀렸습니다.")
    # 자식 → 부모 순서로 지운다(학급/조는 남긴다)
    for model in (
        AIResponse, Review, PromptVersion, Submission,
        Judgment, JudgeItem, RetraceTag, PeerReview, TransferPrompt,
        TeacherAnswer, ActivityOption, TeamNote, SurveyResponse, Event,
    ):
        s.query(model).delete()
    s.commit()
    return {"ok": True}


@app.post("/api/teacher/reset/{classroom_id}")
def teacher_reset(classroom_id: int, body: ResetIn, s: Session = Depends(db)):
    """한 학급의 활동 데이터만 초기화한다(조별 답변·판정·조사·대화·기록).

    학급·조 자체는 남긴다. 다른 학급 데이터는 건드리지 않는다.
    교사용. 코드가 맞아야 실행한다(RESET_CODE)."""
    if body.code != settings.RESET_CODE:
        raise HTTPException(403, "초기화 코드가 틀렸습니다.")
    cr = s.get(Classroom, classroom_id)
    if not cr:
        raise HTTPException(404, "없는 학급")

    team_ids = list(s.scalars(select(Team.id).where(Team.classroom_id == classroom_id)))
    sub_ids = list(s.scalars(
        select(Submission.id).where(Submission.team_id.in_(team_ids)))) if team_ids else []
    ver_ids = list(s.scalars(
        select(PromptVersion.id).where(PromptVersion.submission_id.in_(sub_ids)))) if sub_ids else []

    def wipe(model, cond):
        s.query(model).filter(cond).delete(synchronize_session=False)

    if ver_ids:  # 자식(응답·검토·동료판정·역추적) → 버전 순
        wipe(AIResponse, AIResponse.version_id.in_(ver_ids))
        wipe(Review, Review.version_id.in_(ver_ids))
        wipe(PeerReview, PeerReview.target_version_id.in_(ver_ids))
        wipe(RetraceTag, RetraceTag.source_version_id.in_(ver_ids))
        wipe(PromptVersion, PromptVersion.id.in_(ver_ids))
    if sub_ids:
        wipe(Event, Event.submission_id.in_(sub_ids))
        wipe(Submission, Submission.id.in_(sub_ids))
    if team_ids:  # 조 단위 활동 데이터
        wipe(Judgment, Judgment.team_id.in_(team_ids))
        wipe(JudgeItem, JudgeItem.team_id.in_(team_ids))
        wipe(RetraceTag, RetraceTag.team_id.in_(team_ids))
        wipe(PeerReview, PeerReview.reviewer_team_id.in_(team_ids))
        wipe(TransferPrompt, TransferPrompt.team_id.in_(team_ids))
        wipe(ActivityOption, ActivityOption.team_id.in_(team_ids))
        wipe(TeamNote, TeamNote.team_id.in_(team_ids))
    wipe(TeacherAnswer, TeacherAnswer.classroom_id == classroom_id)
    s.commit()
    return {"ok": True}


@app.post("/api/teacher/reset/{classroom_id}/{session_no}")
def teacher_reset_session(classroom_id: int, session_no: int,
                          body: ResetIn, s: Session = Depends(db)):
    """한 학급의 '한 차시' 활동 데이터만 초기화한다.

    다른 차시·다른 학급은 그대로 둔다. 교사용. 코드가 맞아야 실행한다."""
    if body.code != settings.RESET_CODE:
        raise HTTPException(403, "초기화 코드가 틀렸습니다.")
    cr = s.get(Classroom, classroom_id)
    if not cr:
        raise HTTPException(404, "없는 학급")

    team_ids = list(s.scalars(select(Team.id).where(Team.classroom_id == classroom_id)))

    def wipe(model, *conds):
        s.query(model).filter(*conds).delete(synchronize_session=False)

    if team_ids:
        # 되돌림 루프(제출 기반, 2~5차시) — session_no 로 필터
        sub_ids = list(s.scalars(select(Submission.id).where(
            Submission.team_id.in_(team_ids), Submission.session_no == session_no)))
        ver_ids = list(s.scalars(select(PromptVersion.id).where(
            PromptVersion.submission_id.in_(sub_ids)))) if sub_ids else []
        if ver_ids:
            wipe(AIResponse, AIResponse.version_id.in_(ver_ids))
            wipe(Review, Review.version_id.in_(ver_ids))
            wipe(PeerReview, PeerReview.target_version_id.in_(ver_ids))
            wipe(RetraceTag, RetraceTag.source_version_id.in_(ver_ids))
            wipe(PromptVersion, PromptVersion.id.in_(ver_ids))
        if sub_ids:
            wipe(Event, Event.submission_id.in_(sub_ids))
            wipe(Submission, Submission.id.in_(sub_ids))
        # session_no 가 붙는 활동 데이터
        wipe(ActivityOption, ActivityOption.team_id.in_(team_ids),
             ActivityOption.session_no == session_no)
        wipe(TeamNote, TeamNote.team_id.in_(team_ids),
             TeamNote.session_no == session_no)
        if session_no == 3:   # 3차시 도입 조사 응답
            wipe(SurveyResponse, SurveyResponse.classroom_id == classroom_id)
        # 차시 고유 활동 (session_no 필드가 없는 것들)
        if session_no == 1:
            wipe(JudgeItem, JudgeItem.team_id.in_(team_ids))
            wipe(Judgment, Judgment.team_id.in_(team_ids))
        elif session_no == 6:
            wipe(RetraceTag, RetraceTag.team_id.in_(team_ids))
        elif session_no == 7:
            wipe(PeerReview, PeerReview.reviewer_team_id.in_(team_ids))
        elif session_no == 8:
            wipe(TransferPrompt, TransferPrompt.team_id.in_(team_ids))
    if session_no == 1:  # 1차시 활동3 교사 답(학급 공유)
        wipe(TeacherAnswer, TeacherAnswer.classroom_id == classroom_id)
    s.commit()
    return {"ok": True}


# ─────────────────────────────────────────────────────────────
# 응답 수집 — 모든 조의 활동 응답을 평평한 표로 모은다(연구자용).
# ─────────────────────────────────────────────────────────────
_VERDICT_LABEL = {1: "바로 쓸 수 있다", 2: "우리 학교엔 안 맞다", 3: "우리가 못 한다"}
_TAG_KO = [("tag_purpose", "목적"), ("tag_situation", "상황"), ("tag_audience", "대상"),
           ("tag_condition", "조건"), ("tag_role", "역할"), ("tag_example", "예시")]


def _collect_rows(s: Session, classroom_id: int | None = None) -> list[dict]:
    tmap, cmap = {}, {}
    for t, cr in s.execute(
        select(Team, Classroom).join(Classroom, Classroom.id == Team.classroom_id)
    ).all():
        tmap[t.id] = (cr.name, t.number)
        cmap[cr.id] = cr.name
    only_name = cmap.get(classroom_id) if classroom_id is not None else None
    rows: list[dict] = []

    def add(team_id, session, kind, detail, value):
        cn, tn = tmap.get(team_id, ("?", "?"))
        rows.append({"classroom": cn, "team": tn, "session": session,
                     "kind": kind, "detail": detail, "value": (value or "").strip()})

    # 1차시 판정 — 질문 + AI 답 5개
    for ji in s.scalars(select(JudgeItem).order_by(JudgeItem.team_id, JudgeItem.idx)):
        if ji.idx == 1:
            add(ji.team_id, 1, "판정-질문", "", ji.question)
        add(ji.team_id, 1, "판정-AI답", f"답{ji.idx}", ji.text)
    # 1차시 판정 결과
    for jd in s.scalars(select(Judgment).order_by(Judgment.team_id, Judgment.item_index)):
        v = _VERDICT_LABEL.get(jd.verdict, str(jd.verdict))
        add(jd.team_id, 1, "판정", f"답{jd.item_index}",
            v + (f" / 까닭: {jd.reason}" if jd.reason else ""))
    # 조 메모(1차시 비교, 2차시 최종선정 등)
    for tn in s.scalars(select(TeamNote).order_by(TeamNote.team_id, TeamNote.session_no)):
        add(tn.team_id, tn.session_no, "메모", tn.key, tn.text)
    # 되돌림 루프 — 학생 질문 · 교사 판정 · AI 답
    for sub in s.scalars(select(Submission).order_by(Submission.team_id, Submission.session_no)):
        for v in sub.versions:
            add(sub.team_id, sub.session_no, "학생질문", f"{v.version}판", v.student_prompt)
            if v.review:
                tags = "·".join(lab for k, lab in _TAG_KO if getattr(v.review, k))
                detail = v.review.decision.value + (f" / 태그: {tags}" if tags else "")
                add(sub.team_id, sub.session_no, "교사판정", f"{v.version}판",
                    detail + (f" / 까닭: {v.review.reason}" if v.review.reason else ""))
            if v.response and v.response.text:
                add(sub.team_id, sub.session_no, "AI답", f"{v.version}판", v.response.text)
    # 2차시 적합 판정(O/X)
    for op in s.scalars(select(ActivityOption).order_by(
            ActivityOption.team_id, ActivityOption.session_no, ActivityOption.idx)):
        fit = "O적합" if op.fit is True else ("X부적합" if op.fit is False else "미판정")
        add(op.team_id, op.session_no, "적합판정", f"항목{op.idx}",
            f"{op.text} → {fit}" + (f" / 이유: {op.reason}" if op.reason else ""))
    # 6~8차시
    for rt in s.scalars(select(RetraceTag)):
        add(rt.team_id, 6, "역추적", "", rt.retrace_note)
    for pr in s.scalars(select(PeerReview)):
        add(pr.reviewer_team_id, 7, "동료판정", "", pr.reason)
    for tp in s.scalars(select(TransferPrompt)):
        add(tp.team_id, 8, "전이질문", "", tp.prompt)
    # 1차시 활동3 교사 답(학급 단위)
    for ta in s.scalars(select(TeacherAnswer)):
        nm = cmap.get(ta.classroom_id, "?")
        for det, val in (("교사 프롬프트", ta.prompt), ("AI 답", ta.text)):
            rows.append({"classroom": nm, "team": "-", "session": 1,
                         "kind": "교사답(활동3)", "detail": det, "value": (val or "").strip()})

    if only_name is not None:
        rows = [r for r in rows if str(r["classroom"]) == str(only_name)]
    rows.sort(key=lambda r: (str(r["classroom"]), _team_sort(r["team"]), r["session"], r["kind"]))
    return rows


def _team_sort(team):
    try:
        return (0, int(team))
    except (ValueError, TypeError):
        return (1, 0)   # 교사답('-') 등은 뒤로


@app.get("/api/teacher/responses/{classroom_id}")
def teacher_responses(classroom_id: int, s: Session = Depends(db)):
    """교사용: 자기 학급 학생 응답을 활동별로 실시간 확인한다."""
    cr = s.get(Classroom, classroom_id)
    if not cr:
        raise HTTPException(404, "없는 학급")
    rows = _collect_rows(s, classroom_id)
    return {"count": len(rows), "rows": rows}


@app.get("/api/admin/collect")
def collect(s: Session = Depends(db)):
    rows = _collect_rows(s)
    return {"count": len(rows), "rows": rows}


@app.get("/api/admin/collect.csv")
def collect_csv(s: Session = Depends(db)):
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["학급", "조", "차시", "항목", "세부", "내용"])
    for r in _collect_rows(s):
        w.writerow([r["classroom"], r["team"], r["session"], r["kind"], r["detail"], r["value"]])
    # Excel 한글 깨짐 방지용 BOM
    data = "﻿" + buf.getvalue()
    return Response(content=data, media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": "attachment; filename=promptgate_collect.csv"})


@app.get("/api/health")
def health():
    return {"ok": True}


# ─────────────────────────────────────────────────────────────
# 프런트 서빙 — 백엔드가 index.html 을 / 에서 함께 준다.
# 그러면 배포 시 주소가 하나로 통일된다(프런트가 location.origin 을 API로 씀).
# ─────────────────────────────────────────────────────────────
FRONTEND_HTML = os.path.join(
    os.path.dirname(__file__), "..", "..", "frontend", "index.html"
)


@app.get("/")
def index():
    # 브라우저가 예전 화면을 캐시하지 않도록 — 배포 즉시 새 코드가 보이게 한다.
    return FileResponse(FRONTEND_HTML, headers={"Cache-Control": "no-store, max-age=0"})
