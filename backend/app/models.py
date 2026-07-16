"""데이터 모델.

연구 원자료가 여기서 나온다. 삭제하지 않는다. event_log는 append-only다.
"""

from __future__ import annotations

import datetime as dt
import enum

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Condition(str, enum.Enum):
    TREATMENT = "treatment"   # 되돌림 루프 (태그 + 청자 관점 까닭)
    CONTROL = "control"       # 안전 검토만


class Status(str, enum.Enum):
    PENDING = "pending"       # 교사 검토 대기
    RETURNED = "returned"     # 되돌아감 (학생이 고쳐야 함)
    PASSED = "passed"         # 통과 — AI에 전송됨
    BLOCKED = "blocked"       # 유해 내용으로 차단 (양 조건 공통)


class ReasonType(str, enum.Enum):
    NARRATIVE = "narrative"   # 서사형 — 청자에게 무슨 일이 일어나는지
    ACCURACY = "accuracy"     # 정확성형 — "무엇이 부족하다"
    UNKNOWN = "unknown"


class Classroom(Base):
    __tablename__ = "classrooms"
    id: Mapped[int] = mapped_column(primary_key=True)
    school: Mapped[str] = mapped_column(String(60))
    name: Mapped[str] = mapped_column(String(30))          # 예: 5-3
    condition: Mapped[Condition] = mapped_column(Enum(Condition))
    # 한 질문을 최대 몇 번까지 되돌릴 수 있는지. 0 = 무제한(기본). 교사가 정한다.
    max_returns: Mapped[int] = mapped_column(Integer, default=0)
    teams: Mapped[list["Team"]] = relationship(back_populates="classroom")


class Team(Base):
    """조. 로그인·제출·대화의 단위. 같은 조원은 이 조의 모든 것을 공유한다.

    개인 이름도 출석번호도 저장하지 않는다 — 조 번호만 쓴다.
    """

    __tablename__ = "teams"
    id: Mapped[int] = mapped_column(primary_key=True)
    classroom_id: Mapped[int] = mapped_column(ForeignKey("classrooms.id"))
    number: Mapped[int] = mapped_column(Integer)           # 조 번호 (1~N)
    classroom: Mapped["Classroom"] = relationship(back_populates="teams")


class Submission(Base):
    """한 조가 한 차시에 겪는 되돌림 루프 하나. 조원 전체가 공유한다."""

    __tablename__ = "submissions"
    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    session_no: Mapped[int] = mapped_column(Integer)       # 1~8차시
    return_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[Status] = mapped_column(Enum(Status), default=Status.PENDING)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)
    versions: Mapped[list["PromptVersion"]] = relationship(
        back_populates="submission", order_by="PromptVersion.version"
    )


class PromptVersion(Base):
    """학생이 쓴 프롬프트 한 판. 되돌아올 때마다 새 판이 생긴다."""

    __tablename__ = "prompt_versions"
    id: Mapped[int] = mapped_column(primary_key=True)
    submission_id: Mapped[int] = mapped_column(ForeignKey("submissions.id"))
    version: Mapped[int] = mapped_column(Integer)          # 1부터

    student_prompt: Mapped[str] = mapped_column(Text)      # 학생이 쓴 것
    sent_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)  # 실제 전송된 것

    # ⚠ 불변식: 통과 시 sent_prompt == student_prompt. 교사는 대신 고칠 수 없다.
    invariant_violated: Mapped[bool] = mapped_column(Boolean, default=False)

    edit_distance: Mapped[int] = mapped_column(Integer, default=0)  # 직전 판과의 거리
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)

    submission: Mapped["Submission"] = relationship(back_populates="versions")
    review: Mapped["Review | None"] = relationship(back_populates="version", uselist=False)
    response: Mapped["AIResponse | None"] = relationship(back_populates="version", uselist=False)


class Review(Base):
    """교사의 판단. 처치에서는 통과/되돌림 두 가지뿐이다."""

    __tablename__ = "reviews"
    id: Mapped[int] = mapped_column(primary_key=True)
    version_id: Mapped[int] = mapped_column(ForeignKey("prompt_versions.id"))

    decision: Mapped[Status] = mapped_column(Enum(Status))   # PASSED / RETURNED / BLOCKED
    tag_situation: Mapped[bool] = mapped_column(Boolean, default=False)   # [상황]
    tag_audience: Mapped[bool] = mapped_column(Boolean, default=False)    # [대상]
    tag_condition: Mapped[bool] = mapped_column(Boolean, default=False)   # [조건]
    tag_purpose: Mapped[bool] = mapped_column(Boolean, default=False)     # [목적]
    tag_role: Mapped[bool] = mapped_column(Boolean, default=False)        # [역할] (2차시)
    tag_example: Mapped[bool] = mapped_column(Boolean, default=False)     # [예시] (2차시)

    reason: Mapped[str] = mapped_column(Text, default="")     # 교사가 쓴 까닭 (원문 그대로)
    reason_type: Mapped[ReasonType] = mapped_column(Enum(ReasonType), default=ReasonType.UNKNOWN)
    reason_score: Mapped[float] = mapped_column(Float, default=0.0)  # 서사성 점수 0~1

    teacher_id: Mapped[str] = mapped_column(String(40), default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)

    version: Mapped["PromptVersion"] = relationship(back_populates="review")


class AIResponse(Base):
    """모델 호출 결과. 재현성 정보를 전부 남긴다."""

    __tablename__ = "ai_responses"
    id: Mapped[int] = mapped_column(primary_key=True)
    version_id: Mapped[int] = mapped_column(ForeignKey("prompt_versions.id"))

    text: Mapped[str] = mapped_column(Text)

    provider: Mapped[str] = mapped_column(String(20))
    model: Mapped[str] = mapped_column(String(60))          # 별칭이 아니라 고정 버전
    temperature: Mapped[float] = mapped_column(Float)
    system_prompt_hash: Mapped[str] = mapped_column(String(20))
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)

    # ⚠ 처치 충실도 — 모델이 되돌림을 대신했는가
    asked_back: Mapped[bool] = mapped_column(Boolean, default=False)
    gave_advice: Mapped[bool] = mapped_column(Boolean, default=False)
    retried: Mapped[bool] = mapped_column(Boolean, default=False)
    retry_fixed: Mapped[bool] = mapped_column(Boolean, default=False)

    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)

    version: Mapped["PromptVersion"] = relationship(back_populates="response")


class Event(Base):
    """append-only. 무슨 일이 언제 일어났는지 전부 남긴다. 수정하지 않는다."""

    __tablename__ = "events"
    id: Mapped[int] = mapped_column(primary_key=True)
    at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)
    kind: Mapped[str] = mapped_column(String(40))
    submission_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payload: Mapped[str] = mapped_column(Text, default="")


# ─────────────────────────────────────────────────────────────
# 차시별 활동 (되돌림 루프[2~5차시]가 아닌 차시의 데이터)
#   1차시 판정 · 6차시 역추적 · 7차시 동료 판별 · 8차시 전이
# ─────────────────────────────────────────────────────────────
class JudgeItem(Base):
    """1차시: 조가 쓴 질문에 대해 받은 AI 답 5개. 조 단위로 공유한다.

    학생(조)이 자기 질문을 AI에 보내 받은 답 다섯 개를 조가 판정한다.
    `question`은 조가 쓴 질문(5개 항목이 같은 값을 공유). 재현성 정보(model 등)를 남긴다.
    """

    __tablename__ = "judge_items"
    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    question: Mapped[str] = mapped_column(Text, default="")  # 조가 쓴 질문
    idx: Mapped[int] = mapped_column(Integer)              # 1~5
    text: Mapped[str] = mapped_column(Text)
    provider: Mapped[str] = mapped_column(String(20), default="")
    model: Mapped[str] = mapped_column(String(60), default="")
    system_prompt_hash: Mapped[str] = mapped_column(String(20), default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)


class Judgment(Base):
    """1차시: 한 조가 한 답에 내린 판정. (team, item_index) 당 하나."""

    __tablename__ = "judgments"
    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    item_index: Mapped[int] = mapped_column(Integer)       # 1~5
    verdict: Mapped[int] = mapped_column(Integer)          # 1 바로쓸수있다 / 2 안맞다 / 3 못한다
    reason: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)


class RetraceTag(Base):
    """6차시: 조가 자기 5차시 질문을 스스로 되짚어 태그 + 역추적한 기록.

    되돌림 주체가 교사가 아니라 '자기 + 외부 준거(스티커)'다. AI 호출 없음.
    """

    __tablename__ = "retrace_tags"
    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    source_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("prompt_versions.id"), nullable=True
    )
    tag_situation: Mapped[bool] = mapped_column(Boolean, default=False)
    tag_audience: Mapped[bool] = mapped_column(Boolean, default=False)
    tag_condition: Mapped[bool] = mapped_column(Boolean, default=False)
    tag_purpose: Mapped[bool] = mapped_column(Boolean, default=False)
    retrace_note: Mapped[str] = mapped_column(Text, default="")   # 결과 → 질문 역추적 한 줄
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)


class PeerReview(Base):
    """7차시: 조가 '다른 조'의 질문에 태그 + 까닭을 붙여 되돌림.

    되돌림 주체가 동료다. 원 질문을 대신 고치지 않는다(태그 + 까닭만).
    """

    __tablename__ = "peer_reviews"
    id: Mapped[int] = mapped_column(primary_key=True)
    reviewer_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    target_version_id: Mapped[int] = mapped_column(ForeignKey("prompt_versions.id"))
    tag_situation: Mapped[bool] = mapped_column(Boolean, default=False)
    tag_audience: Mapped[bool] = mapped_column(Boolean, default=False)
    tag_condition: Mapped[bool] = mapped_column(Boolean, default=False)
    tag_purpose: Mapped[bool] = mapped_column(Boolean, default=False)
    reason: Mapped[str] = mapped_column(Text, default="")
    reason_type: Mapped[ReasonType] = mapped_column(Enum(ReasonType), default=ReasonType.UNKNOWN)
    reason_score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)


class TransferPrompt(Base):
    """8차시: 전이 과제. 되돌림·AI·피드백 없음. 저장만 한다(사후 자료)."""

    __tablename__ = "transfer_prompts"
    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    prompt: Mapped[str] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)


class TeacherAnswer(Base):
    """1차시 활동3: 교사가 '자세한 프롬프트'로 받은 AI 답(학급 공유).

    학생이 자기 조의 답과 비교한다 — 맥락을 준 질문이 어떻게 다른 답을 받는지 본다.
    """

    __tablename__ = "teacher_answers"
    id: Mapped[int] = mapped_column(primary_key=True)
    classroom_id: Mapped[int] = mapped_column(ForeignKey("classrooms.id"))
    prompt: Mapped[str] = mapped_column(Text)               # 교사가 넣은 자세한 프롬프트
    text: Mapped[str] = mapped_column(Text)                 # AI 답
    # 생성만으로는 학생에게 안 보인다. 교사가 '전송'을 눌러야 True 가 되어 공개된다.
    published: Mapped[bool] = mapped_column(Boolean, default=False)
    provider: Mapped[str] = mapped_column(String(20), default="")
    model: Mapped[str] = mapped_column(String(60), default="")
    system_prompt_hash: Mapped[str] = mapped_column(String(20), default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)


class ActivityOption(Base):
    """2차시 활동3: 통과한 질문의 AI 답을 항목으로 쪼갠 것. 조가 O/X 적합 판정 + 이유.

    통과 시 AI 답(5가지)을 쪼개 만든다. 조가 우리 학교·학급에 적합한지(O/X) 고른다.
    """

    __tablename__ = "activity_options"
    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    session_no: Mapped[int] = mapped_column(Integer)
    idx: Mapped[int] = mapped_column(Integer)               # 1~
    text: Mapped[str] = mapped_column(Text)
    fit: Mapped[bool | None] = mapped_column(Boolean, nullable=True)  # O(적합)=True / X=False / 미판정=None
    reason: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)


class ClassSetting(Base):
    """학급 단위 설정·자료 (key-value). 예: 3차시 교사가 입력한 정보 카드."""

    __tablename__ = "class_settings"
    id: Mapped[int] = mapped_column(primary_key=True)
    classroom_id: Mapped[int] = mapped_column(ForeignKey("classrooms.id"))
    key: Mapped[str] = mapped_column(String(30))
    text: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)


class TeamNote(Base):
    """조의 자유 서술 메모. key로 용도를 구분한다.

    1차시 'compare'(선생님 답과 우리 답 비교) · 2차시 'final'(최종 선정한 활동) 등.
    (team_id, session_no, key) 당 하나 — 덮어쓴다.
    """

    __tablename__ = "team_notes"
    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    session_no: Mapped[int] = mapped_column(Integer)
    key: Mapped[str] = mapped_column(String(20))
    text: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)
