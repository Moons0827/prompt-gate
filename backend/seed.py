"""연구 학급 6개를 만든다. 처치 3 / 통제 3. 각 학급을 조 6개로 나눈다.

로그인·제출·대화의 단위는 조다. 개인 이름도 출석번호도 저장하지 않는다 — 조 번호만.
학급당 25명 기준 조 6개 → 조당 약 4명.
파일명 규약: [학교]_[학급]_[조건]_[조번호]_[차시]
"""
from app.main import SessionLocal, Base, engine
from app.models import Classroom, Condition, Team

Base.metadata.create_all(engine)

# 가벼운 마이그레이션: 기존 DB에 없을 수 있는 컬럼을 더한다(있으면 무시).
from sqlalchemy import text as _sql
with engine.begin() as _conn:
    for _sqlstr, _label in [
        ("ALTER TABLE teacher_answers ADD COLUMN published BOOLEAN DEFAULT 0", "teacher_answers.published"),
        ("ALTER TABLE classrooms ADD COLUMN max_returns INTEGER DEFAULT 0", "classrooms.max_returns"),
    ]:
        try:
            _conn.execute(_sql(_sqlstr))
            print("마이그레이션:", _label, "추가")
        except Exception:
            pass  # 이미 있으면 통과
    # survey_responses 스키마 변경(개인별→조별): 옛 컬럼이면 테이블을 새로 만든다.
    try:
        cols = [r[1] for r in _conn.execute(_sql("PRAGMA table_info(survey_responses)")).fetchall()]
        if cols and "members" not in cols:
            _conn.execute(_sql("DROP TABLE survey_responses"))
            print("마이그레이션: survey_responses 재생성(조별)")
    except Exception:
        pass

Base.metadata.create_all(engine)   # 드롭된 테이블 재생성

s = SessionLocal()

TEAMS_PER_CLASS = 6

# 학급 1·2·3·4 (school 은 비운다). 모두 처치 조건 — 전체 되돌림 루프가 켜진다.
PLAN = [
    ("", "1", Condition.TREATMENT),
    ("", "2", Condition.TREATMENT),
    ("", "3", Condition.TREATMENT),
    ("", "4", Condition.TREATMENT),
]

if s.query(Classroom).count() == 0:
    for school, name, cond in PLAN:
        c = Classroom(school=school, name=name, condition=cond)
        s.add(c)
        s.flush()
        for n in range(1, TEAMS_PER_CLASS + 1):
            s.add(Team(classroom_id=c.id, number=n))
    s.commit()
    print(f"학급 {len(PLAN)}개 · 조 {len(PLAN)*TEAMS_PER_CLASS}개 생성")
else:
    print("이미 있음")
s.close()
