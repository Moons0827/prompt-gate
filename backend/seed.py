"""연구 학급 6개를 만든다. 처치 3 / 통제 3. 각 학급을 조 6개로 나눈다.

로그인·제출·대화의 단위는 조다. 개인 이름도 출석번호도 저장하지 않는다 — 조 번호만.
학급당 25명 기준 조 6개 → 조당 약 4명.
파일명 규약: [학교]_[학급]_[조건]_[조번호]_[차시]
"""
from app.main import SessionLocal, Base, engine
from app.models import Classroom, Condition, Team

Base.metadata.create_all(engine)
s = SessionLocal()

TEAMS_PER_CLASS = 6

PLAN = [
    ("불곡초", "5-1", Condition.TREATMENT),
    ("불곡초", "5-2", Condition.CONTROL),
    ("불곡초", "6-1", Condition.TREATMENT),
    ("○○초", "5-1", Condition.CONTROL),
    ("○○초", "6-1", Condition.TREATMENT),
    ("○○초", "6-2", Condition.CONTROL),
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
