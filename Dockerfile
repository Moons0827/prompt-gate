FROM python:3.12-slim

WORKDIR /app

# 의존성 먼저 설치 (레이어 캐시 활용)
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# 전체 코드 복사 — backend/ 와 frontend/ 둘 다 필요하다.
# (백엔드가 frontend/index.html 을 / 에서 함께 서빙한다)
COPY . .

WORKDIR /app/backend
ENV PYTHONUNBUFFERED=1

# 시드(학급·조 생성, 이미 있으면 건너뜀) 후 서버 시작. Render 가 $PORT 를 준다.
CMD python seed.py && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}
