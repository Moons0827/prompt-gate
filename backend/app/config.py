"""설정.

★ 모델 ID는 반드시 제공사 문서에서 직접 확인하고 .env 에 적는다.
  코드에 기본값을 박아 두지 않는다 — 검증되지 않은 모델 ID를 쓰면
  "어떤 모델로 실험했는가"를 논문에 못 쓴다.

★ 날짜가 박힌 스냅샷 ID를 쓴다. 'latest' 별칭을 쓰지 않는다.
  4~12월 프로젝트인데 그 사이에 별칭이 가리키는 모델이 바뀌면 재현이 불가능하다.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # anthropic | openai | null
    GENERATOR: str = "null"

    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = ""      # 예: claude-sonnet-5  (문서에서 확인 후 기입)

    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = ""         # 문서에서 확인 후 기입

    TEMPERATURE: float = 0.0       # 재현성을 위해 0 고정
    MAX_TOKENS: int = 600

    # 초기화(활동 데이터 삭제) 시 요구하는 코드. 배포 시 env 로 바꿀 수 있다.
    RESET_CODE: str = "reset"
    TIMEOUT_S: float = 30.0

    DB_URL: str = "sqlite:///./promptgate.db"

    class Config:
        env_file = ".env"


settings = Settings()


def require_model(name: str, value: str) -> str:
    if not value.strip():
        raise RuntimeError(
            f"{name} 이 비어 있다. .env 에 모델 ID를 적어라.\n"
            "제공사 문서에서 현재 사용 가능한 모델 ID를 확인하고, "
            "날짜가 박힌 스냅샷 ID를 쓴다. 'latest' 별칭은 쓰지 않는다."
        )
    if "latest" in value.lower():
        raise RuntimeError(
            f"{name}='{value}' — 'latest' 별칭은 쓸 수 없다. "
            "기간 중 모델이 바뀌면 재현이 불가능하다. 고정된 ID를 써라."
        )
    return value
