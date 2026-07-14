from app.config import settings
from app.generators.anthropic_gen import AnthropicGenerator
from app.generators.base import Generator
from app.generators.null_gen import NullGenerator
from app.generators.openai_gen import OpenAIGenerator

_REGISTRY = {
    "anthropic": AnthropicGenerator,
    "openai": OpenAIGenerator,
    "null": NullGenerator,
}


def get_generator() -> Generator:
    key = settings.GENERATOR.lower().strip()
    if key not in _REGISTRY:
        raise ValueError(f"알 수 없는 생성기: {key}. 가능: {list(_REGISTRY)}")
    if key == "null":
        import warnings
        warnings.warn(
            "NullGenerator는 디버그용이다. 통제 조건이 아니다. "
            "실제 수업에서는 반드시 실모델을 쓴다.",
            stacklevel=2,
        )
    return _REGISTRY[key]()
