import os


def test_model_candidates_include_env_override():
    os.environ["NVIDIA_MODEL"] = "meta/llama-3.1-8b-instruct"
    from app import MODEL_FALLBACKS

    assert "meta/llama-3.1-8b-instruct" in MODEL_FALLBACKS


def test_model_candidates_keep_default_values():
    os.environ.pop("NVIDIA_MODEL", None)
    from app import MODEL_FALLBACKS

    assert MODEL_FALLBACKS
    assert all(isinstance(model, str) and model for model in MODEL_FALLBACKS)
