import importlib
import types


def test_mistral_keys_present() -> None:
    mod = importlib.import_module("medical_tokalign.src.convert_strict")
    assert isinstance(mod._EMBED_DICT, dict)
    assert isinstance(mod._LMHEAD_DICT, dict)
    assert "mistral" in mod._EMBED_DICT, "mistral embed mapping missing"
    assert "mistral" in mod._LMHEAD_DICT, "mistral lm_head mapping missing"


