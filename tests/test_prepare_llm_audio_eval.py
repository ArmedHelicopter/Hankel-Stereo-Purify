import json
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from pytest import MonkeyPatch

import scripts.prepare_llm_audio_eval as llm_audio_eval
from scripts.prepare_llm_audio_eval import (
    anonymous_items,
    build_openai_payload,
    build_prompt,
    load_env_file,
    load_summary_variants,
    named_audio_items,
    write_packet,
)


def test_prepare_llm_audio_eval_packet_is_stateless(tmp_path: Path) -> None:
    audio_paths: dict[str, Path] = {}
    for idx, name in enumerate(("a75", "a80")):
        path = tmp_path / f"{name}.wav"
        sf.write(
            path,
            np.full((16, 2), 0.001 * (idx + 1), dtype=np.float64),
            8_000,
            format="WAV",
            subtype="FLOAT",
        )
        audio_paths[name] = path

    summary_path = tmp_path / "sum.json"
    summary_path.write_text(
        json.dumps(
            {
                "paths": {
                    "orig": str(tmp_path / "orig.wav"),
                    "a75": str(audio_paths["a75"]),
                    "a80": str(audio_paths["a80"]),
                }
            }
        ),
        encoding="utf-8",
    )

    variants = load_summary_variants(summary_path, ["a75", "a80"])
    items = anonymous_items(variants, seed=123)
    prompt = build_prompt("Judge balance.", items)
    payload = build_openai_payload(
        model="gemini-2.5-pro-1m",
        prompt=prompt,
        items=items,
        temperature=0.0,
    )

    assert "self-contained" in prompt
    assert "Return only strict JSON" in prompt
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert sum(part["type"] == "input_audio" for part in content) == 2

    paths = write_packet(
        output_dir=tmp_path / "packet",
        task="Judge balance.",
        seed=123,
        model="gemini-2.5-pro-1m",
        items=items,
        temperature=0.0,
    )
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert sorted(manifest["answer_key"].values()) == ["a75", "a80"]
    assert paths["payload"].is_file()


def test_reference_audio_is_context_but_not_candidate(tmp_path: Path) -> None:
    reference_path = tmp_path / "clean.wav"
    candidate_path = tmp_path / "candidate.wav"
    for path, value in ((reference_path, 0.002), (candidate_path, 0.001)):
        sf.write(
            path,
            np.full((16, 2), value, dtype=np.float64),
            8_000,
            format="WAV",
            subtype="FLOAT",
        )

    references = named_audio_items({"clean_reference": reference_path})
    items = anonymous_items({"candidate": candidate_path}, seed=123)
    prompt = build_prompt("Compare with clean.", items, references=references)
    payload = build_openai_payload(
        model="gemini-2.5-flash",
        prompt=prompt,
        items=items,
        temperature=0.0,
        references=references,
    )

    assert "clean_reference" in prompt
    assert "do not score or rank them" in prompt
    content = payload["messages"][0]["content"]
    assert sum(part["type"] == "input_audio" for part in content) == 2
    assert "Reference audio clean_reference" in content[1]["text"]
    assert "anonymous candidate A" in content[3]["text"]

    paths = write_packet(
        output_dir=tmp_path / "packet",
        task="Compare with clean.",
        seed=123,
        model="gemini-2.5-flash",
        items=items,
        temperature=0.0,
        references=references,
    )
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["references"][0]["name"] == "clean_reference"
    assert manifest["answer_key"] == {"A": "candidate"}


def test_env_file_loads_gemini_aliases_without_overriding_shell(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    env_path = tmp_path / "gemini.env"
    env_path.write_text(
        "\n".join(
            [
                "GEMINI_API_KEY=file-secret",
                "GEMINI_MODEL=file-model",
                "GOOGLE_GEMINI_BASE_URL=https://from-file.example/v1",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "https://shell.example/v1")

    loaded = load_env_file(env_path)

    assert set(loaded) == {"GEMINI_API_KEY", "GEMINI_MODEL"}
    assert llm_audio_eval.resolve_env(("GEMINI_API_KEY",)) == "file-secret"
    assert llm_audio_eval.resolve_env(("GEMINI_MODEL",)) == "file-model"
    assert (
        llm_audio_eval.resolve_env(("GOOGLE_GEMINI_BASE_URL",))
        == "https://shell.example/v1"
    )


def test_main_uses_env_file_for_model_default(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    audio_path = tmp_path / "candidate.wav"
    sf.write(
        audio_path,
        np.full((16, 2), 0.001, dtype=np.float64),
        8_000,
        format="WAV",
        subtype="FLOAT",
    )
    env_path = tmp_path / "provider.env"
    env_path.write_text("GEMINI_MODEL=gemini-from-file\n", encoding="utf-8")
    output_dir = tmp_path / "packet"

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GOOGLE_GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GOOGLE_GEMINI_BASE_URL", raising=False)
    monkeypatch.delenv("GEMINI_BASE_URL", raising=False)

    llm_audio_eval.main(
        [
            "--env-file",
            str(env_path),
            "--audio",
            f"a75={audio_path}",
            "--output-dir",
            str(output_dir),
        ]
    )

    payload = json.loads(
        (output_dir / "openai_chat_audio_payload.json").read_text(encoding="utf-8")
    )
    assert payload["model"] == "gemini-from-file"


def test_call_uses_env_file_provider_config(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    audio_path = tmp_path / "candidate.wav"
    sf.write(
        audio_path,
        np.full((16, 2), 0.001, dtype=np.float64),
        8_000,
        format="WAV",
        subtype="FLOAT",
    )
    env_path = tmp_path / "provider.env"
    env_path.write_text(
        "\n".join(
            [
                "GEMINI_API_KEY=file-secret",
                "GEMINI_MODEL=gemini-call-model",
                "GOOGLE_GEMINI_BASE_URL=https://from-file.example/v1",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "packet"
    captured: dict[str, str] = {}

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GOOGLE_GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GOOGLE_GEMINI_BASE_URL", raising=False)
    monkeypatch.delenv("GEMINI_BASE_URL", raising=False)

    def fake_call_openai_compatible(
        *,
        base_url: str,
        api_key: str,
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        captured["base_url"] = base_url
        captured["api_key"] = api_key
        captured["model"] = str(payload["model"])
        captured["timeout_seconds"] = str(timeout_seconds)
        return {"ok": True}

    monkeypatch.setattr(
        llm_audio_eval,
        "call_openai_compatible",
        fake_call_openai_compatible,
    )

    llm_audio_eval.main(
        [
            "--env-file",
            str(env_path),
            "--audio",
            f"a75={audio_path}",
            "--output-dir",
            str(output_dir),
            "--call",
        ]
    )

    assert captured == {
        "base_url": "https://from-file.example/v1",
        "api_key": "file-secret",
        "model": "gemini-call-model",
        "timeout_seconds": "180.0",
    }
    assert json.loads((output_dir / "response.json").read_text()) == {"ok": True}


def test_call_openai_compatible_sends_browser_user_agent(
    monkeypatch: MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"ok": true}'

    def fake_urlopen(
        request: urllib.request.Request,
        timeout: float,
    ) -> FakeResponse:
        captured["timeout"] = timeout
        captured["headers"] = {
            key.lower(): value for key, value in request.header_items()
        }
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    response = llm_audio_eval.call_openai_compatible(
        base_url="https://provider.example/v1",
        api_key="secret",
        payload={"model": "gemini-test", "messages": []},
        timeout_seconds=12.0,
    )

    assert response == {"ok": True}
    assert captured["timeout"] == 12.0
    assert captured["headers"]["authorization"] == "Bearer secret"
    assert captured["headers"]["accept"] == "application/json"
    assert "Mozilla/5.0" in captured["headers"]["user-agent"]
