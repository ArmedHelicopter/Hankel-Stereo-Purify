"""Prepare a self-contained LLM audio listening-evaluation packet.

The generated packet is intentionally stateless: every prompt contains the
task, rubric, anonymous labels, and all audio attachments. That makes the
evaluation reusable across chats and API clients instead of relying on prior
conversation context.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, cast

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.prepare_whitening_listening_eval import read_audio  # noqa: E402

DEFAULT_SUMMARY = Path("data/processed/a09b/sum.json")
DEFAULT_ENV_FILES = (
    _REPO_ROOT / ".gemini" / "hsp_audio_eval.env",
    _REPO_ROOT / ".env.gemini.local",
    Path.home() / ".gemini/.env",
)
DEFAULT_TASK = (
    "Evaluate classical piano denoising outputs. Prefer low high-frequency "
    "hiss, preserved piano harmonics and note tails, stable stereo image, no "
    "watery/metallic/phasey/fluttering artifacts, and natural tone."
)
DEFAULT_MODEL = "gemini-2.5-pro-1m"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 " "Chrome/124 Safari/537.36"
)
MODEL_ENV_CANDIDATES = ("GEMINI_MODEL", "GOOGLE_GEMINI_MODEL")
BASE_URL_ENV_CANDIDATES = ("GOOGLE_GEMINI_BASE_URL", "GEMINI_BASE_URL")
API_KEY_ENV_CANDIDATES = (
    "GEMINI_API_KEY",
    "GOOGLE_GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)
LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _parse_env_value(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def load_env_file(path: Path) -> list[str]:
    """Load KEY=VALUE entries without overriding existing environment values."""
    expanded = path.expanduser()
    if not expanded.is_file():
        return []

    loaded: list[str] = []
    for line in expanded.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped.removeprefix("export ").strip()
        if "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            continue
        if key not in os.environ:
            os.environ[key] = _parse_env_value(raw_value)
            loaded.append(key)
    return loaded


def load_env_files(paths: list[Path] | tuple[Path, ...]) -> list[Path]:
    loaded_paths: list[Path] = []
    for path in paths:
        loaded = load_env_file(path)
        if loaded:
            loaded_paths.append(path.expanduser())
    return loaded_paths


def resolve_env(names: tuple[str, ...], default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def audio_metadata(path: Path) -> dict[str, Any]:
    data, samplerate = read_audio(path)
    return {
        "path": str(path),
        "samplerate": samplerate,
        "samples": int(data.shape[0]),
        "channels": int(data.shape[1]),
        "duration_seconds": float(data.shape[0] / samplerate),
    }


def load_summary_variants(
    summary_path: Path,
    variants: list[str] | None,
) -> dict[str, Path]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    paths = summary.get("paths")
    if not isinstance(paths, dict):
        raise ValueError(f"Missing paths object in {summary_path}")

    selected = variants or [
        name for name in paths if name != "orig" and not name.startswith("_")
    ]
    if not selected:
        raise ValueError("No variants selected")

    result: dict[str, Path] = {}
    base = summary_path.parent
    for name in selected:
        raw_path = paths.get(name)
        if not isinstance(raw_path, str):
            raise ValueError(f"Variant {name!r} not found in {summary_path}")
        path = Path(raw_path)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.is_file():
            fallback = (base / Path(raw_path).name).resolve()
            if fallback.is_file():
                path = fallback
            else:
                raise FileNotFoundError(path)
        result[name] = path
    return result


def parse_audio_arg(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--audio expects NAME=PATH")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("--audio variant name cannot be empty")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = path
    return result


def anonymous_items(
    variant_paths: dict[str, Path],
    seed: int,
) -> list[dict[str, Any]]:
    if len(variant_paths) > len(LABELS):
        raise ValueError(f"At most {len(LABELS)} audio items are supported")

    names = list(variant_paths)
    rng = random.Random(seed)
    rng.shuffle(names)

    items: list[dict[str, Any]] = []
    for label, name in zip(LABELS, names):
        path = variant_paths[name]
        items.append(
            {
                "label": label,
                "variant": name,
                "path": str(path),
                "metadata": audio_metadata(path),
            }
        )
    return items


def named_audio_items(audio_paths: dict[str, Path]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for name, path in audio_paths.items():
        items.append(
            {
                "name": name,
                "path": str(path),
                "metadata": audio_metadata(path),
            }
        )
    return items


def build_prompt(
    task: str,
    items: list[dict[str, Any]],
    references: list[dict[str, Any]] | None = None,
) -> str:
    labels = ", ".join(item["label"] for item in items)
    lines = [
        "# Stateless Audio Listening Evaluation",
        "",
        "You are evaluating denoised audio. This prompt is "
        "self-contained; do not rely on previous chat history.",
        "",
        f"Task: {task}",
        "",
    ]
    if references:
        reference_names = ", ".join(item["name"] for item in references)
        lines.extend(
            [
                "Named reference audio attachments are provided first: "
                f"{reference_names}. Use them as listening context only; do "
                "not score or rank them.",
                "",
            ]
        )
    lines.extend(
        [
            "Listen to every attached candidate audio item before judging. "
            f"The candidate labels are anonymous: {labels}. Do not infer "
            "quality from label order.",
            "",
        ]
    )
    if references:
        lines.extend(
            [
                "Judge candidates by comparing them against the named "
                "reference audio. Prefer candidates that remove noise from "
                "the noisy input while matching the clean reference when one "
                "is provided.",
                "",
                "Score each candidate on integer 0-10 scales:",
                "- noise_remaining: 0 means no audible remaining noise; "
                "10 means very intrusive noise.",
                "- clean_mismatch: 0 means matches the clean reference; "
                "10 means severe clean-signal loss or tonal mismatch.",
                "- stereo_image_damage: 0 means stable image; 10 means severe "
                "image drift, collapse, or decorrelation.",
                "- unnatural_artifacts: 0 means none; 10 means severe watery, "
                "metallic, phasey, pumping, or fluttering artifacts.",
                "- overall_quality: 0 means unusable; 10 means best balance.",
                "",
                "Return only strict JSON with this schema:",
                "{",
                '  "items": [',
                "    {",
                '      "label": "A",',
                '      "noise_remaining": 0,',
                '      "clean_mismatch": 0,',
                '      "stereo_image_damage": 0,',
                '      "unnatural_artifacts": 0,',
                '      "overall_quality": 0,',
                '      "notes": "short evidence-based notes"',
                "    }",
                "  ],",
                '  "ranking": ["A", "B"],',
                '  "winner": "A",',
                '  "summary": "brief final judgment"',
                "}",
                "",
                "For noise/artifact/damage scores, 0 is best and 10 is worst. "
                "For overall_quality, 10 is best.",
            ]
        )
    else:
        lines.extend(
            [
                "Score each candidate on integer 0-10 scales:",
                "- hiss_noise: 0 means no audible hiss; 10 means very intrusive hiss.",
                "- musical_damage: 0 means no audible musical loss; "
                "10 means severe loss.",
                "- unnatural_artifacts: 0 means none; 10 means severe watery, "
                "metallic, "
                "phasey, pumping, or fluttering artifacts.",
                "- overall_quality: 0 means unusable; 10 means best balance.",
                "",
                "Prefer the output that best balances noise reduction with preserved "
                "transients, harmonics, note tails, stereo stability, and "
                "natural tone.",
                "",
                "Return only strict JSON with this schema:",
                "{",
                '  "items": [',
                "    {",
                '      "label": "A",',
                '      "hiss_noise": 0,',
                '      "musical_damage": 0,',
                '      "unnatural_artifacts": 0,',
                '      "overall_quality": 0,',
                '      "notes": "short evidence-based notes"',
                "    }",
                "  ],",
                '  "ranking": ["A", "B"],',
                '  "winner": "A",',
                '  "summary": "brief final judgment"',
                "}",
            ]
        )
    attachment_order = (
        "named references first, then one anonymous candidate per label."
        if references
        else "one anonymous candidate per label."
    )
    lines.extend(
        ["", f"Audio attachments follow in the same request: {attachment_order}"]
    )
    return "\n".join(lines) + "\n"


def reference_audio_content_item(item: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(item["path"])
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return [
        {
            "type": "text",
            "text": (
                f"Reference audio {item['name']}: named context "
                f"{item['name']}. Duration "
                f"{item['metadata']['duration_seconds']:.3f}s. Do not score "
                "or rank this reference."
            ),
        },
        {
            "type": "input_audio",
            "input_audio": {
                "data": encoded,
                "format": path.suffix.lower().lstrip(".") or "wav",
            },
        },
    ]


def audio_content_item(item: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(item["path"])
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return [
        {
            "type": "text",
            "text": (
                f"Audio {item['label']}: anonymous candidate "
                f"{item['label']}. Duration "
                f"{item['metadata']['duration_seconds']:.3f}s."
            ),
        },
        {
            "type": "input_audio",
            "input_audio": {
                "data": encoded,
                "format": path.suffix.lower().lstrip(".") or "wav",
            },
        },
    ]


def build_openai_payload(
    *,
    model: str,
    prompt: str,
    items: list[dict[str, Any]],
    temperature: float,
    references: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for item in references or []:
        content.extend(reference_audio_content_item(item))
    for item in items:
        content.extend(audio_content_item(item))
    return {
        "model": model,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
    }


def write_packet(
    *,
    output_dir: Path,
    task: str,
    seed: int,
    model: str,
    items: list[dict[str, Any]],
    temperature: float,
    references: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(task, items, references=references)
    payload = build_openai_payload(
        model=model,
        prompt=prompt,
        items=items,
        temperature=temperature,
        references=references,
    )
    manifest = {
        "task": task,
        "seed": seed,
        "model": model,
        "transport": "openai-compatible-chat-audio",
        "references": references or [],
        "items": items,
        "public_mapping": {
            item["label"]: {
                "duration_seconds": item["metadata"]["duration_seconds"],
                "samplerate": item["metadata"]["samplerate"],
                "channels": item["metadata"]["channels"],
            }
            for item in items
        },
        "answer_key": {item["label"]: item["variant"] for item in items},
    }

    paths = {
        "manifest": output_dir / "manifest.json",
        "prompt": output_dir / "prompt.md",
        "payload": output_dir / "openai_chat_audio_payload.json",
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    paths["prompt"].write_text(prompt, encoding="utf-8")
    paths["payload"].write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return paths


def call_openai_compatible(
    *,
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": DEFAULT_USER_AGENT,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Provider HTTP {exc.code}: {detail}") from exc
    return cast(dict[str, Any], json.loads(body))


def main(argv: list[str] | None = None) -> None:
    env_parser = argparse.ArgumentParser(add_help=False)
    env_parser.add_argument(
        "--env-file",
        action="append",
        default=[],
        type=Path,
        help=(
            "Local env file with Gemini/OpenAI-compatible provider settings. "
            "Explicit shell env vars still take priority."
        ),
    )
    env_args, _ = env_parser.parse_known_args(argv)
    load_env_files([*env_args.env_file, *DEFAULT_ENV_FILES])

    parser = argparse.ArgumentParser(description=__doc__, parents=[env_parser])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    source.add_argument(
        "--audio",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Audio candidate; repeat for multiple items.",
    )
    parser.add_argument(
        "--reference-audio",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Named reference audio used as listening context but not scored; "
            "repeat for clean/noisy references."
        ),
    )
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument(
        "--model",
        default=resolve_env(MODEL_ENV_CANDIDATES, DEFAULT_MODEL),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--call", action="store_true")
    parser.add_argument(
        "--base-url",
        default=resolve_env(BASE_URL_ENV_CANDIDATES),
        help=(
            "OpenAI-compatible base URL, for example https://host/v1. "
            "Defaults to GOOGLE_GEMINI_BASE_URL or GEMINI_BASE_URL."
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default=API_KEY_ENV_CANDIDATES[0],
        help=(
            "Environment variable that contains the provider API key. "
            "Fallbacks include GEMINI_API_KEY, GOOGLE_GEMINI_API_KEY, "
            "and GOOGLE_API_KEY."
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args(argv)

    if args.audio:
        variant_paths = parse_audio_arg(args.audio)
        default_output = Path("data/processed/llm_audio_eval")
    else:
        variant_paths = load_summary_variants(args.summary, args.variants)
        default_output = args.summary.parent / "llm_audio_eval"

    output_dir = args.output_dir or default_output
    items = anonymous_items(variant_paths, args.seed)
    references = named_audio_items(parse_audio_arg(args.reference_audio))
    paths = write_packet(
        output_dir=output_dir,
        task=args.task,
        seed=args.seed,
        model=args.model,
        items=items,
        temperature=args.temperature,
        references=references,
    )
    print(f"Wrote {paths['manifest']}")
    print(f"Wrote {paths['prompt']}")
    print(f"Wrote {paths['payload']}")

    if not args.call:
        return

    if not args.base_url:
        raise SystemExit(
            "Missing --base-url, GOOGLE_GEMINI_BASE_URL, or GEMINI_BASE_URL"
        )
    api_key_names = tuple(dict.fromkeys((args.api_key_env, *API_KEY_ENV_CANDIDATES)))
    api_key = resolve_env(api_key_names)
    if not api_key:
        joined = ", ".join(api_key_names)
        raise SystemExit(f"Missing API key env var; tried: {joined}")
    payload = json.loads(paths["payload"].read_text(encoding="utf-8"))
    response = call_openai_compatible(
        base_url=args.base_url,
        api_key=api_key,
        payload=payload,
        timeout_seconds=args.timeout_seconds,
    )
    response_path = output_dir / "response.json"
    response_path.write_text(
        json.dumps(response, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {response_path}")


if __name__ == "__main__":
    main()
