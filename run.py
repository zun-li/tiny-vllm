"""tiny-vllm 本地模型推理入口。

用法:
    python run.py
    python run.py --prompt "你好"
    python run.py --model ~/huggingface/Qwen3-0.6B
    python run.py --config my_config.yaml
"""
import time
import argparse
from typing import Any
from pathlib import Path

import yaml

from config import EngineConfig
from engine import Engine

HERE = Path(__file__).resolve().parent

defaults = {
    "model": "~/huggingface/Qwen3-1.7B",
    "prompt": "Hello, my name is",
    "max_new_tokens": 512,
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 20,
    "enforce_eager": True,
    "dtype": "auto",
}


def load_config(path: Path, defaults: dict) -> dict:
    """读取 YAML 配置；文件不存在时直接使用默认值。"""
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    return {key: cfg.get(key, value) for key, value in defaults.items()}


def merge_cli(cfg: dict, args: Any, keys: tuple[str, ...]) -> None:
    """CLI 中显式传入的参数覆盖 YAML 配置。"""
    for key in keys:
        if (value := getattr(args, key, None)) is not None:
            cfg[key] = value


def run(cfg: dict) -> None:
    model_path = Path(cfg["model"]).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"本地模型目录不存在: {model_path}")
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"模型目录缺少 config.json: {model_path}")

    print(f"Loading {model_path} ...")
    engine = Engine(
        EngineConfig(
            model=str(model_path),
            enforce_eager=cfg["enforce_eager"],
            dtype=cfg["dtype"],
        )
    )

    prompt = cfg["prompt"]
    print(f"Prompt: {prompt}\n{prompt}", end="", flush=True)

    start = time.perf_counter()
    n = 0
    for text, count in engine.generate(
        prompt,
        max_new_tokens=cfg["max_new_tokens"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        top_k=cfg["top_k"],
    ):
        print(text, end="", flush=True)
        n += count
    elapsed = time.perf_counter() - start
    print(f"\n\n{n} tokens in {elapsed:.2f}s ({n / elapsed:.1f} tok/s)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    args = p.parse_args()

    config_path = Path(args.config) if args.config else HERE / "config.yaml"
    config = load_config(config_path, defaults)
    merge_cli(config, args, ("model", "prompt", "max_new_tokens", "temperature", "top_p", "top_k"))
    run(config)


if __name__ == "__main__":
    main()
