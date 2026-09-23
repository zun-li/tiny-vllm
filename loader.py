import json
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from safetensors import safe_open

def _skip_reset_parameters(self: nn.Module) -> None:
    """用于替换 Linear/Embedding 的随机初始化"""
    pass


@contextmanager
def init_context(
    dtype: torch.dtype,
    device: torch.device | str = "cpu"
) -> Generator[None, None, None]:
    """建模型时的临时环境：设定精度与设备、跳过随机初始化，退出时还原。"""
    prev_dtype = torch.get_default_dtype()
    linear_reset = nn.Linear.reset_parameters
    embedding_reset = nn.Embedding.reset_parameters

    torch.set_default_dtype(dtype)
    nn.Linear.reset_parameters = _skip_reset_parameters
    nn.Embedding.reset_parameters = _skip_reset_parameters

    try:
        with torch.device(device):
            yield
    finally:
        torch.set_default_dtype(prev_dtype)
        nn.Linear.reset_parameters = linear_reset
        nn.Embedding.reset_parameters = embedding_reset


def _weight_files(folder: Path) -> list[Path]:
    """列出全部分片并按文件名排序，使读盘顺序与分片编号一致。"""
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        index_data = json.loads(index.read_text(encoding="utf-8"))
        weight_map = index_data["weight_map"]
        # 同一分片可能被多个权重名引用，先去重再排序
        return sorted({folder / name for name in weight_map.values()})

    files = sorted(folder.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors weights found in {folder}")
    return files


def safetensors_weights_iterator(hf_folder: str | Path, *, desc: str | None = "Loading weights") -> Iterator[tuple[str, torch.Tensor]]:
    """流式产出（权重名，张量），内存峰值仅为一个张量"""
    files = _weight_files(Path(hf_folder))
    total_bytes = sum(file.stat().st_size for file in files)

    with tqdm(total=total_bytes, desc=desc, unit="B", unit_scale=True, disable=desc is None) as bar:
        for file in files:
            with safe_open(file, framework="pt", device="cpu") as shard:
                for name in shard.keys():
                    tensor = shard.get_tensor(name)
                    bar.update(tensor.numel() * tensor.element_size())
                    yield name, tensor

        # safetensors 文件头也计入文件大小，最后把进度条补到 100%
        if bar.n < total_bytes:
            bar.update(total_bytes - bar.n)


def load_model(model: nn.Module, hf_folder: str | Path) -> nn.Module:
    """将目录中的 checkpoint 写入模型，名称映射与 QKV 融合由 load_weights 完成。"""
    model.load_weights(safetensors_weights_iterator(hf_folder))
    return model


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """将 loaded_weight 拷贝到 param，校验形状并转换 dtype。"""

    if loaded_weight.shape != param.shape:
        raise ValueError(f"shape mismatch: {tuple(param.shape)} != {tuple(loaded_weight.shape)}")

    if loaded_weight.dtype != param.dtype:
        loaded_weight = loaded_weight.to(param.dtype)

    # 拷贝数据
    with torch.no_grad():
        param.copy_(loaded_weight)
