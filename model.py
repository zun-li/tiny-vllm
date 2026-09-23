from collections.abc import Iterable

import torch
from torch import nn
from transformers.models.qwen3 import Qwen3Config

from layers.rmsnorm import RMSNorm
from layers.rotary import RotaryEmbedding
from layers.attention import Attention
from layers.activation import SiluAndMul
from loader import default_weight_loader


class Qwen3MLP(nn.Module):
    """SwiGLU 前馈网络"""
    def __init__(self,config:Qwen3Config) -> None:
        # 1 Qwen3Config 超参数配置类，模型固有配置，不可训练参数
        # 2 例如 hidden_size 隐藏维度 intermediate MLP中间维度，升维维度 num_hidden_layers FNN层数
        # 3 num_attention_heads Q头数量 num_key_value_heads GQA的KV头数 等等
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        # 1 Linear 全连接线性层，本质是做矩阵乘法 Y = XW^T + b 这里的输入就是 W 的shape
        # 2 其中 b就是bias，也就是偏置向量，千问3不带偏置
        # 3 gate_proj 门控算子 对输入张量升维并生成生成门控信号 从[B,S,hidden_size]到[B,S,intermediate_size]
        # 4 up_proj 升维算子，做特征升维 down_proj 降维算子
        self.act_fn = SiluAndMul()

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        # 两个升维投影各自独立，直接 silu(gate) * up
        return self.down_proj(self.act_fn(self.gate_proj(x),self.up_proj(x)))

class Qwen3Attention(nn.Module):
    """Qwen3 自注意力: FusedQKV -> QK Norm -> RoPE -> GQA -> Linear"""
    def __init__(self,config:Qwen3Config) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads # Q头数
        self.num_kv_heads = config.num_key_value_heads # KV头数
        self.head_dim = getattr(config,"head_dim",None) or config.hidden_size // self.num_heads # 头数
        # 小模型可能显式给head_dim,如果不给自行计算即可
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = nn.Linear(
            config.hidden_size,self.q_size + 2 * self.kv_size,bias=False
        )  # 一次线性同时产出 Q|K|V，加载时再按切片写入三段
        self.o_proj = nn.Linear(self.q_size,config.hidden_size,bias=False)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            self.head_dim,
            config.max_position_embeddings,
            float(getattr(config,"rope_theta",1_000_000)),
        ) # RoPE 按照绝对位置旋转Q, K; Qwen3 全维旋转 (rotary_dim == head_dim)
        self.attn = Attention(self.num_heads,self.head_dim,self.num_kv_heads)
        self.q_norm = RMSNorm(self.head_dim,eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim,eps=config.rms_norm_eps)
        # QK-Norm：按每个 head 的 head_dim 做 RMSNorm

    def forward(self,position:torch.Tensor,hidden_states:torch.Tensor) -> torch.Tensor:
        # 1 一次投影 -> 三段切分 -> 分别做 norm 和 RoPE -> 重新组装 -> 线性投影
        # 2 hidden_states 整批张量，网络在层与层之间传递的核心数据
        q,k,v = self.qkv_proj(hidden_states).split(
            [self.q_size,self.kv_size,self.kv_size],dim=-1
        ) # 算出拼接向量后切分

        q = self.q_norm(q.unflatten(-1, (self.num_heads, self.head_dim)))
        # 1 unflatten 维度拆分，[B,S,H*D] → [B,S,H,D]
        # 2 之后对q的最后一维也就是 head_dim 做归一化，逐 head 做 norm
        # 3 归一化后 flatten 把最后两维合并 shape还原,拍平回 [B,S,H*D]；
        k = self.k_norm(k.unflatten(-1, (self.num_kv_heads, self.head_dim)))
        q,k = self.rotary_emb(position,q,k) # RoPE
        # v 不做norm和RoPE

        b,s,_,_ = q.shape
        v = v.view(b,s,self.num_kv_heads,self.head_dim)
        # 等价于 q.unflatten(-1,(-1,self.head_dim)) 拆分成[B,S,H,D]，方便attn操作
        return self.o_proj(self.attn(q,k,v).flatten(-2))
        # 1 为什么不直接返回 attn 的结果，而要再过一层 o_proj
        # 2 因为 q_size 和 hidden_size 不一定相等，SDPA 也不必用完 token 的所有维度
        # 3 且 SDPA 只在每个头内做 token 间运算，o_proj 负责跨头信息融合

class Qwen3DecoderLayer(nn.Module):
    """Qwen3 解码器层: 自注意 -> 残差 -> MLP -> 残差"""
    def __init__(self,config:Qwen3Config) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        # 输入层 norm 与残差融合；输出层 norm 与残差融合
        self.input_layernorm = RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,eps=config.rms_norm_eps)

    def forward(
            self,
            positions:torch.Tensor,
            hidden_states:torch.Tensor,
            residual:torch.Tensor | None,
    ) -> tuple[torch.Tensor,torch.Tensor | None]:
        # 首层：residual 取 embedding；之后各层在 RMSNorm 内完成 h+residual 再归一化
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states,residual = self.input_layernorm(hidden_states,residual)
        hidden_states = self.self_attn(positions,hidden_states)
        # Attn 输出再归一化,然后过 MLP
        hidden_states,residual = self.post_attention_layernorm(hidden_states,residual)
        return self.mlp(hidden_states),residual

class Qwen3Model(nn.Module):
    def __init__(self,config:Qwen3Config) -> None:
        """Qwen3 模型: Embed -> 堆叠Layers -> Norm"""
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size,config.hidden_size)
        # 1 vocab_size 模型词表总大小,存放token_id
        # 2 embed_tokens 词嵌入层，可查找可训练的权重矩阵，shape=[vocab_size,hidden_size]
        # 3 根据对应的token_id查找相对应的长度为hidden_size的向量
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        # 1 num_hidden_layers 解码器层数 实例化出层数个独立的Qwen3DecoderLayer对象
        # 2 nn.ModuleList PyTorch容器，专门用来存放一堆nn.Module
        # 3 nn.ModuleList 会遍历内部每一个module，自动注册子模块到父模型
        self.norm = RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        # 最终归一化，放在全部解码器层跑完之后
    def forward(self,input_ids:torch.Tensor,positions:torch.Tensor) -> torch.Tensor:
        # residual 初始值为None,首层Layer直接用 embedding 作为残差
        h,residual = self.embed_tokens(input_ids),None
        for layer in self.layers: # 向前传播
            h,residual = layer(positions, h, residual)
        return self.norm(h,residual)[0]
        # 1 全部decoder层跑完之后执行最后一次残差相加和归一化
        # 2 norm返回的是二元组[h,residual]，取h做结果处理

class Qwen3ForCausalLM(nn.Module):
    """Qwen3模型顶层: backbone + LM head"""
    def __init__(self,config:Qwen3Config) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size,config.vocab_size,bias=False)
        # 1 lm_head 语言模型头，backbone输出的最终hidden_states只是hidden_size维的上下文语义向量，不是token概率
        # 2 lm_head 做线性矩阵乘法，把每一个token的向量，映射到整个词表的打分

    def forward(self,input_ids:torch.Tensor,positions:torch.Tensor) -> torch.Tensor:
        # 只算Backbone,推理引擎需要hidden state时再算LM head
        return self.model(input_ids,positions)

    def compute_logits(self,hidden_states:torch.Tensor) -> torch.Tensor:
        # 表征计算与词表投影分离,后续张量并行时lm_head可以单独切分
        return self.lm_head(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        # weights: (权重名, 权重张量) 序列；QKV 分片先合并，其余直接拷贝
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        attn0 = self.model.layers[0].self_attn
        # HF 的 q/k/v 按以下偏移写入本项目融合后的 qkv_proj
        qkv_slices = {
            "q_proj": (0, attn0.q_size),
            "k_proj": (attn0.q_size, attn0.kv_size),
            "v_proj": (attn0.q_size + attn0.kv_size, attn0.kv_size),
        }
        shards_seen: dict[str, int] = {}

        for name, w in weights:
            stem = name.removesuffix(".weight")
            leaf = stem.rsplit(".", 1)[-1]  # 取末级名，如 q_proj
            if leaf in qkv_slices:
                fused = stem.rsplit(".", 1)[0] + ".qkv_proj.weight"
                off, size = qkv_slices[leaf]
                # 切片写入融合权重，copy_ 同时完成精度转换
                with torch.no_grad():
                    params[fused][off : off + size].copy_(w)
                shards_seen[fused] = shards_seen.get(fused, 0) + 1
            elif name in params:
                default_weight_loader(params[name], w)
                loaded.add(name)

        # QKV 三片全部到齐才算加载完成，否则留给下方缺失检查报错
        for fused, seen in shards_seen.items():
            if seen == len(qkv_slices):
                loaded.add(fused)

        if "lm_head.weight" not in loaded and "model.embed_tokens.weight" in loaded:
            # 权重共享：embed_tokens 与 lm_head 绑定，复用已加载的词嵌入
            default_weight_loader(
                params["lm_head.weight"], params["model.embed_tokens.weight"]
            )
            loaded.add("lm_head.weight")
        missing = set(params) - loaded
        if missing:
            preview = ", ".join(sorted(missing)[:8])
            raise ValueError(f"checkpoint is missing {len(missing)} model weights: {preview}")
