from collections.abc import Iterator

import torch
from transformers import AutoTokenizer

from config import EngineConfig
from model import Qwen3ForCausalLM
from loader import init_context, load_model
from sampler import Sampler

class Engine:
    def __init__(self,config:EngineConfig) -> None:
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = self._resolve_dtype()

        
        """
            1 加载模型分词器 即 tokenizer.json tokenizer_config_json
            2 分词器负责将prompt转成token_id给模型，模型输出token_id转回文本
            3 trust_remote_code 允许加载模型自定义配置代码
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model,trust_remote_code=True
        )

        """
            1 如果分词器没有pad_token，就用eos_token代替
            2 eos_token_id 是结束token的id
        """
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id


        """
            在临时上下文中构建模型，跳过随机初始化并指定精度和设备
            退出上下文时精度与设备已就位，无需再 to()，否则会复制整个模型
            eval() 切换评估模式，关闭训练逻辑
        """
        with init_context(self.dtype, self.device):
            self.model = Qwen3ForCausalLM(config.hf_config)     # 加载超参文件，权重直接按目标精度/设备分配
        load_model(self.model,config.model)                     # 加载权重文件，逐张量 copy_ 直接落到目标设备
        self.model = self.model.eval()

        """
            enforce_eager 表示强制使用原生eager执行，不进行编译
            reduce-overhead 降低python层算子调度开销，减少python-cuda来回切换的开销
        """
        if not config.enforce_eager and self.device.type == "cuda":
            self.model = torch.compile(self.model,mode="reduce-overhead")

        self.sampler = Sampler()

    def _resolve_dtype(self) -> torch.dtype:
        """
            1 如果dtype不为auto，则返回dtype
            2 如果dtype为auto，则返回bfloat16
        """
        if self.config.dtype != "auto":
            return getattr(torch, self.config.dtype)
        dt = getattr(self.config.hf_config, "dtype", None)
        return dt if isinstance(dt, torch.dtype) else torch.bfloat16

    @torch.inference_mode() # 推理模式，关闭梯度计算，不记录计算图
    def generate(
        self, 
        prompt: str | list[int], 
        max_new_tokens: int = 128, 
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> Iterator[tuple[str, int]]:
        """
            1 如果输入已经是token_id列表，则直接使用
            2 否则则调用分词器编码
            3 每次 yield (文本片段, 新增token数)：文本按批输出，token数才是真实增量
        """
        ids = (list(prompt) if isinstance(prompt,list) else self.tokenizer.encode(prompt)) 
        
        if not ids:
            raise ValueError("prompt must contain at least one token")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        ctx = self.config.context_len # 最大上下文长度
        if len(ids) > ctx:
            ids = ids[-ctx:] # 取最后ctx个元素

        def pack(v,dtype): # 普通输入包装为1维张量
            return None if v is None else torch.tensor([v],device=self.device,dtype=dtype)

        if temperature <= 0:
            # 贪心解码：采样参数全部关闭
            temp_t = top_p_t = top_k_t = None
        else:
            temp_t = pack(temperature,torch.float)
            top_p_t = pack(top_p,torch.float)
            top_k_t = pack(top_k,torch.long)


        # 预分配最大长度张量，避免循环中反复创建整个序列
        input_ids = torch.empty((1, ctx), device=self.device, dtype=torch.long)
        positions = torch.arange(ctx, device=self.device, dtype=torch.long).unsqueeze(0)
        input_ids[0, :len(ids)] = torch.tensor(ids, device=self.device, dtype=torch.long)
        cur_len = len(ids)

        # 增量解码缓冲：多字节字符可能被拆到相邻多个 token，
        # 只暂存尚不能解出完整字符的部分，其余立即产出。
        pending: list[int] = []

        for _ in range(max_new_tokens):
            # 切片是视图，无拷贝开销
            # 注意：当前每次仍传入完整序列，未使用 KV cache，
            # 因此 O(N²) 的重算仍存在。改造方式见文件末尾说明。
            h = self.model(input_ids[:, :cur_len], positions[:, :cur_len])
            # 前向传播，输出最后一层隐藏状态 h，shape=[1, seq_len, hidden_size]

            logits = self.model.compute_logits(h[:, -1:]).squeeze(1)
            # 1 h[:, -1:] 取序列最后一个 token，生成下一个 token 只用最后位置的隐状态
            # 2 compute_logits 后 shape=[1,1,vocab_size]，再删掉中间维度 shape=[1,vocab_size]

            next_id = int(self.sampler(logits, temperatures=temp_t, top_k=top_k_t, top_p=top_p_t).item())
            # item() 把 0 维标量张量取出，转成 Python 数字再转 int

            # eos 只用于终止，不参与输出
            if self.eos_token_id is not None and next_id == self.eos_token_id:
                break

            ids.append(next_id)
            pending.append(next_id)

            # 缓冲内字节能拼成完整字符时才产出，否则留到下一轮继续拼；
            # decode 在缓冲清空前完成，避免 yield 挂起后取到空列表。
            text = self.tokenizer.decode(pending, skip_special_tokens=False)
            if text and not text.endswith("\ufffd"):
                count = len(pending)
                pending.clear()
                yield text, count

            if cur_len >= ctx:
                break

            # 写入刚采样的 token 供下一轮前向使用，否则会读到未初始化内存
            input_ids[0, cur_len] = next_id
            cur_len += 1

        # 收尾：flush 剩余 token
        if pending:
            yield self.tokenizer.decode(pending, skip_special_tokens=False), len(pending)
