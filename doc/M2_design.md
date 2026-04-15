# M2 Design

## 1. 目标

M2 从这个阶段开始调整方向。

M2 要解决三件事：

1. 先把 `QwenImage` 做成 Forge 自己拥有的第一条 diffusion model-core 样板线。
2. 在不推翻 M1 训练骨架的前提下，把 Forge 的并行抽象扩成可表达 `FSDP + SP` 组合。
3. 为未来和 `sglang-diffusion` 做训推对齐预留正确边界。

M2 的核心判断是：

- Forge 长期目标是独立训练引擎。
- Forge 不能把执行时模型 ownership 长期放在 `diffusers`。
- `sglang-diffusion` 更适合作为模型侧抽象参考，而不是直接依赖对象。
- `diffusers` 仍然重要，但角色降级为：
  - config / checkpoint 来源
  - 数值对齐 oracle
  - 少量稳定底层组件来源

M2 因此不再把：

- `QwenImage + native diffusers SP`

视作目标架构，而把它视作：

- bring-up / benchmark / alignment 参考路径

真正的目标架构变成：

- `Forge-owned QwenImageDiT + Forge SP strategy + diffusers checkpoint bridge`

---

## 2. 非目标

M2 暂时不做：

- 不实现 `TP`
- 不实现完整的 `SP + TP + FSDP` 三维混合并行
- 不把所有模型家族都迁成自有 DiT
- 不直接 import `sglang.multimodal_gen` 作为 Forge 运行时依赖
- 不把 Forge 变成推理系统
- 不引入 `Req`、pipeline stage、scheduler、ZMQ、FastAPI 这类推理抽象

M2 只先把：

- `QwenImage`

做成第一条自有 model-core 样板线。

---

## 3. 设计原则

### 3.1 Forge 既拥有训练语义，也逐步拥有模型执行语义

M1 的默认前提是：

- Forge 拥有训练语义
- `diffusers` 拥有模型执行语义

M2 从这里开始改变：

- Forge 继续拥有 trainer / objective / batch contract
- Forge 开始拥有 priority model family 的执行时模型实现

也就是说，M2 之后的核心优先级变成：

- `trainer is Forge-owned`
- `DiT core is Forge-owned`
- `checkpoint bridge can still depend on diffusers`

### 3.2 `sglang` 是抽象参考，不是运行时依赖

M2 不直接依赖：

- `sglang.multimodal_gen.runtime`

但要明确参考它的做法：

- `BaseDiT` 这种最小模型 contract
- 自有 model registry
- 自有 attention / linear / norm / rope 组合
- 自有 family-specific DiT class

不借它的上层：

- request object
- pipeline stages
- scheduler
- serving stack

### 3.3 `diffusers` 是桥，不是执行真源

M2 允许依赖 `diffusers` 的内容：

- `model_index.json`
- 组件 config
- checkpoint 权重 layout
- 少量稳定叶子模块
- 对齐和回归测试 oracle

M2 不再把下面这些当成长期主路径：

- 官方 `Transformer2DModel` 类作为训练 backbone
- `set_attn_processor`
- `enable_parallelism`
- `_cp_plan`
- pipeline forward

### 3.4 优先级从 `native > patched > mirrored` 改成 `owned core > bridge`

旧思路是：

1. native
2. patched
3. mirrored

M2 的新思路是：

1. 对 priority family，优先上 Forge-owned core
2. `diffusers-native` 只作为验证路径保留
3. bridge 和 oracle 始终收敛到 runtime 边界

对当前阶段，这个 priority family 就是：

- `QwenImage`

### 3.5 并行是模型核心能力，不是上游框架赐予的能力

如果 Forge 最终要做：

- `SP`
- `FSDP + SP`
- 将来的 `SP + TP + FSDP`

那么并行能力必须依附在 Forge 自己的 model-core 和 strategy 层上，而不是绑定：

- `diffusers native entrypoint`

否则未来模型 ownership 依旧外移。

### 3.6 训练默认值优先考虑 correctness 和可迁移性

M2 第一条 sequence parallel 训练基线仍建议是：

- `algorithm="ulysses"`

原因不变：

- 实现最简单
- backward 路径最适合作为第一版训练基线
- 更利于后续迁移到自有 model-core

`USP`、`ring`、`unified` 先不作为第一版默认。

---

## 4. 为什么要在 M2 就改变方向

当前如果继续把 `QwenImage` 主要建在 `diffusers-native` 路线上，会带来三个长期问题。

### 4.1 训推对齐只能做到 checkpoint 对齐，做不到执行图对齐

如果上游推理长期走：

- `sglang-diffusion`

而训练长期走：

- `diffusers` 官方模型类

那么即使权重兼容，下面这些也会逐渐分叉：

- attention 语义
- rope 语义
- mask 语义
- linear / norm 实现
- sequence parallel 行为
- future kernel / backend 路径

这对训推对齐是结构性风险。

### 4.2 ownership 会继续外移到 `diffusers`

一旦训练主路径继续依赖：

- 官方模型类
- processor seam
- native parallel seam

那么 Forge 的演化节奏会继续受 `diffusers` 影响，包括：

- 类结构变化
- 版本漂移
- 某些 family 接入先后顺序
- 内部接口可见性

### 4.3 越晚自建 model-core，迁移成本越高

如果先让：

- `QwenImage`
- `SD3`
- `Flux`

都以不同程度绑定在 `diffusers` 上，再回头统一迁移，自然比从第一条 family 开始就把抽象拉正更贵。

因此 M2 是适合开始转向的时间点。

---

## 5. Forge 最终想要的边界

M2 的长期边界应该是：

- Forge 是独立训练引擎
- `sglang` 是独立推理引擎
- 两边都不强绑定彼此
- 但两边在模型核心层尽量收敛

这意味着正确的抽象不是：

- Forge import sglang runtime

也不是：

- Forge 继续完全依赖 diffusers model class

而是：

- Forge 自己维护 model-core
- model-core 的设计参考 `sglang-diffusion`
- 未来如果需要，可以把 Forge model-core 和 sglang model-core 进一步收敛或抽成 shared layer

---

## 6. M1 的缺口

M1 只够表达：

- `SD3 + FSDP1`

对新的 M2 方向来说有六个主要缺口。

### 6.1 `create_core()` 只支持 `sd3`

模型家族、runtime、objective 仍然是硬编码的。

### 6.2 `ParallelConfig` 只有 `dp_mode`

它不足以表达：

- sequence parallel 是否开启
- SP 算法
- parameter parallel degree
- sequence parallel degree

### 6.3 `DenoiseBatch` 还不够承载 model-family extras

`QwenImage` 至少需要：

- `encoder_hidden_states_mask`
- `img_shapes`
- 未来可能还要：
  - image-side condition
  - guidance metadata
  - packed token metadata

### 6.4 `ModelArchitecture` 仍默认暗示“返回外部模型类”

如果 `build_model()` 继续等价于：

- `diffusers official model class`

那么 execution truth 仍然不在 Forge。

### 6.5 缺少显式 checkpoint bridge

M1 里 `load_weights()` 还是 runtime 内部手工做的事情。  
M2 之后这层必须收敛成明确的职责，因为：

- model-core 是自有类
- 权重来源可能是 HF/diffusers checkpoint

### 6.6 缺少真正的 model-core registry

Forge 需要注册的不只是：

- `model_family -> runtime`

还包括：

- `architecture name -> owned model core class`

---

## 7. M2 核心抽象

M2 仍保留 M1 的大骨架：

- `DenoiseBatch`
- `ModelArchitecture`
- `ModelRuntime`
- `TrainingObjective`
- `ParallelConfig`
- `ParallelRuntime`

但新增或升级下面这些层：

- `ModelRegistry`
- `ModelCoreRegistry`
- `CheckpointAdapter`
- `ParallelPlan`
- `ParallelStrategy`

### 7.1 `DenoiseBatch`：保留，但坚持 `model_extras`

M2 继续保留 `DenoiseBatch` 作为训练统一输入 contract。

建议形态：

```python
@dataclass
class DenoiseBatch:
    latents: torch.Tensor
    prompt_embeds: torch.Tensor
    pooled_embeds: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None
    noise: torch.Tensor | None = None
    sample_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    model_extras: dict[str, Any] = field(default_factory=dict)
```

设计意图：

- 顶层字段只承载跨 family 稳定的训练语义
- family-specific forward kwargs 都放进 `model_extras`

`QwenImage` 当前最重要的 extras：

- `encoder_hidden_states_mask`
- `img_shapes`
- `attention_kwargs`
- `guidance`

### 7.2 `ModelArchitecture`：保留，但不再默认代表外部模型类

`ModelArchitecture` 仍然负责：

- condition schema
- 静态并行能力声明
- checkpoint 参数映射规则
- 构建未加载权重的模型

但 M2 的关键变化是：

- `build_model()` 长期目标应返回 Forge-owned model-core class

而不是：

- 外部 framework 官方类

### 7.3 新增 `ModelCoreRegistry`

Forge 需要一层专门的 registry，把：

- family-specific architecture identifier

映射到：

- Forge-owned DiT class

建议：

```python
register_model_core("qwen_image", QwenImageDiT)
register_model_core("sd3", StableDiffusion3DiT)
register_model_core("flux", FluxDiT)
```

这层和 `ModelRegistry` 不同：

- `ModelRegistry` 负责 family -> runtime / objective
- `ModelCoreRegistry` 负责 family -> owned model class

### 7.4 新增 `CheckpointAdapter`

这是 M2 里最关键的新抽象之一。

建议职责：

1. 从 HF/diffusers config 读取必要结构信息
2. 把 checkpoint state dict 映射到 Forge-owned model-core 参数名
3. 负责必要的 shape / layout 转换
4. 作为对齐和回归测试的唯一 bridge

建议接口：

```python
class CheckpointAdapter(ABC):
    def load_config(self, model_name_or_path: str) -> dict[str, Any]: ...
    def build_model_config(self, raw_config: dict[str, Any]) -> Any: ...
    def load_state_dict(self, model_name_or_path: str) -> dict[str, torch.Tensor]: ...
    def remap_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...
```

`QwenImageCheckpointAdapter` 第一版应支持：

- 从 HF/diffusers `transformer/config.json` 读配置
- 从 `transformer` 子目录加载权重
- 把参数名映射到 Forge-owned `QwenImageDiT`

### 7.5 `ModelRuntime`：从“diffusers adapter”升级成“family adapter”

M2 不引入新的 `ModelAdapter` 名字。  
但 `ModelRuntime` 的语义要变。

它现在应该负责：

1. 构建 Forge-owned model-core
2. 通过 `CheckpointAdapter` 加载权重
3. 把 `raw_batch` 标准化成 `DenoiseBatch`
4. 把 `DenoiseBatch` 映射成 model-core `forward` kwargs
5. 基于 `ModelArchitecture + ParallelConfig` 产出 `ParallelPlan`

也就是说，`ModelRuntime` 不再主要是：

- “调用 diffusers 官方模型类的壳”

而是：

- “Forge family runtime + bridge to external checkpoints”

### 7.6 `TrainingObjective`：抽象本身不变

这一层不因为 model-core ownership 变化而重写。

`TrainingObjective` 仍然只负责：

- timestep/noise 语义
- noisy input 构造
- target 构造
- loss 计算

因为：

- model-core ownership 是执行语义问题
- objective 是训练语义问题

### 7.7 `ParallelConfig`：保留双轴结构，但去掉 `native/patched` 语义中心

M2 仍建议保留“参数并行”和“序列并行”分离的配置方式：

```python
@dataclass(frozen=True)
class ParameterParallelConfig:
    mode: str = "none"      # none | fsdp1 | fsdp2
    degree: int = 1


@dataclass(frozen=True)
class SequenceParallelConfig:
    mode: str = "none"      # none | auto | enabled
    algorithm: str = "ulysses"   # ulysses | ring | usp
    degree: int = 1
    attention_backend: str = "flash"
    convert_to_fp32: bool = False


@dataclass(frozen=True)
class ParallelConfig:
    backend: str = "torch"
    mixed_precision: str = "bf16"
    parameter_parallel: ParameterParallelConfig = ParameterParallelConfig()
    sequence_parallel: SequenceParallelConfig = SequenceParallelConfig()
```

变化点：

- 不再把 `native` / `patched` 视为对用户暴露的主语义
- 主语义变成：
  - 我是否开 SP
  - 我用什么算法

具体是：

- `diffusers-native`
- 还是 `Forge-owned`

这是 runtime 的实现细节，不应成为最终架构的中心。

### 7.8 `ArchitectureParallelSpec`：声明 Forge-owned core 的并行能力

M2 不再以：

- `native_sequence_parallel`
- `patched_sequence_parallel`

作为长期中心。

更合适的形态是：

```python
@dataclass(frozen=True)
class SequenceParallelCapability:
    supported_algorithms: tuple[str, ...]
    default_algorithm: str
    required_batch_extras: tuple[str, ...] = ()
    supports_replicated_context_tokens: bool = False


@dataclass(frozen=True)
class ArchitectureParallelSpec:
    wrap_block_classes: tuple[str, ...] = ()
    no_shard_modules: tuple[str, ...] = ()
    shardable_inputs: dict[str, int] = field(default_factory=dict)
    replicate_inputs: tuple[str, ...] = ()
    sequence_parallel: SequenceParallelCapability | None = None
```

设计意图：

- 并行能力声明的是 Forge-owned core 能做什么
- 不再声明“是否有某个外部框架的 native seam”

`QwenImage` 例子：

- `supported_algorithms = ("ulysses", "ring", "usp")`
- `default_algorithm = "ulysses"`
- `required_batch_extras = ("encoder_hidden_states_mask", "img_shapes")`

### 7.9 `ParallelPlan`：保留

`ParallelPlan` 仍然需要，因为：

- `ModelRuntime` 决定模型适合什么并行策略
- `ParallelRuntime` 只负责执行 plan

建议继续保留：

```python
@dataclass(frozen=True)
class StrategySpec:
    kind: str
    config: dict[str, Any]


@dataclass(frozen=True)
class ParallelPlan:
    parameter_degree: int = 1
    sequence_degree: int = 1
    strategy_order: tuple[str, ...] = ()
    strategies: tuple[StrategySpec, ...] = ()
    required_batch_extras: tuple[str, ...] = ()
```

### 7.10 `ParallelStrategy`：保留，但第一条 SP strategy 变成 Forge-owned

M2 里的第一批策略应是：

- `FSDP1Strategy`
- `OwnedSequenceParallelStrategy`

而不是：

- `DiffusersNativeSPStrategy`

`OwnedSequenceParallelStrategy` 的职责：

- 挂接 Forge-owned attention backend
- 构造 SP process groups
- 做必要的 batch redistribution / metadata preparation
- 提供 sequence-parallel specific runtime state

---

## 8. `QwenImage` 作为 M2 第一条 owned model-core 样板线

`QwenImage` 适合作为第一条 owned line，原因是：

- 它对 text/image joint attention、mask、img_shapes、rope 的要求比较完整
- 它能逼出真正需要的 model-core contract
- 它已经有 `diffusers` 和 `sglang` 两边实现可参考

### 8.1 `QwenImageDiT` 的实现方向

M2 的 `QwenImageDiT` 应该是：

- Forge-owned `nn.Module`
- family-specific forward 语义明确
- attention / rope / linear / norm 尽量靠近 `sglang-diffusion`

不要求第一天就完全摆脱所有 `diffusers` 叶子依赖，但要求：

- 不再依赖 `QwenImageTransformer2DModel`

### 8.2 允许复用的底层组件

M2 初期可以接受复用少量 `diffusers` 叶子模块，例如：

- `Timesteps`
- `TimestepEmbedding`
- 某些稳定的 norm / FFN 模块

因为这些更像数学积木，不是执行 ownership 核心。

但下面这些应由 Forge 自己拥有：

- attention module
- joint attention 语义
- qkv projection 组织方式
- rope / token layout
- replicated linear / distributed linear
- sequence-parallel path

### 8.3 `QwenImageDiT.forward` 的训练 contract

`QwenImageDiT` 至少要支持下面这些输入：

```python
def forward(
    self,
    hidden_states,
    encoder_hidden_states,
    timestep,
    encoder_hidden_states_mask=None,
    img_shapes=None,
    guidance=None,
    attention_kwargs=None,
    return_dict=False,
):
    ...
```

训练侧 `ModelRuntime.prepare_forward_inputs()` 负责把：

- `DenoiseBatch`
- `objective_state`

映射成这组 kwargs。

### 8.4 `QwenImageCheckpointAdapter`

第一阶段必须把 checkpoint bridge 单独做清楚。

它要解决：

1. 读 HF/diffusers config
2. 构建 Forge-owned `QwenImageDiTConfig`
3. 读取权重
4. remap 参数名
5. 支持和 reference implementation 做逐参数对齐检查

### 8.5 `QwenImage` 的 sequence parallel 第一版

`QwenImage` 的第一版 SP 目标应是：

- Forge-owned `ulysses`

不是：

- `diffusers native ulysses`

也不是：

- 直接第一版上 `usp`

原因：

- `ulysses` 更适合作为训练 baseline
- 它能先把 core 的 qkv/rope/mask/layout 抽象拉正
- 等这条路径稳定后，再加 `usp`

### 8.6 `diffusers-native QwenImage` 的定位

M2 之后，这条路径不再是主架构。

它的定位应该改成：

- reference implementation
- regression oracle
- speed / numeric benchmark baseline

也就是说，它可以保留，但不应该继续驱动 Forge 核心抽象。

---

## 9. M2 对 `sglang-diffusion` 的借鉴边界

### 9.1 应该借鉴的内容

应借鉴：

- `BaseDiT` 最小 contract  
  [legacy/sglang/python/sglang/multimodal_gen/runtime/models/dits/base.py](/data/zhihengy/Forge/legacy/sglang/python/sglang/multimodal_gen/runtime/models/dits/base.py:15)
- runtime model registry 的思路  
  [legacy/sglang/python/sglang/multimodal_gen/runtime/models/registry.py](/data/zhihengy/Forge/legacy/sglang/python/sglang/multimodal_gen/runtime/models/registry.py:41)
- family-specific DiT 自己维护 attention / rope / linear / norm  
  [legacy/sglang/python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py](/data/zhihengy/Forge/legacy/sglang/python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py:16)  
  [legacy/sglang/python/sglang/multimodal_gen/runtime/models/dits/flux.py](/data/zhihengy/Forge/legacy/sglang/python/sglang/multimodal_gen/runtime/models/dits/flux.py:34)

### 9.2 不应该借鉴的内容

不应该借到 Forge 训练核心里：

- `Req`
- `ComposedPipelineBase`
- pipeline stage execution
- scheduler
- FastAPI/ZMQ
- request/job/store 语义

这些是推理系统抽象，不是训练系统抽象。

### 9.3 最佳关系

M2 最佳关系不是：

- Forge 依赖 `sglang runtime`

而是：

- Forge 自己维护 model-core
- 这个 model-core 的分层方式尽量和 `sglang-diffusion` 收敛
- 未来如果需要，再考虑把两边的 model-core 合并或抽 shared package

---

## 10. `FSDP + SP` 的组合方式

M2 仍然需要把 hybrid 并行作为目标。

### 10.1 策略顺序

对 owned model-core，推荐顺序仍然是：

1. 先应用 sequence-parallel 相关改造
2. 再应用 `FSDP1`

原因：

- SP 需要在原始模块层次上看见 attention / qkv / layout
- FSDP wrap 之后再做这些动作更麻烦

### 10.2 group 语义

M2 至少要能表达：

```text
world_size = parameter_degree * sequence_degree
```

即使第一版先只把单轴跑稳，这个语义也必须先定下来。

### 10.3 backward / step 归属

在 `FSDP1 + SP` 组合里：

- `SP` 不拥有 optimizer step
- `FSDP1` 拥有 backward、grad clip、optimizer step

这个分工不因 model-core ownership 改变。

---

## 11. M2 对 `diffusers` 的依赖边界

M2 仍然依赖 `diffusers`，但边界要比上一版更严格。

### 11.1 允许依赖

允许依赖：

- HF/diffusers checkpoint layout
- 配置格式
- `model_index.json`
- tokenizer / scheduler config
- 少量稳定叶子模块
- reference/oracle 对齐脚本

### 11.2 不允许外溢到 Forge 核心的内容

下面这些不应该外溢到 Forge 核心架构：

- 官方 DiT model class 作为训练主干
- pipeline class
- processor seam
- `enable_parallelism`
- `_cp_plan`
- example scripts

这些如果需要出现，也只能出现在：

- `CheckpointAdapter`
- 对齐脚本
- benchmark harness

### 11.3 `diffusers-native` 路线在 M2 的角色

这条路线仍然有价值，但角色变成：

- 检查 Forge-owned core 的数值对齐
- 检查权重映射是否正确
- 提供同任务 speed baseline

而不是：

- 作为 Forge 主运行时模型实现

---

## 12. M2 的执行流

### 12.1 初始化阶段

1. `create_core(model_family="qwen_image", ...)`
2. 从 `ModelRegistry` 创建 `QwenImageRuntime`
3. `model = core.model_runtime.build_model()`
4. `core.model_runtime.load_weights(model)`  
   这里内部通过 `QwenImageCheckpointAdapter`
5. `plan = core.model_runtime.make_parallel_plan(parallel_config)`
6. `core.parallel_runtime.setup(plan)`
7. `model = core.parallel_runtime.parallelize_model(model, plan)`
8. 构造 optimizer / scheduler

### 12.2 每个训练 step

1. `batch = core.model_runtime.canonicalize_batch(raw_batch)`
2. `batch = core.parallel_runtime.redistribute_batch(batch, plan)`
3. `objective_state = core.objective.prepare(batch, core.model_runtime, model)`
4. `model_inputs = core.model_runtime.prepare_forward_inputs(batch, objective_state)`
5. `outputs = model(**model_inputs)`
6. `loss, metrics, artifacts = core.objective.compute_loss(outputs, batch, objective_state)`
7. `core.parallel_runtime.backward(loss)`
8. `core.parallel_runtime.step(optimizer, scheduler)`

和 M1 相比，主要变化是：

- `build_model()` 现在目标是 Forge-owned core
- `load_weights()` 现在通过显式 checkpoint bridge
- `ParallelPlan` 现在以 Forge-owned SP strategy 为中心

---

## 13. 需要做的代码层改动

M2 最小实现建议分四批。

### 13.1 Phase A：先把 `QwenImage` 做成 owned model-core

需要做：

- 新增 Forge-owned `BaseDiT`
- 新增 `QwenImageDiT`
- 新增 `QwenImageDiTConfig`
- 新增 `QwenImageCheckpointAdapter`
- 新增 `ModelCoreRegistry`
- 调整 `QwenImageArchitecture.build_model()` 走 owned core
- 调整 `QwenImageRuntime.load_weights()` 走 adapter

完成标准：

- 单卡前向和训练 loss 能和 `diffusers QwenImage` 对齐
- 参数映射正确

### 13.2 Phase B：把 `QwenImage + Forge-owned Ulysses SP` 跑通

需要做：

- 实现 `OwnedSequenceParallelStrategy`
- 在 `QwenImageDiT` 内挂接 Forge-owned attention backend
- 让 `make_parallel_plan()` 产出自有 SP plan
- 保留 `diffusers-native` 路线作为 benchmark/oracle

完成标准：

- 多卡 SP 能跑通
- 与单卡 reference 数值行为稳定

### 13.3 Phase C：把 `FSDP1 + SP` 组合跑通

需要做：

- `ParallelRuntime.setup()` 支持 parameter/sequence 两个 degree
- `ParallelRuntime.parallelize_model()` 支持顺序应用多个 strategy
- `FSDP1Strategy` 从旧 runtime 中拆出来

完成标准：

- `QwenImage + FSDP1 + SP`

可训练、可保存、可恢复。

### 13.4 Phase D：把这套模式推广到第二个 family

建议下一条 family 是：

- `SD3`

原因：

- 它和 `QwenImage` 的 joint-attn 语义不同
- 能检验 Forge-owned model-core 是否足够通用

---

## 14. 最终结论

M2 的核心不再是：

- “把 `QwenImage` 接进 `diffusers-native SP`”

而是：

- 从这个阶段开始，把 Forge 从“训练语义 owned、模型执行语义外包”升级成“训练语义 owned、priority model-core 也 owned”

具体到当前阶段：

- Forge 保持独立训练引擎定位
- 不直接 import `sglang runtime`
- 借鉴 `sglang-diffusion` 的 DiT 抽象方式
- 自己迁出 Forge-owned `QwenImageDiT`
- 用 `diffusers` 作为 checkpoint/config bridge 和 reference oracle

换句话说，M2 的真正目标是：

- 先把 `QwenImage` 做成第一条 Forge-owned diffusion model-core
- 再在这条线上把 `SP` 和未来的 `FSDP + SP` 做正确

