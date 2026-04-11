# M1 Design

## 1. 目标

M1 先只解决一件事：把 Forge 收敛成一套清晰的 SD3-only DiT training core，并且保持它既能独立训练，也能被 Miles 嵌入调用。

当前阶段边界：

- 模型只考虑 `SD3`
- 语义参考以 `sglang-diffusion` 的 denoise 边界为准
- 并行只先做 `FSDP1`
- 先不讨论 `FSDP2 / SP / USPAttention`
- 最重要的验证目标是和 `diffusers` 的训练语义对齐

---

## 2. 六个核心抽象

### 2.1 `DenoiseBatch`

定义：

- Forge 内部统一的 denoise-step 输入 contract

职责：

- 承载进入 Forge 后的标准化训练输入
- 隔离上游富态数据和 Forge 内部训练语义

输入：

- 上游 raw batch
- 由 `ModelRuntime.canonicalize_batch()` 生成

输出：

- 给 `TrainingObjective`
- 给 `ModelRuntime.prepare_forward_inputs()`
- 给并行计划生成逻辑

SD3 例子：

- 原始字段可能是 `latent / prompt_embed / pooled_prompt_embed`
- 进入 Forge 后统一成 `latents / prompt_embeds / pooled_embeds`

---

### 2.2 `ModelArchitecture`

定义：

- 模型家族的静态定义

职责：

- 定义模型结构
- 定义条件输入 schema
- 定义参数映射规则
- 定义静态并行提示

输入：

- model config
- checkpoint 命名规则

输出：

- 未加载权重的模型结构
- 条件 schema
- 静态 parallel spec

SD3 例子：

- `SD3Architecture` 声明：
  - required fields: `latents`, `prompt_embeds`, `pooled_embeds`
  - optional fields: `timesteps`, `noise`, `attention_mask`, `image_embeds`
  - `latents -> hidden_states`
  - `prompt_embeds -> encoder_hidden_states`
  - `pooled_embeds -> pooled_projections`
  - wrap block 粒度为 `JointTransformerBlock`

说明：

- `ConditionSchema` 不是顶层抽象，而是 `ModelArchitecture` 的一部分

---

### 2.3 `ModelRuntime`

定义：

- 模型家族的动态执行壳

职责：

- build 模型实例
- load 权重和运行时辅助模块
- `raw_batch -> DenoiseBatch`
- `DenoiseBatch -> forward kwargs`
- 基于 `ModelArchitecture` 和 `ParallelConfig` 生成内部并行计划

输入：

- `ModelArchitecture`
- 权重来源
- raw batch
- `ParallelConfig`

输出：

- 运行时模型实例
- `DenoiseBatch`
- forward kwargs
- 内部并行计划对象

SD3 例子：

- `SD3Runtime` 从 diffusers checkpoint 读取 transformer 和 scheduler
- 把别名字段统一成 Forge 的 canonical batch 字段
- 生成 `hidden_states / timestep / encoder_hidden_states / pooled_projections`

---

### 2.4 `TrainingObjective`

定义：

- 训练目标语义

职责：

- 解析 timestep / noise
- 构造 noisy inputs
- 构造 target
- 计算 loss

输入：

- `DenoiseBatch`
- `ModelRuntime`
- model outputs

输出：

- objective state
- loss
- metrics / artifacts

SD3 例子：

- `FlowMatchObjective`:
  - 用 scheduler 解析 timestep
  - 用 sigma 构造 `noisy_latents`
  - 用 `noise - latents` 作为 target
  - 用 MSE 计算 loss

---

### 2.5 `ParallelConfig`

定义：

- 用户声明的并行意图

职责：

- 描述要使用的 backend 和并行模式
- 不携带模型语义

输入：

- CLI / yaml / launcher / Miles 配置

输出：

- 一个规范化的并行配置对象

当前阶段例子：

```python
ParallelConfig(
    backend="torch",
    dp_mode="fsdp1",
    mixed_precision="bf16",
)
```

---

### 2.6 `ParallelRuntime`

定义：

- 并行执行器

职责：

- 初始化 distributed 环境
- 根据内部并行计划并行化模型
- 根据内部并行计划重分布 batch
- backward / step / grad clip
- distributed checkpoint save/load

输入：

- `ParallelConfig`
- `ModelRuntime` 生成的内部并行计划
- model
- batch

输出：

- wrapped model
- redistributed batch
- distributed save/load state

当前阶段例子：

- `TorchFSDP1ParallelRuntime`：
  - 在 world size > 1 时用 `torch.distributed.fsdp.FSDP(...)` 包模型
  - 单卡时退化成普通本地执行
  - 统一负责 backward / step / grad clip / checkpoint

---

## 3. `core` 层

`core` 不是第七个抽象，而是一个 facade / factory。

职责：

- 把 `ModelRuntime + TrainingObjective + ParallelRuntime` 组装成统一入口
- 给 Forge 自己和 Miles 都提供稳定的创建方式

例子：

```python
core = create_core(
    model_family="sd3",
    model_name_or_path=...,
    parallel_config=ParallelConfig(...),
)
```

然后调用方使用：

- `core.model_runtime`
- `core.objective`
- `core.parallel_runtime`

---

## 4. `Trainer` 的位置

`Trainer` 仍然存在，但它不是这层设计里的核心抽象。

它是平台编排层，职责是：

- 调 dataloader
- 调 `core`
- 驱动训练循环
- save / resume / logging

`Trainer` 不拥有：

- SD3 条件语义
- objective 语义
- FSDP1/FSDP2/SP 的模型细节

---

## 5. SD3 + FSDP1 执行流

初始化阶段：

1. `create_core(...)`
2. `model = core.model_runtime.build_model()`
3. `core.model_runtime.load_weights(model)`
4. `plan = core.model_runtime.make_parallel_plan(parallel_config)`
5. `model = core.parallel_runtime.parallelize_model(model, plan)`
6. 构造 optimizer / scheduler

每个 step：

1. `batch = core.model_runtime.canonicalize_batch(raw_batch)`
2. `batch = core.parallel_runtime.redistribute_batch(batch, plan)`
3. `objective_state = core.objective.prepare(batch, core.model_runtime, model)`
4. `model_inputs = core.model_runtime.prepare_forward_inputs(batch, objective_state)`
5. `outputs = model(**model_inputs)`
6. `loss, metrics, artifacts = core.objective.compute_loss(outputs, batch, objective_state)`
7. `core.parallel_runtime.backward(loss)`
8. `core.parallel_runtime.step(optimizer, scheduler)`
