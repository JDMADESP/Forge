from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class UlyssesParallelContext:
    degree: int
    rank: int
    group: dist.ProcessGroup


@dataclass(frozen=True)
class ShardMetadata:
    start: int
    end: int
    chunk_size: int


def build_ulysses_context(*, degree: int, group: dist.ProcessGroup | None = None) -> UlyssesParallelContext:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before enabling Ulysses sequence parallelism.")
    if degree <= 1:
        raise ValueError(f"Ulysses sequence parallelism requires degree > 1, got {degree}")

    resolved_group = group if group is not None else dist.group.WORLD
    world_size = dist.get_world_size(group=resolved_group)
    if world_size != degree:
        raise ValueError(f"Ulysses degree must match process-group world size, got degree={degree}, world_size={world_size}")
    return UlyssesParallelContext(
        degree=degree,
        rank=dist.get_rank(group=resolved_group),
        group=resolved_group,
    )


class AllToAllTensorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: dist.ProcessGroup):
        ctx.group = group
        input_tensor = tensor.contiguous()
        output = torch.empty(input_tensor.shape, device=input_tensor.device, dtype=input_tensor.dtype)
        dist.all_to_all_single(output, input_tensor, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_output = grad_output.contiguous()
        grad_input = torch.empty(grad_output.shape, device=grad_output.device, dtype=grad_output.dtype)
        dist.all_to_all_single(grad_input, grad_output, group=ctx.group)
        return grad_input, None


class GatherTensorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, dim: int, group: dist.ProcessGroup):
        ctx.dim = dim
        ctx.group = group
        ctx.world_size = dist.get_world_size(group=group)
        ctx.rank = dist.get_rank(group=group)

        input_tensor = tensor.contiguous()
        gathered = [torch.empty(input_tensor.shape, device=input_tensor.device, dtype=input_tensor.dtype) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, input_tensor, group=group)
        return torch.cat(gathered, dim=dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_chunks = torch.chunk(grad_output, ctx.world_size, dim=ctx.dim)
        return grad_chunks[ctx.rank].contiguous(), None, None


def _all_to_all_single(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    return AllToAllTensorFunction.apply(tensor, group)


def get_shard_metadata(length: int, context: UlyssesParallelContext) -> ShardMetadata:
    if length % context.degree != 0:
        raise ValueError(
            f"Expected dimension length divisible by ulysses degree, got length={length}, degree={context.degree}"
        )
    chunk_size = length // context.degree
    start = context.rank * chunk_size
    return ShardMetadata(start=start, end=start + chunk_size, chunk_size=chunk_size)


def shard_tensor(tensor: torch.Tensor, *, dim: int, context: UlyssesParallelContext) -> torch.Tensor:
    metadata = get_shard_metadata(tensor.size(dim), context)
    return tensor.narrow(dim, metadata.start, metadata.chunk_size).contiguous()


def gather_tensor(tensor: torch.Tensor, *, dim: int, context: UlyssesParallelContext) -> torch.Tensor:
    return GatherTensorFunction.apply(tensor, dim, context.group)


def ulysses_heads_to_sequence(x: torch.Tensor, *, context: UlyssesParallelContext) -> torch.Tensor:
    batch_size, seq_len_local, num_heads, head_dim = x.shape
    if num_heads % context.degree != 0:
        raise ValueError(
            f"Ulysses requires num_heads divisible by degree, got num_heads={num_heads}, degree={context.degree}"
        )
    num_heads_local = num_heads // context.degree
    x_temp = x.reshape(batch_size, seq_len_local, context.degree, num_heads_local, head_dim).transpose(0, 2).contiguous()
    exchanged = _all_to_all_single(x_temp, context.group)
    seq_len_global = seq_len_local * context.degree
    return exchanged.reshape(seq_len_global, batch_size, num_heads_local, head_dim).permute(1, 0, 2, 3).contiguous()


def ulysses_sequence_to_heads(x: torch.Tensor, *, context: UlyssesParallelContext) -> torch.Tensor:
    batch_size, seq_len_global, num_heads_local, head_dim = x.shape
    if seq_len_global % context.degree != 0:
        raise ValueError(
            f"Ulysses requires sequence length divisible by degree, got seq_len={seq_len_global}, degree={context.degree}"
        )
    seq_len_local = seq_len_global // context.degree
    x_temp = (
        x.reshape(batch_size, context.degree, seq_len_local, num_heads_local, head_dim)
        .permute(1, 3, 2, 0, 4)
        .reshape(context.degree, num_heads_local, seq_len_local, batch_size, head_dim)
    )
    exchanged = _all_to_all_single(x_temp, context.group)
    return (
        exchanged.reshape(num_heads_local * context.degree, seq_len_local, batch_size, head_dim)
        .transpose(0, 2)
        .contiguous()
        .reshape(batch_size, seq_len_local, num_heads_local * context.degree, head_dim)
    )
