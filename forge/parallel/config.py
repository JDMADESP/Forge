from __future__ import annotations

from dataclasses import dataclass, field
from math import isqrt


def derive_usp_mesh_shape(
    degree: int,
    *,
    ulysses_degree: int | None = None,
    ring_degree: int | None = None,
) -> tuple[int, int]:
    if degree < 1:
        raise ValueError(f"sequence_parallel.degree must be >= 1, got {degree}")

    if ulysses_degree is not None or ring_degree is not None:
        if ulysses_degree is None or ring_degree is None:
            raise ValueError("USP requires both ulysses_degree and ring_degree when either one is provided.")
        if ulysses_degree <= 1 or ring_degree <= 1:
            raise ValueError(
                f"USP requires both ulysses_degree and ring_degree to be > 1, got {(ulysses_degree, ring_degree)}"
            )
        if ulysses_degree * ring_degree != degree:
            raise ValueError(
                "sequence_parallel.degree must equal ulysses_degree * ring_degree for USP, "
                f"got {degree} != {ulysses_degree} * {ring_degree}"
            )
        return ring_degree, ulysses_degree

    for candidate_ulysses_degree in range(isqrt(degree), 1, -1):
        if degree % candidate_ulysses_degree == 0:
            candidate_ring_degree = degree // candidate_ulysses_degree
            if candidate_ring_degree > 1:
                return candidate_ring_degree, candidate_ulysses_degree

    raise ValueError(
        "USP requires a composite sequence-parallel degree so it can form both ring and ulysses groups, "
        f"got degree={degree}"
    )


def resolve_context_parallel_degrees(
    *,
    algorithm: str,
    degree: int,
    ulysses_anything: bool = False,
    ulysses_degree: int | None = None,
    ring_degree: int | None = None,
) -> tuple[int, int]:
    normalized_algorithm = algorithm.lower()
    if degree < 1:
        raise ValueError(f"sequence_parallel.degree must be >= 1, got {degree}")

    if normalized_algorithm == "ulysses":
        resolved_ulysses_degree = degree if ulysses_degree is None else ulysses_degree
        if resolved_ulysses_degree != degree:
            raise ValueError(
                "ulysses requires sequence_parallel.degree to match ulysses_degree, "
                f"got degree={degree}, ulysses_degree={resolved_ulysses_degree}"
            )
        if ring_degree not in (None, 1):
            raise ValueError("ulysses does not support ring_degree > 1.")
        return 1, resolved_ulysses_degree

    if normalized_algorithm == "ring":
        resolved_ring_degree = degree if ring_degree is None else ring_degree
        if resolved_ring_degree != degree:
            raise ValueError(
                "ring requires sequence_parallel.degree to match ring_degree, "
                f"got degree={degree}, ring_degree={resolved_ring_degree}"
            )
        if ulysses_degree not in (None, 1):
            raise ValueError("ring does not support ulysses_degree > 1.")
        if ulysses_anything:
            raise ValueError("ulysses_anything is incompatible with ring sequence parallelism.")
        return resolved_ring_degree, 1

    if normalized_algorithm == "ulysses_anything":
        resolved_ulysses_degree = degree if ulysses_degree is None else ulysses_degree
        if resolved_ulysses_degree != degree:
            raise ValueError(
                "ulysses_anything requires sequence_parallel.degree to match ulysses_degree, "
                f"got degree={degree}, ulysses_degree={resolved_ulysses_degree}"
            )
        if ring_degree not in (None, 1):
            raise ValueError("ulysses_anything does not support ring_degree > 1.")
        return 1, resolved_ulysses_degree

    if normalized_algorithm == "usp":
        if ulysses_anything:
            raise ValueError("ulysses_anything is incompatible with usp.")
        return derive_usp_mesh_shape(
            degree,
            ulysses_degree=ulysses_degree,
            ring_degree=ring_degree,
        )

    raise ValueError(f"Unsupported sequence-parallel algorithm: {algorithm}")


@dataclass(frozen=True)
class ParameterParallelConfig:
    mode: str = "fsdp1"
    degree: int = 1


@dataclass(frozen=True)
class SequenceParallelConfig:
    mode: str = "none"
    algorithm: str = "ulysses"
    degree: int = 1
    attention_backend: str = "native"
    convert_to_fp32: bool = True
    ulysses_anything: bool = False
    ulysses_degree: int | None = None
    ring_degree: int | None = None


@dataclass(frozen=True)
class ParallelConfig:
    backend: str = "torch"
    dp_mode: str = "fsdp1"
    mixed_precision: str = "bf16"
    parameter_parallel: ParameterParallelConfig | None = None
    sequence_parallel: SequenceParallelConfig = field(default_factory=SequenceParallelConfig)

    def __post_init__(self) -> None:
        if self.parameter_parallel is None:
            object.__setattr__(self, "parameter_parallel", ParameterParallelConfig(mode=self.dp_mode, degree=1))
