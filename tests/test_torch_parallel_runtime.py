from __future__ import annotations

import unittest
from unittest import mock

import torch

from forge.parallel.torch_runtime import TorchParallelRuntime


class TorchParallelRuntimeGradientSyncTest(unittest.TestCase):
    def test_step_all_reduces_sequence_parallel_gradients(self) -> None:
        runtime = TorchParallelRuntime()
        runtime.sequence_parallel_group = object()
        runtime.use_fsdp = False

        param = torch.nn.Parameter(torch.tensor([1.0]))
        param.grad = torch.tensor([2.0])
        optimizer = torch.optim.SGD([param], lr=0.1)

        def fake_all_reduce(tensor, *, op, group):
            del op, group
            tensor.mul_(2)

        with (
            mock.patch.object(runtime, "is_distributed", return_value=True),
            mock.patch("forge.parallel.torch_runtime.dist.all_reduce", side_effect=fake_all_reduce) as all_reduce,
        ):
            runtime.step(optimizer)

        all_reduce.assert_called_once()
        grad_arg = all_reduce.call_args.args[0]
        self.assertTrue(torch.equal(grad_arg, torch.tensor([4.0])))
        self.assertEqual(all_reduce.call_args.kwargs["op"], torch.distributed.ReduceOp.SUM)
        self.assertIs(all_reduce.call_args.kwargs["group"], runtime.sequence_parallel_group)
        self.assertTrue(torch.equal(param, torch.tensor([0.6])))

    def test_step_skips_gradient_sync_without_sequence_parallel(self) -> None:
        runtime = TorchParallelRuntime()
        param = torch.nn.Parameter(torch.tensor([1.0]))
        param.grad = torch.tensor([2.0])
        optimizer = torch.optim.SGD([param], lr=0.1)

        with mock.patch.object(runtime, "is_distributed", return_value=True), mock.patch(
            "forge.parallel.torch_runtime.dist.all_reduce"
        ) as all_reduce:
            runtime.step(optimizer)

        all_reduce.assert_not_called()


if __name__ == "__main__":
    unittest.main()
