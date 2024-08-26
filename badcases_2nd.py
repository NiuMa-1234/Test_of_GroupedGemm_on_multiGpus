import unittest
import itertools

from absl.testing import parameterized
import sys
sys.path.append('/mnt/pfs-guan-ssai/nlu/lizr/zhanjiqing/operator_optimize/lizr/chenxingru/cases/grouped_gemm_base/')
from grouped_gemm import ops
import numpy as np
import torch

def allclose(x, y, pct=2.0):
    mask = torch.isclose(x, y, rtol=1e-5)
    pct_diff = (mask.numel() - mask.sum()) / mask.numel() * 100
    if pct_diff > pct:
        print(x[torch.logical_not(mask)], y[torch.logical_not(mask)])
        print("{:.2f}% of values not close.".format(pct_diff))
        return False
    return True


def add_transpose_flags(x):
    out = []
    for y in x:
        for f in [(False,), (True,)]:
            out.append(y + f)
    return out


_TEST_PROBLEMS = add_transpose_flags((
    (1, 128, 128, 128),
    (8, 128, 128, 128),
    (16, 128, 128, 128),
    (1, 128, 256, 512),
    (8, 128, 256, 512),
    (16, 128, 256, 512),
))


def randn(bs, x, y, device_id):
    out = (torch.rand(bs, x, y) - 0.5 * 2) / (y * x)
    device_id = 0
    device = torch.device("cuda:%d"%device_id if torch.cuda.is_available() else "cpu")
    return out.to(device).to(torch.bfloat16)



def gmm(a, b, batch_sizes, trans_b=False):
    batch_sizes = batch_sizes.numpy()

    out = []
    start = 0
    for i, size in enumerate(batch_sizes):
        rhs = b[i, :, :].t() if trans_b else b[i, :, :]
        out.append(a[start:start + size, :] @ rhs)
        start += size
    return torch.cat(out)


@parameterized.parameters(*_TEST_PROBLEMS)
class OpsTest(parameterized.TestCase):
  
    def testGroupedGemm_FixedSizes(self, z, m, k, n, trans_b):
        torch.manual_seed(0)
        device_id = 3
        a = randn(z, m, k, device_id).view(-1, k)
        b = randn(z, n, k, device_id) if trans_b else randn(z, k, n, device_id)
        batch_sizes = torch.tensor([m] * z)

        a.requires_grad_(True)
        b.requires_grad_(True)
        print("a.device: ",a.device)
        a_ref = a.detach().clone().requires_grad_(True)
        b_ref = b.detach().clone().requires_grad_(True)

        out = ops.gmm(a, b, batch_sizes, trans_b)
        expected_out = gmm(a_ref, b_ref, batch_sizes, trans_b)
        self.assertTrue(allclose(out, expected_out))

        # Check gradients.
        out.sum().backward()
        expected_out.sum().backward()
        self.assertTrue(allclose(a.grad, a_ref.grad))
        self.assertTrue(allclose(b.grad, b_ref.grad))

    def testGroupedGemm_VariableSizes(self, z, m, k, n, trans_b):
        torch.manual_seed(0)
        device_id = 3
        a = randn(z, m, k, device_id).view(-1, k)
        b = randn(z, n, k, device_id) if trans_b else randn(z, k, n, device_id)

        dist = torch.rand(z, )
        dist /= dist.sum()
        batch_sizes = (dist * m).to(torch.long)
        error = m * z - batch_sizes.sum()
        batch_sizes[-1] += error
        assert batch_sizes.sum() == (m * z)

        a.requires_grad_(True)
        b.requires_grad_(True)
        a_ref = a.detach().clone().requires_grad_(True)
        b_ref = b.detach().clone().requires_grad_(True)

        out = ops.gmm(a, b, batch_sizes, trans_b)
        expected_out = gmm(a_ref, b_ref, batch_sizes, trans_b)
        self.assertTrue(allclose(out, expected_out))

        # Check gradients.
        out.sum().backward()
        expected_out.sum().backward()
        self.assertTrue(allclose(a.grad, a_ref.grad))
        self.assertTrue(allclose(b.grad, b_ref.grad))



if __name__ == '__main__':
    def check_device(device_id):
        try:
            device = torch.device(f"cuda:{device_id}")
            tensor = torch.randn(3, 3, device=device)
            print(f"Tensor created successfully on GPU {device_id}")
        except RuntimeError as e:
            print(f"Error on GPU {device_id}: {e}")

    for i in range(torch.cuda.device_count()):
        check_device(i)

    if torch.cuda.is_available():
        print("GPU 3 memory summary:")
        print(torch.cuda.memory_summary(device='cuda:3'))
    unittest.main()
