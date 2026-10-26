import unittest

import torch

from comfy_kitchen.backends.eager.sol_attn import coarse_output


class CoarseOutputTests(unittest.TestCase):
    def test_query_chunks_match_full(self):
        for blocks, query_chunk in ((131, 64), (17, 7)):
            with self.subTest(blocks=blocks, query_chunk=query_chunk):
                generator = torch.Generator().manual_seed(271)
                shape = (3, blocks, 32)
                qm, km, vm = (
                    torch.randn(shape, generator=generator) for _ in range(3)
                )
                scale = shape[-1] ** -0.5

                expected = coarse_output(qm, km, vm, scale)
                actual = coarse_output(
                    qm, km, vm, scale, query_chunk=query_chunk,
                )

                torch.testing.assert_close(
                    actual, expected, rtol=2e-4, atol=5e-5,
                )


if __name__ == "__main__":
    unittest.main()
