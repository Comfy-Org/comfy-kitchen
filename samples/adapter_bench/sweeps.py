"""The sweeps: each varies one quantity around a fixed real layer.

Fixed layer: a Flux attention projection, 3072 x 3072, rank 32, 4096 tokens
(a 1024x1024 image's latent). A linear layer sees batch x context only as
their product, so the context sweep at batch 1 is the token sweep; the batch
sweep keeps the (batch, context, features) shape to exercise the lead dims.
"""

FIXED_DIM = 3072
FIXED_RANK = 32
FIXED_TOKENS = 4096
FIXED_FACTOR = -1
FIXED_BLOCK = 32

TOKENS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
BATCH = [1, 2, 4, 8]
# Square layers from SDXL (640, 1280) through Flux (3072) to Wan (5120).
DIMS = [640, 1280, 2048, 3072, 4096, 5120]
RANKS = [4, 8, 16, 32, 64, 128, 256]
FACTORS = [-1, 4, 8, 16, 32]
BLOCKS = [4, 8, 16, 32, 64, 128]
# Conv OFT on an SDXL 640-channel block: spatial side of the feature map.
SPATIAL = [16, 32, 64, 128]
CONV_CHANNELS = 640


def thin(values: list, quick: bool) -> list:
    """Every other point (ends kept) for a smoke run."""
    if not quick or len(values) <= 3:
        return values
    picked = values[::2]
    return picked if picked[-1] == values[-1] else [*picked, values[-1]]


def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """LyCORIS's split of a dimension into (m, n), m <= n, m the nearest factor."""
    if factor > 0 and dimension % factor == 0 and dimension >= factor * factor:
        m, n = factor, dimension // factor
        return (n, m) if m > n else (m, n)
    if factor < 0:
        factor = dimension
    m, n = 1, dimension
    while m < n:
        new_m = m + 1
        while dimension % new_m:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > m + n or new_m > factor:
            break
        m, n = new_m, new_n
    return (n, m) if m > n else (m, n)
