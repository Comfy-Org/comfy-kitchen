"""One module per adapter family; each yields Cases for a named sweep."""

from . import dora, ia3, loha, lokr, lora, oft

FAMILIES = {
    "lora": lora,
    "loha": loha,
    "lokr": lokr,
    "oft": oft,
    "dora": dora,
    "ia3": ia3,
}
