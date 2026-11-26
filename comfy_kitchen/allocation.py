import contextlib

_context = None


# Some Kitchen APIs do their own internal allocations. These persist outside
# the scope of the Kitchen call in question, so register them with Comfy for
# general management.
def set_allocation_context(context):
    global _context
    _context = context


def allocation_context():
    return contextlib.nullcontext() if _context is None else _context


def has_allocation_context():
    return _context is not None
