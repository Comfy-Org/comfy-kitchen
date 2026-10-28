import contextlib

_context = contextlib.nullcontext()


# Some Kitchen APIs do their own internal allocations. These persist outside
# the scope of the Kitchen call in question, so register them with Comfy for
# general management.
def set_allocation_context(context):
    global _context
    _context = contextlib.nullcontext() if context is None else context


def allocation_context():
    return _context
