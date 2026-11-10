"""Named-handler registry. Only functions registered here are reachable over HTTP."""

HANDLERS = {}


class BridgeArgError(ValueError):
    """Invalid argument that cannot be clamped (unknown name, wrong type). Returned as ok=false."""


def handler(name):
    def deco(fn):
        if name in HANDLERS:
            raise RuntimeError(f"duplicate handler {name}")
        HANDLERS[name] = fn
        return fn
    return deco
