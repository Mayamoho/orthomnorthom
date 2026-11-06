"""NB Assist package and optional language renderer."""

try:
    from . import bn_render as _bn_render
    _bn_render.install()
except Exception as _exc:
    # The core FAQ assistant remains usable in lexical smoke-test mode.
    print("[bn_render] not installed:", type(_exc).__name__, _exc)
