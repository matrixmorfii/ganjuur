# =============================================================================
# Ganjuur — shared Gradio upstream compatibility patches
# =============================================================================
# Applies monkey-patches to gradio_client.utils and gradio.networking BEFORE
# any other code imports them.  Every entrypoint that uses Gradio MUST import
# this module as its very first import, before import gradio / import gradio as gr.
#
# Running patches via a single shared module means the five near‑identical
# copies that used to live at the top of app.py, gpt.py, longcat.py,
# appvmeta.py, app06_24.py, app06_24_3.py, and ganjuur_v2.py are now
# replaced by:
#
#     import grad_patch          # must be the first import
#
# Behavior is identical in every way.
# =============================================================================

import gradio_client.utils
import gradio.networking

# ── Patch 1: Pydantic-v2 boolean JSON schema ───────────────────────────────
# Without this patch, Gradio's internal schema parser raises:
#   TypeError: argument of type 'bool' is not iterable
# when encountering Pydantic-v2 boolean JSON schemas.

_orig_json_schema_to_python_type = gradio_client.utils._json_schema_to_python_type


def _patched_json_schema_to_python_type(schema, defs=None):
    """Return 'any' for boolean schemas, delegate to original otherwise."""
    if isinstance(schema, bool):
        return "any"
    return _orig_json_schema_to_python_type(schema, defs)


_orig_get_type = gradio_client.utils.get_type


def _patched_get_type(schema):
    """Return 'bool' for boolean schemas, delegate to original otherwise."""
    if isinstance(schema, bool):
        return "bool"
    return _orig_get_type(schema)


gradio_client.utils._json_schema_to_python_type = _patched_json_schema_to_python_type
gradio_client.utils.get_type = _patched_get_type

# ── Patch 2: LAN/Nginx localhost-reachability check ────────────────────────
# Gradio's networking module probes localhost on startup.  When the app is
# served behind Nginx or on a headless LAN host this probe can fail with:
#   ValueError: When localhost is not accessible, a shareable link cannot be
#   created. Please share locally or set share=True.
# Disabling this check allows the app to start behind any proxy.

gradio.networking.url_ok = lambda url: True
