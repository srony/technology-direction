"""
ST4TE shared utilities.

Light-weight convenience imports.  Heavy modules (embeddings)
are imported explicitly by the sub-projects that need them to avoid
pulling in torch / matplotlib at package-init time.
"""

from shared.utils import ensure_dir, safe_alias, parse_int_or_none, extract_tasks
from shared.cleaning import add_identifier, compile_any, flag_noise_pages
from shared.llm import ask_openrouter, openrouter_chat, read_openrouter_key

__all__ = [
    "ensure_dir",
    "safe_alias",
    "parse_int_or_none",
    "extract_tasks",
    "add_identifier",
    "compile_any",
    "flag_noise_pages",
    "ask_openrouter",
    "openrouter_chat",
    "read_openrouter_key",
]
