"""Native Qwen tool-schema defaults for Ciru's serving plugin."""
from vllm import envs
from vllm.tool_parsers.tool_strict_level import ToolStrictLevel
from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag


def native_tool_tag(parser, request, *, reasoning, fallback, strict_level=ToolStrictLevel.AUTO):
    """Preserve caller strictness for auto calls without rewriting tools.

    An explicit strict=False remains an opt-out. Explicit output formats and
    vLLM's global enforcement switch retain their existing meaning. Required
    and named choices continue through the upstream implementation.
    """
    if request.tool_choice not in (None, "auto"):
        return fallback(request, reasoning=reasoning, strict_level=strict_level)
    if not request.tools or not envs.VLLM_ENFORCE_STRICT_TOOL_CALLING:
        return None
    if getattr(request, "structured_outputs", None) is not None:
        return None
    response_format = getattr(request, "response_format", None)
    if response_format is not None and response_format.type != "text":
        return None
    text = getattr(request, "text", None)
    if text is not None and text.format is not None and text.format.type != "text":
        return None

    # Match upstream auto semantics. An unspecified strict flag is not an
    # opt-in to schema-constrained generation. Keep prompt-visible tools and
    # explicit strict=True/False exactly as provided by the caller.
    compiler_tools = list(request.tools)
    return get_model_structural_tag(
        model=parser.structural_tag_model,
        tools=compiler_tools,
        tool_choice=request.tool_choice or "auto",
        reasoning=reasoning,
        strict_level=strict_level,
    )
