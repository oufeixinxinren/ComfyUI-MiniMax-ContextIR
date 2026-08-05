from .h3_context_ir import MiniMaxH3ContextIR, MiniMaxH3ContextIRExtension


async def comfy_entrypoint():
    return MiniMaxH3ContextIRExtension()


__all__ = ["MiniMaxH3ContextIR", "MiniMaxH3ContextIRExtension", "comfy_entrypoint"]
