try:
    from lm_eval.models.huggingface import HFLM
except Exception:  # pragma: no cover - depends on lm-eval version.
    HFLM = None


class LMClass:
    """Thin lm-eval wrapper for an already loaded Hugging Face model."""

    def __new__(cls, model, args):
        if HFLM is None:
            raise ImportError(
                "lm_eval.models.huggingface.HFLM is unavailable. "
                "Install a compatible lm-eval version or run without --tasks."
            )
        return HFLM(pretrained=model, tokenizer=getattr(args, "model", None), device=args.device)
