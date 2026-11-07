"""Exception types shared across splatprep."""


class SplatPrepError(RuntimeError):
    """Generic processing failure (reported as ``{"error": str(exc)}``)."""


class ModelUnavailable(SplatPrepError):
    """Requested model cannot run here (missing package, binary or weights)."""

    def __init__(self, model_id: str, reason: str = ""):
        self.model_id = model_id
        self.reason = reason
        msg = f"model unavailable: {model_id}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


class MapperUnavailable(SplatPrepError):
    """No SfM mapper (pycolmap / colmap / glomap) is available."""


class RequestError(SplatPrepError):
    """Malformed ProcessingRequest."""
