"""splatprep — feature matching, SfM, masking and benchmarking for Splat Studio.

See docs/PROTOCOL.md §2 and §6 for the contract with the Mac app and the
Linux worker.  Public entry points:

* ``splatprep.runner.run_request(request, workdir, progress) -> stats``
* ``splatprep.models.available_models() -> list[str]``
* CLI: ``python -m splatprep run --request request.json --progress-jsonl -``

Heavy dependencies (torch, lightglue, kornia, sam2, mast3r, vggt, pycolmap)
are imported lazily; a missing dependency only marks the corresponding model
unavailable.
"""

__version__ = "0.1.0"

PROTOCOL_VERSION = 1
