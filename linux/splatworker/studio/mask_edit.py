"""The mask editor's model (GTK-free; the window is ``gui/gtk/studio/mask_editor.py``), as the Mac's
``MaskEditorModel``: browse a dataset's images with their masks, click an object to select it with SAM 2
(positive / negative points), paint out / paint back with a brush, track the clicked objects through the
sequence, mask fixed parts of the frame, and save.

Edits are a mask layer of their own, so "Prepare again" (which recomputes every model layer) keeps them::

    dataset/splatprep-masks/manual/masks/<stem>.png     255 = keep, 0 = removed here (ANDed like any layer)
    dataset/splatprep-masks/manual/restore/<stem>.png   255 = keep whatever the other layers say ("paint back")

Saving re-stacks only the edited images into ``masks/`` (:func:`importers.stack_masks`) and updates their
``mask`` entries in ``dataset.json``. Only the current image is held at full resolution; edits of other images
are kept bit-packed until saved, and thumbnails are small.

SAM 2 runs on the local worker: a click is a ``maskpreview`` processing job (one image, the clicked points →
``preview.png`` with 255 on the object), tracking is a ``masks`` job with ``maskModel: sam2_track`` and
``maskPrompts`` (every queued object, tracked forward and backward through ``images/``).
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import importers

UNDO_LIMIT = 40


def _read_gray(path: Path, size: Tuple[int, int]) -> Optional[np.ndarray]:
    from PIL import Image

    if not path.is_file():
        return None
    try:
        with Image.open(path) as im:
            g = im.convert("L")
            if g.size != size:
                g = g.resize(size, Image.NEAREST)
            return np.asarray(g)
    except Exception:  # noqa: BLE001 — an unreadable mask counts as none
        return None


@dataclass
class Frame:
    image: str                   # relative to the dataset ("images/000012.jpg")
    width: int
    height: int
    camera: Optional[int] = None  # index into dataset.json cameras

    @property
    def stem(self) -> str:
        return Path(self.image).stem

    @property
    def name(self) -> str:
        return Path(self.image).name


@dataclass
class _Edit:
    shape: Tuple[int, int]
    remove: np.ndarray = None     # packed bool: removed here
    restore: np.ndarray = None    # packed bool: painted back

    @classmethod
    def pack(cls, remove: np.ndarray, restore: np.ndarray) -> "_Edit":
        return cls(remove.shape, np.packbits(remove), np.packbits(restore))

    def unpack(self) -> Tuple[np.ndarray, np.ndarray]:
        n = self.shape[0] * self.shape[1]
        return (np.unpackbits(self.remove)[:n].reshape(self.shape).astype(bool),
                np.unpackbits(self.restore)[:n].reshape(self.shape).astype(bool))


@dataclass
class TrackObject:
    image: str
    points: List[List[float]]     # [x, y, label] in image pixels
    object_id: int

    def prompt(self) -> Dict:
        return {"image": self.image, "points": self.points, "objectId": self.object_id}


class MaskEditModel:
    def __init__(self, dataset_dir: Path):
        self.ds = Path(dataset_dir)
        self.frames = self._frames()
        self.index = 0
        self.dirty: set = set()
        self.points: List[List[float]] = []          # SAM 2 clicks on the current image
        self.tracked: List[TrackObject] = []
        self._edits: Dict[int, _Edit] = {}
        self._undo: List[Tuple[int, _Edit]] = []
        self._cur: Optional[Tuple[int, np.ndarray, np.ndarray]] = None   # (index, remove, restore) unpacked
        self._base: Dict[int, np.ndarray] = {}       # other layers' keep mask of the current image (cache of 1)
        self._before_clicks: Optional[_Edit] = None

    # ------------------------------------------------------------- frames
    def _frames(self) -> List[Frame]:
        from PIL import Image

        doc_path = self.ds / "dataset.json"
        out: List[Frame] = []
        if doc_path.is_file():
            doc = json.loads(doc_path.read_text(encoding="utf-8"))
            for i, c in enumerate(doc.get("cameras", [])):
                out.append(Frame(c["image"], int(c["width"]), int(c["height"]), i))
            if out:
                return out
        for p in importers.list_images(self.ds / "images") if (self.ds / "images").is_dir() else []:
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception:  # noqa: BLE001
                continue
            out.append(Frame(f"images/{p.name}", w, h))
        return out

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def current(self) -> Frame:
        return self.frames[self.index]

    def select(self, i: int) -> None:
        i = max(0, min(len(self.frames) - 1, i))
        if i != self.index:
            self._stash()
            self.points.clear()
            self._before_clicks = None
            self.index = i

    def image(self, i: int, max_side: int = 0) -> np.ndarray:
        """RGB uint8 of frame ``i``, downscaled so the longer side is at most ``max_side`` (0 = full)."""
        from PIL import Image, ImageOps

        with Image.open(self.ds / self.frames[i].image) as im:
            if max_side:
                im.draft("RGB", (max_side, max_side))
            im = ImageOps.exif_transpose(im).convert("RGB")
            if max_side and max(im.size) > max_side:
                im.thumbnail((max_side, max_side), Image.BILINEAR)
            return np.asarray(im)

    def thumbnail(self, i: int, side: int = 96) -> np.ndarray:
        return self.image(i, side)

    def has_mask(self, i: int) -> bool:
        return i in self.dirty or (self.ds / "masks" / f"{self.frames[i].stem}.png").is_file()

    # ------------------------------------------------------------- masks
    def _layers(self) -> List[Path]:
        return [p for p in importers.mask_layers(self.ds) if p.parent.name != "manual"]

    def base_keep(self, i: int) -> np.ndarray:
        """Keep mask of every layer except the editor's (AND), at full resolution."""
        if i in self._base:
            return self._base[i]
        f = self.frames[i]
        keep = np.ones((f.height, f.width), bool)
        layers = self._layers()
        if layers:
            for layer in layers:
                g = _read_gray(layer / f"{f.stem}.png", (f.width, f.height))
                if g is not None:
                    keep &= g >= 128
        else:                       # an imported dataset without layers: its masks/ is the base
            g = _read_gray(self.ds / "masks" / f"{f.stem}.png", (f.width, f.height))
            if g is not None:
                keep &= g >= 128
        self._base = {i: keep}
        return keep

    def _load_edit(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._cur is not None and self._cur[0] == i:
            return self._cur[1], self._cur[2]
        if i in self._edits:
            rem, res = self._edits[i].unpack()
        else:
            f = self.frames[i]
            size = (f.width, f.height)
            m = _read_gray(self.ds / importers.MANUAL_LAYER / "masks" / f"{f.stem}.png", size)
            r = _read_gray(self.ds / importers.MANUAL_LAYER / "restore" / f"{f.stem}.png", size)
            rem = (m < 128) if m is not None else np.zeros((f.height, f.width), bool)
            res = (r >= 128) if r is not None else np.zeros((f.height, f.width), bool)
        self._stash()
        self._cur = (i, rem, res)
        return rem, res

    def _stash(self) -> None:
        if self._cur is not None:
            i, rem, res = self._cur
            if i in self.dirty:
                self._edits[i] = _Edit.pack(rem, res)
            self._cur = None

    def keep(self, i: Optional[int] = None) -> np.ndarray:
        """The effective keep mask (base AND manual, OR painted back) — what training will ignore is ~keep."""
        i = self.index if i is None else i
        rem, res = self._load_edit(i)
        return (self.base_keep(i) & ~rem) | res

    def _push_undo(self) -> None:
        rem, res = self._load_edit(self.index)
        self._undo.append((self.index, _Edit.pack(rem, res)))
        del self._undo[:-UNDO_LIMIT]

    def paint(self, x: float, y: float, radius: float, remove: bool, begin: bool = True) -> None:
        """A brush dab at image pixel (x, y): ``remove`` paints out, else paints back."""
        if begin:
            self._push_undo()
        rem, res = self._load_edit(self.index)
        h, w = rem.shape
        r = max(1, int(round(radius)))
        x0, x1 = max(0, int(x) - r), min(w, int(x) + r + 1)
        y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 1)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disk = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
        rem[y0:y1, x0:x1][disk] = remove
        res[y0:y1, x0:x1][disk] = not remove
        self.dirty.add(self.index)

    def undo(self) -> bool:
        if not self._undo:
            return False
        i, e = self._undo.pop()
        self._stash()
        self.index = i
        rem, res = e.unpack()
        self._cur = (i, rem, res)
        self.dirty.add(i)
        return True

    def clear_frame(self) -> None:
        """Forget this image's manual edits (other layers stay)."""
        self._push_undo()
        rem, res = self._load_edit(self.index)
        rem[:] = False
        res[:] = False
        self.dirty.add(self.index)

    # ------------------------------------------------------------- SAM 2
    def add_point(self, x: float, y: float, positive: bool) -> None:
        if not self.points:
            rem, res = self._load_edit(self.index)
            self._before_clicks = _Edit.pack(rem, res)
            self._push_undo()
        self.points.append([float(x), float(y), 1 if positive else 0])

    def preview_request(self, out: str) -> Dict:
        f = self.current
        return {"task": "maskpreview", "imagesDir": "images", "outputDir": out, "previewImage": f.name,
                "previewObjectMask": True, "maskPrompts": [{"image": f.name, "points": list(self.points)}]}

    def apply_object(self, obj_png: Path) -> float:
        """SAM 2 returned the object (255 on it): remove it on top of the mask from before this click session.
        Returns the object's area fraction."""
        f = self.current
        obj = _read_gray(Path(obj_png), (f.width, f.height))
        if obj is None:
            raise ValueError("SAM 2 returned no mask")
        obj = obj >= 128
        if self._before_clicks is not None:
            rem, res = self._before_clicks.unpack()
        else:
            rem, res = (a.copy() for a in self._load_edit(self.index))
        rem = rem | obj
        res = res & ~obj
        self._stash()
        self._cur = (self.index, rem, res)
        self.dirty.add(self.index)
        return float(obj.mean())

    def reset_clicks(self) -> None:
        if self._before_clicks is not None:
            rem, res = self._before_clicks.unpack()
            self._stash()
            self._cur = (self.index, rem, res)
        self.points.clear()
        self._before_clicks = None

    def next_object(self) -> None:
        """Queue the clicked object for tracking and start a new one."""
        if self.points:
            self.tracked.append(TrackObject(self.current.name, list(self.points), len(self.tracked) + 1))
        self.points.clear()
        self._before_clicks = None

    def track_request(self, out: str) -> Optional[Dict]:
        self.next_object()
        if not self.tracked:
            return None
        return {"task": "masks", "imagesDir": "images", "outputDir": out, "maskModel": "sam2_track",
                "maskPrompts": [t.prompt() for t in self.tracked], "replaceMasks": True}

    def apply_tracked(self, masks_dir: Path, progress: Callable[[float, str], None] = lambda f, m="": None) -> int:
        """AND the tracker's keep masks into the manual layer of every image (one image at a time)."""
        n = 0
        self._stash()
        for i, f in enumerate(self.frames):
            g = _read_gray(Path(masks_dir) / f"{f.stem}.png", (f.width, f.height))
            if g is None or (g >= 128).all():
                continue
            rem, res = self._load_edit(i)
            obj = g < 128
            rem |= obj
            res &= ~obj
            self.dirty.add(i)
            self._stash()
            n += 1
            if i % 20 == 0:
                progress(i / len(self.frames), f"{i + 1}/{len(self.frames)}")
        self._cur = None
        self.tracked.clear()
        return n

    # ------------------------------------------------------------- save
    def save(self) -> Dict[str, int]:
        """Write the manual layer of the edited images, re-stack their ``masks/`` and update dataset.json."""
        from PIL import Image

        self._stash()
        layer = self.ds / importers.MANUAL_LAYER
        stems = []
        for i in sorted(self.dirty):
            f = self.frames[i]
            e = self._edits.get(i)
            if e is None:
                continue
            rem, res = e.unpack()
            for sub, arr, val in (("masks", rem, 0), ("restore", res, 255)):
                p = layer / sub / f"{f.stem}.png"
                if arr.any():
                    p.parent.mkdir(parents=True, exist_ok=True)
                    img = np.where(arr, val, 255 - val).astype(np.uint8)
                    tmp = p.with_name(f".{p.name}.tmp.png")
                    Image.fromarray(img).save(tmp)
                    tmp.replace(p)
                else:
                    p.unlink(missing_ok=True)
            stems.append(f.stem)
        layers = importers.mask_layers(self.ds)
        if not any(p.parent.name != "manual" for p in layers) and (self.ds / "masks").is_dir() \
                and any((self.ds / "masks").glob("*.png")):
            # imported dataset: its own masks/ were the base — keep them as a layer so re-stacking can't lose them
            base = self.ds / "splatprep-masks" / "imported" / "masks"
            if not base.is_dir():
                shutil.copytree(self.ds / "masks", base)
            layers = importers.mask_layers(self.ds)
        masks = importers.stack_masks(self.ds, layers, restore=importers.manual_restore_dirs(self.ds),
                                      only=stems) if stems else {}
        self._relink(stems, masks)
        self.dirty.clear()
        self._edits.clear()
        self._base.clear()
        return {"saved": len(stems), "masked": len(masks)}

    def restack_all(self) -> Dict[str, int]:
        """Re-stack every image (after "mask fixed parts of the frame")."""
        self._stash()
        stems = [f.stem for f in self.frames]
        masks = importers.stack_masks(self.ds, importers.mask_layers(self.ds),
                                      restore=importers.manual_restore_dirs(self.ds), only=stems)
        self._relink(stems, masks)
        self._base.clear()
        return {"masked": len(masks)}

    def _relink(self, stems: List[str], masks: Dict[str, str]) -> None:
        path = self.ds / "dataset.json"
        if not path.is_file():
            return
        doc = json.loads(path.read_text(encoding="utf-8"))
        want = set(stems)
        for c in doc.get("cameras", []):
            s = Path(c["image"]).stem
            if s not in want:
                continue
            if s in masks:
                c["mask"] = masks[s]
            else:
                c.pop("mask", None)
        tmp = path.with_name("dataset.json.tmp")
        tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        tmp.replace(path)


# ------------------------------------------------------------------------------- worker jobs

def run_job(client, dataset_dir: Path, request: Dict, name: str, dest: Path,
            on_status: Callable[[Dict], None] = lambda s: None, cancelled: Callable[[], bool] = lambda: False,
            poll: float = 0.5) -> Path:
    """Run one processing request on the dataset folder; its output is linked into ``dest`` (inside the
    dataset, which the server only exports into when registered). Returns ``dest``."""
    reg = client.local_dataset(dataset_dir)["datasetId"]
    try:
        job = client.process(reg, request, name, 1)
        while True:
            if cancelled():
                client.cancel(job)
            s = client.job(job)
            on_status(s)
            st = s.get("state")
            if st == "completed":
                break
            if st in ("failed", "cancelled"):
                raise RuntimeError(s.get("message") or st)
            time.sleep(poll)
        out = s.get("outputDatasetId")
        if not out:
            raise RuntimeError("the worker kept no output")
        shutil.rmtree(dest, ignore_errors=True)
        client.export_dataset(out, dest)
        client.delete_dataset(out)
        return dest
    finally:
        client.delete_dataset(reg)


def scratch_dir(dataset_dir: Path) -> Path:
    d = Path(dataset_dir) / ".mask-editor" / uuid.uuid4().hex[:8]
    return d
