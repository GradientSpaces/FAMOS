"""Draw the object masks for scripts/reconstruct.py with SAM2, in the browser.

    python scripts/segment.py path/to/object

Then open http://localhost:8000 (on a remote machine, forward the port first with
`ssh -L 8000:localhost:8000 <host>`). The page walks through every photo in `<object>/images/`:
drag a box around the object, fix the mask with clicks (left click adds missing parts, right
click or shift-click removes background), and save it. Masks are saved to `<object>/masks/`,
under the photo's file name stem.
"""

import argparse
import functools
import http.server
import io
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image, ImageOps
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy.ndimage import binary_dilation

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PAGE = Path(__file__).with_name("segment.html")
SAM2_MODEL = "facebook/sam2.1-hiera-large"
# Photos are sent to the browser at this size at most. Prompts come back as fractions of the
# photo size, and masks are predicted and saved at the full size.
DISPLAY_SIZE = 1280
OUTLINE_COLOR = (106, 79, 182)
OUTLINE_WIDTH = 4  # pixels at display size
BACKGROUND_DIMMING = 150  # alpha of the black over the background


class Segmenter:
    def __init__(self, object_dir: Path):
        self.object_dir = object_dir
        self.photos = sorted(
            p for p in (object_dir / "images").iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        self.mask_dir = object_dir / "masks"
        self.predictor = SAM2ImagePredictor.from_pretrained(SAM2_MODEL)
        # The photo whose SAM2 features are cached, and its size.
        self.current, self.size = None, None
        # Unsaved masks per photo: a boolean mask, or None if the mask was cleared.
        self.drafts = {}

    def mask_path(self, i: int) -> Path:
        return self.mask_dir / f"{self.photos[i].stem}.png"

    def photo(self, i: int) -> Image.Image:
        # Phone photos are often stored sideways, with an EXIF tag that rotates them for display.
        return ImageOps.exif_transpose(Image.open(self.photos[i])).convert("RGB")

    @functools.lru_cache(maxsize=None)
    def display_jpeg(self, i: int) -> bytes:
        image = self.photo(i)
        image.thumbnail((DISPLAY_SIZE, DISPLAY_SIZE), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=90)
        return buffer.getvalue()

    def overlay_png(self, i: int, mask: np.ndarray) -> bytes:
        """The mask at display size: the background dimmed and the object outlined."""
        size = Image.open(io.BytesIO(self.display_jpeg(i))).size
        mask = np.asarray(Image.fromarray(mask).resize(size, Image.Resampling.NEAREST))
        overlay = np.zeros((*mask.shape, 4), np.uint8)
        overlay[~mask, 3] = BACKGROUND_DIMMING
        overlay[binary_dilation(mask, iterations=OUTLINE_WIDTH) & ~mask] = (*OUTLINE_COLOR, 255)
        buffer = io.BytesIO()
        Image.fromarray(overlay).save(buffer, "PNG")
        return buffer.getvalue()

    def binary_png(self, i: int, mask: np.ndarray) -> bytes:
        """The mask as saved for reconstruct.py (white = object), at display size."""
        size = Image.open(io.BytesIO(self.display_jpeg(i))).size
        buffer = io.BytesIO()
        Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST).save(buffer, "PNG")
        return buffer.getvalue()

    def saved_mask(self, i: int):
        path = self.mask_path(i)
        return np.asarray(Image.open(path).convert("L")) > 0 if path.exists() else None

    def current_mask(self, i: int):
        """The mask that saving photo `i` would write: its draft if it has one, else the saved one."""
        return self.drafts[i] if i in self.drafts else self.saved_mask(i)

    def save(self, i: int):
        """Write the draft of photo `i` to disk (or remove the file if the mask was cleared)."""
        if i not in self.drafts:
            return
        mask, path = self.drafts.pop(i), self.mask_path(i)
        if mask is None:
            path.unlink(missing_ok=True)
        else:
            self.mask_dir.mkdir(exist_ok=True)
            Image.fromarray(mask.astype(np.uint8) * 255).save(path)

    @torch.inference_mode()
    def segment(self, i: int, box, points, labels):
        """Mask photo `i` from a box and clicks given as fractions of the photo size, as a draft
        that `save` writes. Without prompts, the draft is an empty (cleared) mask."""
        if box is None and not points:
            self.drafts[i] = None
            return None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.current != i:
                image = self.photo(i)
                self.predictor.set_image(np.asarray(image))
                self.current, self.size = i, np.array(image.size)
            masks, scores, _ = self.predictor.predict(
                point_coords=np.array(points) * self.size if points else None,
                point_labels=np.array(labels) if points else None,
                box=np.array(box) * np.tile(self.size, 2) if box is not None else None,
                # A single click is ambiguous: let SAM2 propose several masks and keep the best.
                multimask_output=box is None and len(points) == 1,
            )
        mask = masks[np.argmax(scores)] > 0
        self.drafts[i] = mask
        return mask


class Handler(http.server.BaseHTTPRequestHandler):
    """Serves segment.html, the photos and masks, and runs SAM2 on posted prompts."""

    def __init__(self, segmenter: Segmenter, *args, **kwargs):
        self.segmenter = segmenter
        super().__init__(*args, **kwargs)

    def do_GET(self):
        s = self.segmenter
        url = urlparse(self.path)
        kind, _, index = url.path.strip("/").partition("/")
        if kind in ("", "index.html"):
            return self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
        if kind == "info.json":
            # A (re)loaded page starts from the saved masks.
            s.drafts.clear()
            info = {
                "object": str(s.object_dir),
                "mask_dir": str(s.mask_dir.resolve()),
                "photos": [
                    {"name": p.name, "mask": s.mask_path(i).name, "saved": s.mask_path(i).exists()}
                    for i, p in enumerate(s.photos)
                ],
            }
            return self._send(json.dumps(info).encode(), "application/json")
        if not index.isdigit() or int(index) >= len(s.photos):
            return self.send_error(404)
        if kind == "photo":
            return self._send(s.display_jpeg(int(index)), "image/jpeg")
        if kind in ("mask", "binary"):
            # `?saved=1`: the mask on disk; otherwise the unsaved draft if there is one.
            i = int(index)
            mask = s.saved_mask(i) if "saved=1" in url.query else s.current_mask(i)
            if mask is None:
                return self.send_error(404)
            png = s.overlay_png if kind == "mask" else s.binary_png
            return self._send(png(i, mask), "image/png")
        self.send_error(404)

    def do_POST(self):
        s = self.segmenter
        kind, _, index = urlparse(self.path).path.strip("/").partition("/")
        if kind not in ("segment", "save", "discard") or not index.isdigit() or int(index) >= len(s.photos):
            return self.send_error(404)
        if kind == "save":
            s.save(int(index))
        elif kind == "discard":
            s.drafts.pop(int(index), None)
        if kind != "segment":
            self.send_response(204)
            return self.end_headers()
        prompts = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        mask = s.segment(int(index), prompts["box"], prompts["points"], prompts["labels"])
        if mask is None:
            self.send_response(204)
            return self.end_headers()
        self._send(s.overlay_png(int(index), mask), "image/png")

    def _send(self, body, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main(object_dir: Path, port: int):
    if not (object_dir / "images").is_dir():
        raise SystemExit(f"{object_dir} has no images/ folder.")
    segmenter = Segmenter(object_dir)
    if not segmenter.photos:
        raise SystemExit(f"No photos in {object_dir / 'images'}.")
    # One request at a time, since they share the GPU and SAM2's cached photo features.
    server = http.server.HTTPServer(("127.0.0.1", port), functools.partial(Handler, segmenter))
    print(
        f"Segmenting {len(segmenter.photos)} photos of {object_dir} at http://localhost:{port} "
        f"(Ctrl+C to stop). Saved masks go to {segmenter.mask_dir.resolve()}/."
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("object_dir", type=Path, help="Folder with images/.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(args.object_dir, args.port)
