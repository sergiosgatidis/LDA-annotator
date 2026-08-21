# LDA Annotator

A browser-based annotation tool for chest radiographs, optimised for iPad + Apple Pencil. Annotates lines, tubes, and devices as ordered stroke paths — capturing not just pixel masks but the full temporal trajectory of each structure.

## Features

- **27 pre-defined clinical structures** (ETT, CVC, PICC, ECMO, pacemaker wires, drains, etc.)
- **Ordered stroke recording** — each pen gesture is saved as `[x, y, brush_width]` points in draw order, preserving directionality for tube-tracking models
- **Multiple instances** — press `]` or tap ⊕ to add a second/third entity of the same structure; all instances share one class ID on export but are recorded as separate strokes
- **Per-case instance isolation** — instances reset between cases; navigating back to a case restores its exact annotation state
- **Save & Next** — writes `{case}_strokes.json` to an output folder (iCloud Drive, local storage) and advances to the next image
- **Session backup** — `annotations.json` stores all in-session canvas states for restore
- **PWA** — installable to iPad Home Screen for fullscreen, offline-capable use

## Output format

Each `_strokes.json` file contains:

```json
{
  "image": "case001.jpg",
  "image_size": [H, W],
  "structures": [
    {"channel": 0, "name": "ETT", "color": "#ff4d4d"},
    ...
  ],
  "strokes": [
    {
      "stroke_index": 0,
      "structure": "ETT",
      "sid": 0,
      "instance": 1,
      "n_points": 187,
      "points": [[x, y, width], ...]
    }
  ]
}
```

`points` are in image-pixel coordinates, in draw order. `width` is the rendered brush diameter at that sample (varies with Apple Pencil pressure).

### Load and reconstruct masks in Python

```python
import json, numpy as np, cv2

def load_strokes(path):
    """Returns (H, W, N_structures) uint8 mask and metadata dict."""
    d = json.load(open(path))
    H, W = d["image_size"]
    N = len(d["structures"])
    ch_map = {s["name"]: s["channel"] for s in d["structures"]}
    channels = [np.zeros((H, W), dtype=np.uint8) for _ in range(N)]
    for stroke in d["strokes"]:
        ch = ch_map.get(stroke["structure"], -1)
        if ch < 0:
            continue
        pts = np.array(stroke["points"])          # (K, 3): x, y, width
        for i in range(len(pts) - 1):
            x0, y0, w0 = pts[i];  x1, y1, w1 = pts[i + 1]
            cv2.line(channels[ch], (int(x0), int(y0)), (int(x1), int(y1)),
                     255, max(1, int(round((w0 + w1) / 2))), cv2.LINE_AA)
    return np.stack(channels, axis=-1), d

mask, meta = load_strokes("case001_strokes.json")
ett_mask = mask[:, :, 0]   # ETT binary mask (H, W), values 0 or 255
```

See `test_strokes.ipynb` for a full validation and visualisation notebook.

## Usage

### In the browser (desktop or iPad via Wi-Fi)

```bash
cd /path/to/LDA_annotator
python3 -m http.server 8080
# Open http://<your-mac-ip>:8080/tube-annotator.html on iPad Safari
```

### As a PWA on iPad (recommended)

1. Open the URL above in Safari
2. Tap **Share → Add to Home Screen**
3. Launch from the Home Screen icon — runs fullscreen, works offline after first load

### Via GitHub Pages (no Mac needed after setup)

Enable Pages in repo Settings → Pages → Deploy from branch `main` / root.  
App will be live at `https://sergiosgatidis.github.io/LDA-annotator/tube-annotator.html`

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `1`–`9` | Select structure by list position |
| `B` | Brush tool |
| `E` | Eraser tool |
| `]` or `+` | Add next instance of active structure |
| `⌘Z` / `Ctrl+Z` | Undo |

## Workflow

1. **Open images** — select multiple files (sorted naturally)
2. **Set output folder** — pick an iCloud Drive or local folder (Safari/iOS 16+)
3. **Annotate** — select structure, draw with Pencil
4. **Save & Next** → writes `{name}_strokes.json` and advances
5. On the last case the button becomes **Save ✓**
6. Load `_strokes.json` files in Python for training
