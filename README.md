# Foam Board Skull — Slice Stack Viewer

A PACS-style web viewer celebrating a 13× scale hemisection skull built from foam board in medical school (UMass Chan Medical School) by Matt B.

The skull was made from a coronal maxillofacial CT. Each coronal slice was projected onto foam board and traced by hand, then cut and stacked. Only the left half was built. It stands against a mirror, which completes the skull. The sculpture was later photographed slice by slice as it was taken apart, front to back.

## What the viewer shows

| Panel | Key | Contents |
|---|---|---|
| Photo | 1 | The 114 photos of the sculpture as it is taken apart (plus the empty board) |
| Outline | 2 | A symmetric diagram of each cut face, built from the original tracing images and checked against the photo |
| CT | 3 | The original coronal CT slice for each layer |
| Tracing | 4 | The original left-half tracing image that was projected and traced for each layer (red = next slice) |

The panels are linked to one another, and to the anatomy:

- **Shared view:** scroll, zoom, pan and rotate are shared across panels.
- **Hover crosshair:** a crosshair follows the mouse in every panel.
- **Triangulation marker:** clicking drops a marker fixed to the anatomy. It stays on that spot in every panel and every photo, and shows how many slices in front or behind you are.
- **Overlay (V):** draws the outline on the photo, CT and tracing. **Ghosts (N)** add a trail of neighbouring slices.
- **Labels:** labels can be placed on any slice and exported as JSON.
- **Re-linking:** the ◀ ▶ buttons (or `[` `]`) re-link a photo to a different tracing, and **save links** downloads your corrections.
- **About:** the original UMass page, photos and the time-lapse video.

Press **?** in the viewer for all keyboard shortcuts. Add `#embed` to the URL to hide the toolbar when embedding in an iframe.

## Viewing it

- **Online:** https://matthewbreen.github.io/giant-skull/ (GitHub Pages; the root page forwards to the viewer in `docs/`).
- **Locally:** open `docs/index.html` in a browser. If the browser blocks local files, run `python3 -m http.server` inside `docs/` and open http://localhost:8000.

## Repository layout

```
docs/                 the complete, self-contained website (GitHub Pages root)
  index.html          the viewer; outlines and registration are baked in
  slices/             photos, 720×900 (the original photos are not altered; aspect is corrected on screen)
  masks/              background-removal masks
  ct/                 coronal CT slices 27–134
  trace/              original left-half tracing images 27–134
  about/              photos for the About panel
pipeline/             Python scripts used to build the outlines and registration
  reg.json            per-photo registration: tracing/CT slice number, seam, scale and offset
  outlines_v6.json    outline path for each photo
  lay2.py, mkface.py  pull the traced cut face out of each tracing image
  tw.py, build6.py    map tracing faces onto the photos and build the mirrored outlines
  tfit.py             fit each photo to its tracing (photo scale and position)
  dp4.py, dp4b.py     match photos to tracings, anchored to hand-checked pairs
```

## Notes

- Tracing images #47, #56, #57 and #59 are damaged in the source data; the viewer shows the nearest good slice for those.
- Photos 19, 20 and 60 are exact duplicates of photos 17, 18 and 59 in the source data.
- The photo-to-tracing links were matched automatically and anchored to hand-checked pairs (photo 12 = tracing 36, 22 = 45, 110 = 131). Others may be a slice or two off; corrections are welcome via the viewer's **save links** file.
