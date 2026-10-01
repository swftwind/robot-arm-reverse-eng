"""
Checkpoint 1 helper: extract good frames from a reference video, annotate
components, and calibrate scale from a known reference feature.

Setup:
    pip install opencv-python numpy yt-dlp

Workflow:
    python checkpoint1_tool.py download "https://youtu.be/yhdL4jz74WM" -o video.mp4
    python checkpoint1_tool.py extract video.mp4 --interval 0.5 --window 3 --max 40
    # look at frames/contact_sheet.jpg, delete frames you don't want
    python checkpoint1_tool.py annotate frames/
    # annotated images + annotations.json land in annotated/

Annotate controls (click on the image window):
    b  draw a labeled box around a component (drag, then ENTER; type label in terminal)
    l  draw a SCALE REFERENCE line (click 2 points; type known length in mm in terminal)
    m  measure a distance using the last scale from this frame (click 2 points)
    u  undo last annotation
    n  save and go to next frame
    p  save and go to previous frame
    q  save and quit
"""
import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------- download
def cmd_download(args):
    subprocess.run(
        ["yt-dlp", "-f", "bv*[height<=1080]/b", "-o", args.out, args.url],
        check=True,
    )


# ----------------------------------------------------------------- extract
def sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def cmd_extract(args):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps * args.interval))
    print(f"Video: {total} frames @ {fps:.1f} fps (~{total / fps:.0f}s). Sampling every {step} frames.")

    # 1) one sequential pass, scoring sharpness at each sample point
    cands = []
    idx = 0
    while cap.grab():
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                c = {
                    "idx": idx,
                    "t": idx / fps,
                    "sharp": sharpness(gray),
                    "thumb": cv2.resize(gray, (64, 36)).astype(np.float32),
                }
                cands.append(c)
                print(f"\r  scanning: {100 * idx / max(total, 1):5.1f}%  "
                      f"t={c['t']:7.2f}s  candidates={len(cands)}", end="", flush=True)
        idx += 1
    print(f"\nSampled {len(cands)} candidate frames")

    # 2) sharpest frame per time window
    best = {}
    for c in cands:
        w = int(c["t"] // args.window)
        if w not in best or c["sharp"] > best[w]["sharp"]:
            best[w] = c
    picks = [best[w] for w in sorted(best)]

    # 3) drop near-duplicate views
    kept = []
    for c in picks:
        if all(np.abs(c["thumb"] - k["thumb"]).mean() > args.diff for k in kept):
            kept.append(c)
            print(f"  found frame at t={c['t']:7.2f}s (sharpness {c['sharp']:.0f})")

    # 4) cap the count, keeping the sharpest, then restore time order
    if len(kept) > args.max:
        kept = sorted(kept, key=lambda c: -c["sharp"])[: args.max]
        kept.sort(key=lambda c: c["idx"])
    print(f"Keeping {len(kept)} frames")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    thumbs = []
    for n, c in enumerate(kept, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, c["idx"])
        ok, frame = cap.read()
        if not ok:
            continue
        name = f"frame_t{c['t']:07.2f}s.png"
        cv2.imwrite(str(out / name), frame)
        print(f"  saved {n}/{len(kept)}: {name}")
        small = cv2.resize(frame, (320, 180))
        cv2.putText(small, f"{c['t']:.1f}s", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)
        thumbs.append(small)

    if thumbs:
        cols = 4
        while len(thumbs) % cols:
            thumbs.append(np.zeros_like(thumbs[0]))
        rows = [np.hstack(thumbs[i:i + cols]) for i in range(0, len(thumbs), cols)]
        cv2.imwrite(str(out / "contact_sheet.jpg"), np.vstack(rows))
    print(f"Done. Saved to {out}/ (see contact_sheet.jpg)")


# ---------------------------------------------------------------- annotate
class Clicker:
    def __init__(self, win):
        self.win, self.pts = win, []
        cv2.setMouseCallback(win, self.on_mouse)

    def on_mouse(self, event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pts.append((x, y))

    def get_points(self, img, n):
        self.pts = []
        while len(self.pts) < n:
            disp = img.copy()
            for p in self.pts:
                cv2.circle(disp, p, 4, (0, 0, 255), -1)
            cv2.imshow(self.win, disp)
            if cv2.waitKey(20) & 0xFF == 27:  # ESC cancels
                return None
        return list(self.pts)


def render(base, anns):
    img = base.copy()
    for a in anns:
        if a["type"] == "box":
            x, y, w, h = a["rect"]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(img, a["label"], (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 200, 0), 2, cv2.LINE_AA)
        else:  # scale or measure line
            color = (0, 140, 255) if a["type"] == "scale" else (255, 100, 0)
            p1, p2 = map(tuple, a["pts"])
            cv2.line(img, p1, p2, color, 2)
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 8)
            cv2.putText(img, a["label"], mid, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2, cv2.LINE_AA)
    return img


def cmd_annotate(args):
    src = Path(args.folder)
    files = sorted(p for p in src.glob("*.png"))
    if not files:
        raise SystemExit("No PNG frames found")
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    db_path = out / "annotations.json"
    db = json.loads(db_path.read_text()) if db_path.exists() else {}

    win = "annotate"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    clicker = Clicker(win)
    i = 0
    while 0 <= i < len(files):
        f = files[i]
        base = cv2.imread(str(f))
        anns = db.get(f.name, [])
        print(f"\n[{i + 1}/{len(files)}] {f.name}")
        while True:
            cv2.imshow(win, render(base, anns))
            k = cv2.waitKey(30) & 0xFF
            if k == ord("b"):
                r = cv2.selectROI(win, render(base, anns), showCrosshair=False)
                if r[2] and r[3]:
                    label = input("  component label: ").strip() or "component"
                    anns.append({"type": "box", "rect": list(map(int, r)), "label": label})
                clicker = Clicker(win)  # selectROI resets the callback
            elif k == ord("l"):
                pts = clicker.get_points(render(base, anns), 2)
                if pts:
                    name = input("  reference feature name (e.g. 'table hole pitch'): ").strip()
                    mm = float(input("  known length in mm: "))
                    px = math.dist(*pts)
                    anns.append({"type": "scale", "pts": pts, "mm": mm, "px": px,
                                 "px_per_mm": px / mm,
                                 "label": f"{name}: {mm:g} mm ({px / mm:.2f} px/mm)"})
                    print(f"  -> {px / mm:.3f} px/mm")
            elif k == ord("m"):
                scales = [a for a in anns if a["type"] == "scale"]
                if not scales:
                    print("  draw a scale line first (l)")
                    continue
                pts = clicker.get_points(render(base, anns), 2)
                if pts:
                    mm = math.dist(*pts) / scales[-1]["px_per_mm"]
                    anns.append({"type": "measure", "pts": pts, "label": f"~{mm:.1f} mm"})
                    print(f"  -> ~{mm:.1f} mm (valid only near the reference's plane/depth)")
            elif k == ord("u") and anns:
                anns.pop()
            elif k in (ord("n"), ord("p"), ord("q")):
                db[f.name] = anns
                cv2.imwrite(str(out / f.name), render(base, anns))
                db_path.write_text(json.dumps(db, indent=2))
                if k == ord("q"):
                    cv2.destroyAllWindows()
                    return
                i += 1 if k == ord("n") else -1
                break
    cv2.destroyAllWindows()


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download")
    d.add_argument("url")
    d.add_argument("-o", "--out", default="video.mp4")
    d.set_defaults(fn=cmd_download)

    e = sub.add_parser("extract")
    e.add_argument("video")
    e.add_argument("--out", default="frames")
    e.add_argument("--interval", type=float, default=0.5, help="sampling step in seconds")
    e.add_argument("--window", type=float, default=3.0, help="keep sharpest frame per N seconds")
    e.add_argument("--diff", type=float, default=6.0, help="min visual difference to count as a new view")
    e.add_argument("--max", type=int, default=40)
    e.set_defaults(fn=cmd_extract)

    a = sub.add_parser("annotate")
    a.add_argument("folder")
    a.add_argument("--out", default="annotated")
    a.set_defaults(fn=cmd_annotate)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()