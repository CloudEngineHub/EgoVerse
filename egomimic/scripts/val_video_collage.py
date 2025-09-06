#!/usr/bin/env python3
import argparse, math, shlex, subprocess
from pathlib import Path
from typing import List, Tuple

# 🔑 Hard-coded mapping: edit this dict directly
# Keys = directories containing .mp4 files (concat per dir)
# Values = label to render on that tile
LABEL_MAP = {
    "/nethome/rpunamiya6/flash/Projects/EgoVerse/logs/cup_on_saucer_benchmark/1_lab_64latent_resnet_2025-08-30_15-19-25/0/videos/epoch_1799/ARIA_BIMANUAL": "ResNet/1 Lab",
    "/nethome/rpunamiya6/flash/Projects/EgoVerse/logs/cup_on_saucer_benchmark/2_lab_64latent_resnet_2025-08-30_15-18-23/0/videos/epoch_1799/ARIA_BIMANUAL": "ResNet/2 Lab",
    "/nethome/rpunamiya6/flash/Projects/EgoVerse/logs/cup_on_saucer_benchmark/3_lab_64latent_resnet_2025-08-30_15-14-18/0/videos/epoch_1799/ARIA_BIMANUAL": "ResNet/3 Lab",
}

def list_mp4s(dir_path: str) -> List[str]:
    files = sorted(str(x) for x in Path(dir_path).glob("*.mp4"))
    if not files:
        raise FileNotFoundError(f"No .mp4 files found in: {dir_path}")
    return files

def write_concat_file(mp4s: List[str], outfile: Path) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        for p in mp4s:
            f.write(f"file '{p}'\n")

def probe_first_size(dir_path: str) -> Tuple[int, int]:
    first = list_mp4s(dir_path)[0]
    out = subprocess.check_output(
        ["ffprobe","-v","error","-select_streams","v:0",
         "-show_entries","stream=width,height","-of","csv=p=0", first],
        text=True
    ).strip()
    w, h = out.split(",")
    return int(h), int(w)  # (H, W)

def escape_drawtext(text: str) -> str:
    return text.replace("\\","\\\\").replace(":","\\:").replace("'","\\'")

def compute_grid(n: int, cols: int | None) -> tuple[int,int]:
    if cols is None or cols <= 0:
        cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols

def main():
    ap = argparse.ArgumentParser(description="NxM tiled collage via FFmpeg (concat per dir, dirs from LABEL_MAP).")
    ap.add_argument("--out", type=str, default="collage_latents.mp4")
    ap.add_argument("--tile_h", type=int, default=-1)
    ap.add_argument("--tile_w", type=int, default=-1)
    ap.add_argument("--cols", type=int, default=0, help="Number of columns (default: ceil(sqrt(N))).")
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--bitrate", type=str, default="8M")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--fontfile", type=str, default="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    args = ap.parse_args()

    # 1) Collect dirs + labels from LABEL_MAP (order preserved)
    if not LABEL_MAP:
        raise ValueError("LABEL_MAP is empty. Please add dir->label entries.")
    dirs = list(LABEL_MAP.keys())
    labels = [LABEL_MAP[d] if LABEL_MAP[d] else Path(d).name for d in dirs]

    # 2) Build concat lists
    workdir = Path(".ffmpeg_lists")
    list_files: List[str] = []
    for i, d in enumerate(dirs):
        lf = workdir / f"list_{i}.txt"
        write_concat_file(list_mp4s(d), lf)
        list_files.append(str(lf))

    # 3) Determine tile size
    if args.tile_h < 0 or args.tile_w < 0:
        ph, pw = probe_first_size(dirs[0])
        tile_h = ph if args.tile_h < 0 else args.tile_h
        tile_w = pw if args.tile_w < 0 else args.tile_w
    else:
        tile_h, tile_w = args.tile_h, args.tile_w

    # 4) Grid
    n_inputs = len(list_files)
    rows, cols = compute_grid(n_inputs, args.cols if args.cols > 0 else None)

    # 5) FFmpeg inputs
    cmd = ["ffmpeg","-y","-hide_banner","-loglevel","warning","-threads",str(args.threads)]
    for lf in list_files:
        cmd += ["-f","concat","-safe","0","-i",lf]

    # 6) Filter graph: per-input + xstack
    setpts_expr = f"{1.0/args.speed}*PTS"
    filters = []
    for i, d in enumerate(dirs):
        lab = escape_drawtext(labels[i])
        chain = f"[{i}:v]setpts={setpts_expr},scale={tile_w}:{tile_h}:flags=bicubic"
        if args.labels:
            chain += (
                f",drawbox=x=0:y=0:w={tile_w}:h=28:color=black@0.65:t=fill"
                f",drawtext=fontfile='{args.fontfile}':text='{lab}':x=8:y=4:fontsize=20:fontcolor=white"
            )
        chain += f"[v{i}]"
        filters.append(chain)

    layout_elems = []
    for i in range(n_inputs):
        r, c = divmod(i, cols)
        x = c * tile_w
        y = r * tile_h
        layout_elems.append(f"{x}_{y}")

    inputs_labels = "".join(f"[v{i}]" for i in range(n_inputs))
    filters.append(f"{inputs_labels}xstack=inputs={n_inputs}:layout=" + "|".join(layout_elems) + "[vout]")

    # 7) Encoding (MPEG-4; swap to libx264 if you prefer)
    encode = [
        "-map","[vout]","-r",str(args.fps),
        "-c:v","mpeg4","-qscale:v","3",
        "-pix_fmt","yuv420p","-movflags","+faststart"
    ]

    cmd += ["-filter_complex",";".join(filters)] + encode + [args.out]

    # 8) Run
    print("Grid:", rows, "rows x", cols, "cols  |  tiles:", n_inputs)
    print("Tile size:", tile_w, "x", tile_h)
    print("Running FFmpeg command:\n"," ".join(shlex.quote(c) for c in cmd))
    subprocess.check_call(cmd)

    final_cols = min(cols, n_inputs)
    print(f"✅ Saved {args.out} ({final_cols*tile_w}x{rows*tile_h} @ {args.fps} fps, speed={args.speed}x)")

if __name__ == "__main__":
    main()
