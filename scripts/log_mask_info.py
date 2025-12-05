#!/usr/bin/env python3
"""
Mega-NeRF Mask Overlap Auditor (quick ref)

Purpose
- Scan per-centroid mask folders and report per-image & aggregate overlap stats.
- Optional exclusivity check (fail if any pixel belongs to >1 centroid).

Input layout
- --mask_path/
    <cid>/  <image_id>.pt|.npy|.zip  (cid = integer)

Supported mask file types
- zipped .pt (inner .pt), plain .pt, .npy, or .zip containing a .npy (prefers mask.npy)
- Non-zero ⇒ True (boolean)

Outputs
- <mask_path>/stats.txt (also printed via SimpleLogger)
- Per image: % per-cell (exclusive), % overlap, % unassigned, top overlap combos
- Global: true-pixel coverage per cell; counts for sum=0 / sum=1 / sum>1

Exit codes
- 0: OK (exclusive satisfied or not enforced)
- 1: exclusivity violated (with --expect_exclusive)
- 2: structure issues (no cells or no masks)

Example Usage
----------------
    ./scripts/log_mask_info.py --mask_path data/drz/out/active/masks/g22_kmeans_bm115
"""

import argparse, os, zipfile, sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Tuple
import numpy as np
import torch
from common.utils import SimpleLogger as Logger

# EX. ./scripts/log_mask_info.py --mask_path data/drz/out/active/masks/g22_kmeans_bm115


# ---------- helpers ----------
def _is_int_dir(name: str) -> bool:
    return name.isdigit()


def _stem(fname: str) -> str:
    base = os.path.basename(fname)
    for ext in (".pt", ".npy", ".zip"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return os.path.splitext(base)[0]


def _load_mask(path: str) -> np.ndarray:
    """
    Load a mask file and return a boolean numpy array (H, W).

    Supports:
      1) ZIPPED .PT  (your format, created by write_zipped_tensor)
      2) Plain .PT   (torch.save/torch.load)
      3) .NPY        (numpy array)
      4) .ZIP        (contains 'mask.npy' or any .npy)

    Any nonzero value is treated as True.
    """
    ext = os.path.splitext(path)[1].lower()

    # 1) Try "zipped .pt" first (a zip containing an inner .pt)
    if ext == ".pt":
        try:
            with zipfile.ZipFile(path, "r") as zf:
                names = zf.namelist()
                if not names:
                    raise ValueError(f"{path} is an empty zip")
                inner = os.path.basename(path)
                if inner not in names:
                    inner = names[0]
                with zf.open(inner) as f:
                    if torch is None:
                        raise RuntimeError("torch not available to load inner .pt")
                    obj = torch.load(f, map_location="cpu")
        except zipfile.BadZipFile:
            if torch is None:
                raise RuntimeError(f"Cannot load {path}: torch not available.")
            obj = torch.load(path, map_location="cpu")

        if isinstance(obj, dict):
            for k in ("mask", "masks", "data"):
                if k in obj:
                    obj = obj[k]
                    break
            if isinstance(obj, dict):
                vals = [v for v in obj.values() if hasattr(v, "shape")]
                if vals:
                    obj = vals[0]
        if hasattr(obj, "detach"):
            obj = obj.detach().cpu().numpy()
        elif hasattr(obj, "numpy"):
            obj = obj.numpy()
        return (obj != 0).astype(np.bool_)

    if ext == ".npy":
        arr = np.load(path)
        return (arr != 0).astype(np.bool_)

    if ext == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            npys = [n for n in zf.namelist() if n.endswith(".npy")]
            if not npys:
                raise ValueError(f"{path} contains no .npy")
            name = "mask.npy" if "mask.npy" in npys else npys[0]
            with zf.open(name) as f:
                arr = np.load(f)
        return (arr != 0).astype(np.bool_)

    raise ValueError(f"Unsupported mask file type: {path}")


def _index_submodule_files(
    mask_path: str,
) -> Tuple[List[int], Dict[int, Dict[str, str]]]:
    submods = [
        int(d)
        for d in os.listdir(mask_path)
        if os.path.isdir(os.path.join(mask_path, d)) and _is_int_dir(d)
    ]
    submods.sort()
    files: Dict[int, Dict[str, str]] = {}
    for sid in submods:
        sdir = os.path.join(mask_path, str(sid))
        mapping: Dict[str, str] = {}
        for fname in os.listdir(sdir):
            if fname.endswith((".pt", ".npy", ".zip")):
                mapping[_stem(fname)] = os.path.join(sdir, fname)
        files[sid] = mapping
    return submods, files


def _collect_image_ids(files: Dict[int, Dict[str, str]]) -> List[str]:
    ids = set()
    for m in files.values():
        ids.update(m.keys())
    return sorted(ids, key=lambda x: (len(x), x))


def _popcount(x: int) -> int:
    return x.bit_count()  # py3.8+: use bin(x).count("1") if needed


def _format_combo(mask_code: int, submods: List[int]) -> str:
    # mask_code's bits map to submods order
    members = []
    for bit_idx, sid in enumerate(submods):
        if mask_code & (1 << bit_idx):
            members.append(str(sid))
    return "&".join(members)


# ---------- main check (refactored) ----------
def gen_mask_stats(mask_path: str, expect_exclusive: bool, topk: int = 10) -> int:
    """
    Generate and print detailed mask stats, and write the same output to <mask_path>/stats.txt.
    Logs per-image exclusive percentages per cell, overlap breakdown, and global aggregates.

    Returns
    -------
    int
        0 on success; 1 if exclusivity enforced and violated; 2 for structural issues.
    """
    stats_path = os.path.join(mask_path, "stats.txt")
    log = Logger(stats_path)

    # How many overlap combos to print per image (sorted by descending %)
    MAX_OVERLAP_LINES = 8

    try:
        submods, files = _index_submodule_files(mask_path)
        if not submods:
            log.write(f"No submodule dirs found under {mask_path}")
            return 2
        image_ids = _collect_image_ids(files)
        if not image_ids:
            log.write(f"No mask files found under submodule dirs at {mask_path}")
            return 2

        log.write(f"[INFO] Found {len(submods)} submodules: {submods}")
        log.write(f"[INFO] Found {len(image_ids)} images across submodules.")

        coverage = {
            sid: 0 for sid in submods
        }  # total True pixels per cell across all images
        per_image_stats = []
        errors = []
        ref_shape = None

        for img_id in image_ids:
            masks = []
            shapes = set()
            loaded_any = False

            for sid in submods:
                path = files[sid].get(img_id)
                if path is None:
                    masks.append(None)
                    continue
                m = _load_mask(path)
                shapes.add(m.shape)
                masks.append(m)
                loaded_any = True

            if not loaded_any:
                log.write(f"[WARN] {img_id}: missing in all submodules; skipping")
                continue

            # shape reconciliation
            if len(shapes) > 1:
                errors.append(
                    (img_id, f"shape mismatch across submodules: {sorted(shapes)}")
                )
                counts = {}
                for m in masks:
                    if m is None:
                        continue
                    counts[m.shape] = counts.get(m.shape, 0) + 1
                ref_shape = max(counts, key=counts.get)
            else:
                ref_shape = shapes.pop() if shapes else ref_shape

            # build aligned stack (fill missing with zeros, coerce mismatches)
            z = np.zeros(ref_shape, dtype=np.bool_)
            stack_list = []
            for sid, m in zip(submods, masks):
                if m is None:
                    stack_list.append(z)
                    continue
                if m.shape != ref_shape:
                    errors.append(
                        (
                            img_id,
                            f"submodule {sid} shape {m.shape} != ref {ref_shape} (coerced)",
                        )
                    )
                    H = min(m.shape[0], ref_shape[0])
                    W = min(m.shape[1], ref_shape[1])
                    tmp = np.zeros(ref_shape, dtype=np.bool_)
                    tmp[:H, :W] = m[:H, :W]
                    m = tmp
                stack_list.append(m)
                coverage[sid] += int(m.sum())

            stack = np.stack(stack_list, axis=0).astype(np.uint8)  # (K,H,W)
            H, W = ref_shape
            nt = H * W

            # ---- pattern histogram over 2^K codes (safe int64) ----
            K = len(submods)
            if K >= 62:
                raise RuntimeError(
                    f"Too many submodules ({K}); 64-bit bitmask would overflow."
                )
            weights = (1 << np.arange(K, dtype=np.int64)).reshape(-1, 1, 1)
            codes = (
                (stack.astype(np.int64) * weights)
                .sum(axis=0)
                .reshape(-1)
                .astype(np.int64, copy=False)
            )

            max_code = 1 << K
            hist = np.bincount(codes, minlength=max_code).astype(np.int64, copy=False)

            # counts
            n0 = int(hist[0])  # unassigned
            singleton_counts = {submods[i]: int(hist[1 << i]) for i in range(K)}
            overlap_count = int(
                sum(
                    hist[code]
                    for code in range(1, max_code)
                    if (code & (code - 1)) != 0
                )
            )  # popcount>=2
            nt = int(stack.shape[1] * stack.shape[2])  # H*W

            # percentages
            per_cell_pct = {
                sid: (100.0 * singleton_counts[sid] / nt if nt > 0 else 0.0)
                for sid in submods
            }
            overlap_pct = 100.0 * overlap_count / nt if nt > 0 else 0.0
            unassigned_pct = 100.0 * n0 / nt if nt > 0 else 0.0

            # OPTIONAL: overlap combos (pairs, triplets, ...)
            combo_rows = []
            for code in range(1, max_code):
                # popcount >= 2
                if code & (code - 1) and hist[code] > 0:
                    pct = 100.0 * hist[code] / nt if nt > 0 else 0.0
                    combo_rows.append((pct, code))
            combo_rows.sort(reverse=True)  # by pct desc

            # legacy aggregate for summaries
            n1 = sum(singleton_counts.values())
            ngt = overlap_count
            per_image_stats.append((img_id, n0, n1, ngt, nt))

            # ---- logging (dict that sums to ~100) ----
            report_dict_items = [(sid, per_cell_pct[sid]) for sid in submods]
            report_dict_items.append(("overlap", overlap_pct))
            if unassigned_pct > 0.0:
                report_dict_items.append(("unassigned", unassigned_pct))
            report_str = (
                "{" + ", ".join(f"{k}: {v:.2f}" for k, v in report_dict_items) + "}"
            )
            log.write(f"[IMG {img_id}] per-cell % {report_str}")

            # pretty combos line like "1&2: 3.27% | 0&3: 1.02% ..."
            if combo_rows:

                def _format_combo(mask_code: int) -> str:
                    members = []
                    for bit_idx, sid in enumerate(submods):
                        if mask_code & (1 << bit_idx):
                            members.append(str(sid))
                    return "&".join(members)

                line_parts = [
                    f"{_format_combo(code)}: {pct:.2f}%"
                    for pct, code in combo_rows[:MAX_OVERLAP_LINES]
                ]
                log.write("          overlaps: " + " | ".join(line_parts))

        # ----- totals & summary (kept for continuity) -----
        total0 = sum(a for _, a, _, _, _ in per_image_stats)
        total1 = sum(b for _, _, b, _, _ in per_image_stats)
        totalg = sum(c for _, _, _, c, _ in per_image_stats)
        totalt = sum(t for _, _, _, _, t in per_image_stats)

        log.write("\n=== SUMMARY ===")
        log.write(f"Images checked: {len(per_image_stats)}")
        log.write("Per-submodule coverage (True pixels across all images):")
        for sid in submods:
            log.write(f"  - submodule {sid}: {coverage[sid]:,}")

        if totalt > 0:
            log.write("Aggregate pixel distribution:")
            log.write(f"  sum=0 : {total0:,} ({100.0*total0/totalt:.2f}%)")
            log.write(f"  sum=1 : {total1:,} ({100.0*total1/totalt:.2f}%)")
            log.write(f"  sum>1 : {totalg:,} ({100.0*totalg/totalt:.2f}%)")

        ranked = sorted(
            per_image_stats, key=lambda t: (t[3] / max(1, t[4])), reverse=True
        )
        log.write("\nTop images by overlap (sum>1):")
        for i, (img_id, n0, n1, ngt, nt) in enumerate(ranked[:topk]):
            log.write(
                f"  {i+1:2d}. {img_id}: overlap {ngt}/{nt} = {100.0*ngt/max(1,nt):.2f}% | unique {n1} | zero {n0}"
            )

        if expect_exclusive and totalg > 0:
            log.write(
                f"\n[FAIL] Exclusive check enabled but found {totalg:,} overlapped pixels (sum>1)."
            )
            return 1
        else:
            log.write(
                "\n[OK] Completed."
                + (
                    " Exclusive constraint satisfied."
                    if expect_exclusive and totalg == 0
                    else " Exclusive constraint not enforced."
                )
            )
            return 0
    finally:
        log.close()


def main():
    ap = argparse.ArgumentParser(
        description="Verify and summarize Mega-NeRF mask overlaps."
    )
    ap.add_argument(
        "--mask_path",
        type=str,
        required=True,
        help="Root directory with per-submodule mask folders (e.g., .../masks/g22_kmeans_bm1)",
    )
    ap.add_argument(
        "--expect_exclusive",
        action="store_true",
        help="Fail if any pixel belongs to more than one mask (sum>1).",
    )
    ap.add_argument(
        "--topk", type=int, help="Log the 'k' images with most overlap", default=10
    )
    args = ap.parse_args()

    code = gen_mask_stats(args.mask_path, args.expect_exclusive, topk=args.topk)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
