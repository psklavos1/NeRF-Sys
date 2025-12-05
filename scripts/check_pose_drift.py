#!/usr/bin/env python3
import sys, numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # adjust as needed
from data.colmap_utils import read_model, qvec2rotmat


def rot_angle_deg(R):
    tr = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_model", required=True)
    ap.add_argument("--new_model", required=True)
    ap.add_argument("--max_dt", type=float, default=0.02)  # meters
    ap.add_argument("--max_ddeg", type=float, default=0.2)  # degrees
    a = ap.parse_args()

    _, ref_imgs, _ = read_model(a.ref_model)
    _, new_imgs, _ = read_model(a.new_model)

    common = set(ref_imgs.keys()) & set(new_imgs.keys())
    worst_dt = 0.0
    worst_ddeg = 0.0
    offenders = []

    for k in common:
        r = ref_imgs[k]
        n = new_imgs[k]
        Rr = qvec2rotmat(r.qvec)
        tr = np.asarray(r.tvec).reshape(3, 1)
        Rn = qvec2rotmat(n.qvec)
        tn = np.asarray(n.tvec).reshape(3, 1)
        Cr = (-Rr.T @ tr).reshape(3)
        Cn = (-Rn.T @ tn).reshape(3)
        dt = np.linalg.norm(Cr - Cn)
        dR = rot_angle_deg(Rr.T @ Rn)
        worst_dt = max(worst_dt, dt)
        worst_ddeg = max(worst_ddeg, dR)
        if dt > a.max_dt or dR > a.max_ddeg:
            offenders.append((k, dt, dR))

    if offenders:
        print("[FAIL] Drift exceeds thresholds:")
        for k, dt, dR in sorted(offenders, key=lambda x: (-x[1], -x[2]))[:10]:
            print(f"  image_id={k}  Δt={dt:.4f} m  ΔR={dR:.3f}°")
        print(f"Worst Δt={worst_dt:.4f} m  Worst ΔR={worst_ddeg:.3f}°")
        sys.exit(2)

    print(
        f"[PASS] Drift within thresholds. Worst Δt={worst_dt:.4f} m  Worst ΔR={worst_ddeg:.3f}°"
    )
