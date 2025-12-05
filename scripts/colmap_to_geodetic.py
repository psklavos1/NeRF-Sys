#!/usr/bin/env python3
"""
colmap_camera_to_gps.py

Usage examples:
# If you already have an aligned model (in ECEF):
python colmap_camera_to_gps.py --images_bin_aligned ./photos_2048_aligned/images.bin --out cameras_ecef.csv

# If you only have original (unaligned) model and also the aligned model you ran earlier:
python colmap_camera_to_gps.py --images_bin_to_transform ./photos_2048_out/colmap/sparse_registered/images.bin \
    --orig_images_bin ./photos_2048_out/colmap/sparse/0/images.bin \
    --aligned_images_bin ./photos_2048_aligned/images.bin \
    --out cameras_ecef.csv
"""
import argparse, struct, math, numpy as np
from pathlib import Path


# -------------------- I/O helpers for images.bin --------------------
def read_next_bytes(fid, num_bytes, fmt, endian="<"):
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError("Bad read")
    return struct.unpack(endian + fmt, data)


def read_images_bin(path):
    images = {}
    with open(path, "rb") as f:
        n = read_next_bytes(f, 8, "Q")[0]
        for _ in range(n):
            vals = read_next_bytes(f, 64, "idddddddi")
            img_id = vals[0]
            qvec = np.array(vals[1:5], dtype=float)
            tvec = np.array(vals[5:8], dtype=float)
            cam_id = vals[8]
            # name: null-terminated string
            name_b = bytearray()
            ch = read_next_bytes(f, 1, "c")[0]
            while ch != b"\x00":
                name_b += ch
                ch = read_next_bytes(f, 1, "c")[0]
            name = name_b.decode("utf-8")
            # skip 2D points (we don't need them)
            n2d = read_next_bytes(f, 8, "Q")[0]
            if n2d > 0:
                _ = read_next_bytes(f, 24 * n2d, "ddq" * n2d)
            images[img_id] = {
                "id": img_id,
                "qvec": qvec,
                "tvec": tvec,
                "cam_id": cam_id,
                "name": name,
            }
    return images


# -------------------- quaternion / rotation helpers --------------------
def qvec_to_rotmat(q):
    qw, qx, qy, qz = q
    w, x, y, z = qw, qx, qy, qz
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
    return R


def rotmat_to_qvec(R):
    m = R
    t = np.trace(m)
    if t > 0:
        S = math.sqrt(t + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m[2, 1] - m[1, 2]) / S
        qy = (m[0, 2] - m[2, 0]) / S
        qz = (m[1, 0] - m[0, 1]) / S
    else:
        if (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            S = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / S
            qx = 0.25 * S
            qy = (m[0, 1] + m[1, 0]) / S
            qz = (m[0, 2] + m[2, 0]) / S
        elif m[1, 1] > m[2, 2]:
            S = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / S
            qx = (m[0, 1] + m[1, 0]) / S
            qy = 0.25 * S
            qz = (m[1, 2] + m[2, 1]) / S
        else:
            S = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / S
            qx = (m[0, 2] + m[2, 0]) / S
            qy = (m[1, 2] + m[2, 1]) / S
            qz = 0.25 * S
    q = np.array([qw, qx, qy, qz], dtype=float)
    return q / np.linalg.norm(q)


def camera_center_from_qt(q, t):
    R = qvec_to_rotmat(q)
    C = -R.T.dot(t)
    return C


# -------------------- Umeyama similarity (model -> aligned) --------------------
def estimate_similarity_umeyama(X, Y, with_scale=True):
    X = np.asarray(X)
    Y = np.asarray(Y)
    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)
    Xc = X - mu_X
    Yc = Y - mu_Y
    cov = (Yc.T @ Xc) / X.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    if with_scale:
        var_X = (Xc**2).sum() / X.shape[0]
        s = np.trace(np.diag(D) @ S) / var_X
    else:
        s = 1.0
    t = mu_Y - s * R @ mu_X
    return s, R, t


# -------------------- ECEF <-> geodetic (WGS84) --------------------
# WGS84 constants
a = 6378137.0  # semi-major axis (m)
f = 1 / 298.257223563
b = a * (1 - f)
e2 = f * (2 - f)


def ecef_to_geodetic(x, y, z):
    # returns lat (deg), lon (deg), alt (meters)
    # iterative solution for latitude
    lon = math.degrees(math.atan2(y, x))
    p = math.sqrt(x * x + y * y)
    # initial lat
    lat = math.atan2(z, p * (1 - e2))
    lat_prev = 0
    while abs(lat - lat_prev) > 1e-12:
        sinlat = math.sin(lat)
        N = a / math.sqrt(1 - e2 * sinlat * sinlat)
        alt = p / math.cos(lat) - N
        lat_prev = lat
        lat = math.atan2(z, p * (1 - e2 * (N / (N + alt))))
    lat_deg = math.degrees(lat)
    sinlat = math.sin(lat)
    N = a / math.sqrt(1 - e2 * sinlat * sinlat)
    alt = p / math.cos(lat) - N
    return lat_deg, lon, alt


# -------------------- main -------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--images_bin_aligned",
        type=str,
        default="data/drz/model/images.bin",
        help="If you already have an aligned images.bin: read it and convert centers->GPS",
    )
    p.add_argument(
        "--images_bin_to_transform",
        type=str,
        default=None,
        help="images.bin whose poses you want to transform to GPS (model coordinates)",
    )
    p.add_argument(
        "--orig_images_bin",
        type=str,
        default=None,
        help="original unaligned images.bin used together with aligned_images_bin to compute similarity",
    )
    p.add_argument(
        "--aligned_images_bin",
        type=str,
        default=None,
        help="aligned images.bin corresponding to orig_images_bin for computing similarity (Umeyama)",
    )
    p.add_argument(
        "--out", type=str, default="cameras_ecef_2.csv", help="output CSV path"
    )
    args = p.parse_args()

    if args.images_bin_aligned is None and args.images_bin_to_transform is None:
        raise SystemExit(
            "Provide --images_bin_aligned (already aligned) or --images_bin_to_transform with --orig_images_bin & --aligned_images_bin"
        )

    # Case 1: already aligned
    if args.images_bin_aligned:
        imgs = read_images_bin(args.images_bin_aligned)
        # compute centers and ECEF->geodetic
        rows = []
        for rec in imgs.values():
            q = rec["qvec"]
            t = rec["tvec"]
            C = camera_center_from_qt(q, t)  # already in aligned frame (ECEF)
            lat, lon, alt = ecef_to_geodetic(C[0], C[1], C[2])
            rows.append((rec["name"], rec["cam_id"], C, (lat, lon, alt), q, t))
        # write CSV
        with open(args.out, "w") as f:
            f.write(
                "name,cam_id,Cx_ecef,Cy_ecef,Cz_ecef,lat,lon,alt,qw,qx,qy,qz,tx,ty,tz\n"
            )
            for r in rows:
                name, camid, C, geod, q, t = r
                lat, lon, alt = geod
                f.write(
                    f"{name},{camid},{C[0]:.6f},{C[1]:.6f},{C[2]:.6f},{lat:.8f},{lon:.8f},{alt:.3f},{q[0]:.9f},{q[1]:.9f},{q[2]:.9f},{q[3]:.9f},{t[0]:.6f},{t[1]:.6f},{t[2]:.6f}\n"
                )
        print("Wrote", args.out)
        return

    # Case 2: compute similarity from orig_images_bin -> aligned_images_bin and apply to images_bin_to_transform
    if not (
        args.images_bin_to_transform
        and args.orig_images_bin
        and args.aligned_images_bin
    ):
        raise SystemExit(
            "If not using --images_bin_aligned, provide --images_bin_to_transform AND --orig_images_bin AND --aligned_images_bin"
        )

    orig = read_images_bin(args.orig_images_bin)
    aligned = read_images_bin(args.aligned_images_bin)
    totran = read_images_bin(args.images_bin_to_transform)

    # find common names between orig and aligned
    name_to_orig = {v["name"]: k for k, v in orig.items()}
    name_to_al = {v["name"]: k for k, v in aligned.items()}
    common = [name for name in name_to_orig.keys() if name in name_to_al]
    if len(common) < 3:
        raise SystemExit(
            "Need >=3 common images between orig and aligned to estimate similarity transform; found %d"
            % len(common)
        )

    X = []
    Y = []
    for name in common:
        oid = name_to_orig[name]
        aid = name_to_al[name]
        C1 = camera_center_from_qt(orig[oid]["qvec"], orig[oid]["tvec"])
        C2 = camera_center_from_qt(aligned[aid]["qvec"], aligned[aid]["tvec"])
        X.append(C1)
        Y.append(C2)
    X = np.vstack(X)
    Y = np.vstack(Y)
    s, R_align, t_align = estimate_similarity_umeyama(X, Y, with_scale=True)
    print(
        "Estimated similarity: scale s=%.9f, det(R)=%.6f, |t|=%.6f"
        % (s, np.linalg.det(R_align), np.linalg.norm(t_align))
    )

    rows = []
    for rec in totran.values():
        q = rec["qvec"]
        t = rec["tvec"]
        C_model = camera_center_from_qt(q, t)
        C_ecef = s * (R_align @ C_model) + t_align
        # new camera rotation in ECEF frame
        R_cam_model = qvec_to_rotmat(q)
        R_cam_ecef = R_align @ R_cam_model
        q_new = rotmat_to_qvec(R_cam_ecef)
        # new tvec in ECEF
        t_new = -R_cam_ecef @ C_ecef
        lat, lon, alt = ecef_to_geodetic(C_ecef[0], C_ecef[1], C_ecef[2])
        rows.append((rec["name"], rec["cam_id"], C_ecef, (lat, lon, alt), q_new, t_new))
    with open(args.out, "w") as f:
        f.write(
            "name,cam_id,Cx_ecef,Cy_ecef,Cz_ecef,lat,lon,alt,qw,qx,qy,qz,tx,ty,tz\n"
        )
        for r in rows:
            name, camid, C, geod, q, t = r
            lat, lon, alt = geod
            f.write(
                f"{name},{camid},{C[0]:.6f},{C[1]:.6f},{C[2]:.6f},{lat:.8f},{lon:.8f},{alt:.3f},{q[0]:.9f},{q[1]:.9f},{q[2]:.9f},{q[3]:.9f},{t[0]:.6f},{t[1]:.6f},{t[2]:.6f}\n"
            )
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
