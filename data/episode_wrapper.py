from typing import Dict, List, Optional, Any, Iterable, Tuple
from collections import defaultdict
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import warnings

# ----------------------------------------------------------------------
# Defaults / knobs
# ----------------------------------------------------------------------

DEFAULT_YAW_BINS = 8
DEFAULT_PITCH_BINS = 3
DEFAULT_MAX_PER_ORIENT_BIN = 1
DEFAULT_MIN_COVERAGE_PER_SPLIT = 2  # >= this many image-level candidates per split
ANGLE_MIN_DEG = 10.0
ANGLE_MAX_DEG = 120.0
MAX_SEED_RETRIES = 64


# ----------------------------------------------------------------------
# Utility: episode index (optional determinism)
# ----------------------------------------------------------------------
class EpisodeIndex:
    """Stores precomputed (seed, support, query, query_holdout) id lists per episode."""

    def __init__(self) -> None:
        self._items: List[Tuple[int, List[int], List[int], List[int]]] = []

    def add(
        self, seed: int, support: List[int], query: List[int], query_holdout: List[int]
    ) -> None:
        self._items.append((seed, support, query, query_holdout))

    def __len__(self) -> int:
        return len(self._items)

    def get(self, i: int) -> Tuple[int, List[int], List[int], List[int]]:
        return self._items[i % len(self._items)]


# ----------------------------------------------------------------------
# RayRingBuffers: deterministic disjoint streams per image (even/odd)
# ----------------------------------------------------------------------
class RayRingBuffers:
    """
    Maintains a fixed permutation and a moving pointer per image so that:
      • Every masked pixel eventually appears (epoch-like coverage).
      • Support/Query can be made disjoint by construction using even/odd strides.
      • Behavior is deterministic and repeatable across epochs.
    """

    def __init__(self, indices_by_image: Dict[int, Tensor], seed: int = 0) -> None:
        self._rng = np.random.RandomState(seed)
        self._tgen = torch.Generator()
        self._tgen.manual_seed(int(seed) if seed is not None else 0)
        self._img_ids: List[int] = list(sorted(indices_by_image.keys()))
        self._perm: Dict[int, Tensor] = {}
        self._ptr_even: Dict[int, int] = {}
        self._ptr_odd: Dict[int, int] = {}
        self._len: Dict[int, int] = {}
        for img_id in self._img_ids:
            pool = indices_by_image[img_id]
            n = int(pool.numel())
            perm = torch.randperm(n, generator=self._tgen)
            self._perm[img_id] = pool.index_select(0, perm)
            self._ptr_even[img_id] = 0
            self._ptr_odd[img_id] = 1  # disjoint offset relative to even
            self._len[img_id] = n
        # round-robin scheduler over images
        self._rr_idx = 0

    def image_ids(self) -> List[int]:
        return self._img_ids

    def set_epoch(self, epoch: int) -> None:
        """Rotate pointers deterministically; keep permutations fixed for repeatability."""
        stride = 9973  # large prime stride → decorrelated rotations per epoch
        for img_id in self._img_ids:
            n = max(1, self._len[img_id])
            base = (hash((img_id, epoch)) + epoch * stride) % n
            # Even/odd pointers stay one apart to preserve disjoint slices
            self._ptr_even[img_id] = base if base % 2 == 0 else (base - 1) % n
            self._ptr_odd[img_id] = (self._ptr_even[img_id] + 1) % n
        self._rr_idx = epoch % max(1, len(self._img_ids))

    def _take_from_stream(self, img_id: int, k: int, *, odd: bool) -> Tensor:
        if k <= 0:
            return torch.empty(0, dtype=torch.long)
        perm = self._perm[img_id]
        n = int(perm.numel())
        if n == 0:
            return torch.empty(0, dtype=torch.long)
        # select k entries with step 2 to preserve even/odd disjointness
        ptr = self._ptr_odd[img_id] if odd else self._ptr_even[img_id]
        idx = []
        for _ in range(k):
            idx.append(perm[ptr])
            ptr = (ptr + 2) % n
        if odd:
            self._ptr_odd[img_id] = ptr
        else:
            self._ptr_even[img_id] = ptr
        return torch.stack(idx) if idx else torch.empty(0, dtype=torch.long)

    def take(self, img_id: int, k: int, split: str) -> Tensor:
        """
        split ∈ {"support","query"}.
        We return k indices from the even stream ("support") or odd stream ("query").
        """
        if split not in ("support", "query"):
            raise ValueError(f"Invalid split={split}")
        return self._take_from_stream(img_id, k, odd=(split == "query"))





# ----------------------------------------------------------------------
# Episode wrapper
# ----------------------------------------------------------------------
class EpisodeWrapper(Dataset):
    """
    Builds support/query episodes over per-ray data with image-level structure.

    - Uses 1-hop co-visibility candidates with optional angle gating.
    - Enforces pixel-level disjointness via even/odd ring streams when images overlap.
    - When capacity is insufficient, gracefully degrades (non-replacement top-up, then last-resort replacement).

    Returned dict:
        {
          "support": {"rays": [S,D], "rgbs": [S,3], "indices": [S]},
          "query":   {"rays": [Q,D], "rgbs": [Q,3], "indices": [Q]},
          # optional: "query_holdout": {...}  # currently disabled
        }
    """

    def __init__(
        self,
        ray_dataset: Any,  # expects: _rays [N,D], _rgbs [N,3], _img_indices [N]
        support_rays: int,
        query_rays: int,
        cell_id: int,
        *,
        # --- scene graph / candidates ---
        points3D: Optional[Iterable] = None,  # iterable of SfM points with .image_ids
        image_meta_lookup: Optional[Dict[int, Any]] = None,  # img_id -> meta with .c2w
        # --- bins & diversity ---
        yaw_bins: int = DEFAULT_YAW_BINS,
        pitch_bins: int = DEFAULT_PITCH_BINS,
        max_per_orient_bin: int = 2,
        selection_preference_support: Optional[str] = 'farthest',# 'farthest' | 'closest' | 'median'
        selection_preference_query: Optional[str] = 'median',
        min_coverage_support: Optional[int] = 2,
        min_coverage_query: Optional[int] = 1,
        # --- disjointness policy ---
        disjoint_support_query_policy: str = "prefer_disjoint_with_fallback",
        max_support_images: int = 4,
        max_query_images: int = 3,
        # --- seed policy ---
        seed_sampling: str = "inverse",  # 'random' | 'ray_weighted' | 'inverse' | 'round_robin'
        seed: Optional[int] = None,
        max_seed_tries: int = 4,
        # --- determinism & coverage ---
        use_ring_buffers: bool = True,
        precompute_index: bool = False,
        # --- misc ---
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.seed_sampling = str(seed_sampling).lower().strip()
        self.cell_id = None if cell_id is None else int(cell_id)
        self.debug = bool(debug)
        self._seed = int(seed) if seed is not None else 0
        self._rng = np.random.RandomState(self._seed)

        # Share big tensors in CPU shared memory for DataLoader workers
        self._rays: Tensor = ray_dataset._rays.contiguous().cpu()
        self._rays.share_memory_()
        self._rgbs: Tensor = ray_dataset._rgbs.contiguous().cpu()
        self._rgbs.share_memory_()

        img_idx: Tensor = ray_dataset._img_indices.long().cpu().view(-1)
        self._img_of_ray: Tensor = img_idx  # for fast forbid bucketing
        self._indices_by_image: Dict[int, Tensor] = {}
        for img_id_t in torch.unique(img_idx, sorted=True):
            img_id = int(img_id_t.item())
            pool = (img_idx == img_id_t).nonzero(as_tuple=False).squeeze(1)
            if pool.numel() > 0:
                pool = pool.contiguous()
                pool.share_memory_()
                self._indices_by_image[img_id] = pool
        self._image_ids: List[int] = list(sorted(self._indices_by_image.keys()))
        if not self._image_ids:
            raise RuntimeError("No images available.")

        # Split-specific preferences / coverage
        self.selection_preference_support = (selection_preference_support)
        self.selection_preference_query = (selection_preference_query)
        self.min_coverage_support = int(min_coverage_support)
        self.min_coverage_query = int(min_coverage_query)

        # Geometry for angles & bins
        self._yaw_bins = int(yaw_bins)
        self._pitch_bins = int(pitch_bins)
        self._max_per_orient_bin = int(max_per_orient_bin)
        self._centers: Optional[Dict[int, np.ndarray]] = None
        self._view_dirs: Optional[Dict[int, np.ndarray]] = None
        self._orient_bins: Optional[Dict[int, Tuple[int, int]]] = None
        self._build_geometry(image_meta_lookup)

        # Neighbors
        self.use_covis_neighbors = points3D is not None
        if self.use_covis_neighbors:
            self.neighbors = _build_neighbors_from_points3D(
                points3D, set(self._image_ids)
            )
        else:
            self.neighbors = {
                i: [j for j in self._image_ids if j != i] for i in self._image_ids
            }
        if not self.neighbors:
            raise RuntimeError("No neighbors available.")
    
        # Round-robin seed order (only if requested)
        if self.seed_sampling == "round_robin":
            self._seed_rr_order = list(self._image_ids)
            # shuffle once for fairness
            self._rng.shuffle(self._seed_rr_order)
            self._seed_rr_ptr = 0
        
        
        # Disjointness policy
        self.policy = str(disjoint_support_query_policy).lower().strip()
        if self.policy not in (
            "strict_image_disjoint",
            "prefer_disjoint_with_fallback",
            "same_image_only",
        ):
            raise ValueError(
                f"Unknown disjoint policy: {disjoint_support_query_policy}"
            )

        self.support_rays = int(support_rays)
        self.query_rays = int(query_rays)
        self.query_holdout_images = 0  # disabled by default (held-out queries removed)

        self.max_support_images = int(max_support_images)
        self.max_query_images = int(max_query_images)
        self.max_seed_tries = int(max_seed_tries)
        if self.max_seed_tries <= 0:
            self.max_seed_tries = MAX_SEED_RETRIES

        # Optional ring buffers
        self._use_ring_buffers = bool(use_ring_buffers)
        self._rings = RayRingBuffers(self._indices_by_image, seed=self._seed) if self._use_ring_buffers else None

        # Optional EpisodeIndex (precompute image choices once per epoch)
        self._precompute = bool(precompute_index)
        self._episode_index: Optional[EpisodeIndex] = (
            EpisodeIndex() if self._precompute else None
        )

        total_rays = sum(int(v.numel()) for v in self._indices_by_image.values())
        rays_per_ep = max(1, self.support_rays + self.query_rays)
        # If we enable held-out images, we add ~1/2 query size; keep the same denominator to stay conservative
        episodes_per_epoch = int(np.ceil(total_rays / float(rays_per_ep)))
        self._episodes_per_epoch = max(1, int(episodes_per_epoch))

        # Epoch state
        self._epoch = 0
        self.set_epoch(0)

    # --------------------------- public API ---------------------------
    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        # advance RNG deterministically
        self._rng = np.random.RandomState(self._seed + epoch)
        # rotate round-robin seed order and ring buffer pointers
        if self.seed_sampling == "round_robin":
            # Ensure it's initialized even if __init__ path was different
            if not hasattr(self, "_seed_rr_order") or not self._seed_rr_order:
                self._seed_rr_order = list(self._image_ids)
                self._rng.shuffle(self._seed_rr_order)
                self._seed_rr_ptr = 0
            # rotate by epoch to vary start point deterministically
            shift = epoch % max(1, len(self._seed_rr_order))
            self._seed_rr_order = self._seed_rr_order[shift:] + self._seed_rr_order[:shift]
            self._seed_rr_ptr = 0
            
        if self._rings is not None:
            self._rings.set_epoch(epoch)
        # rebuild EpisodeIndex for the new epoch if requested
        if self._precompute and self._episode_index is not None:
            self._episode_index = EpisodeIndex()
            for _ in range(self._episodes_per_epoch):
                seed, sup, qry, qryh = self._build_image_sets()
                self._episode_index.add(seed, sup, qry, qryh)

    def __len__(self) -> int:
        return self._episodes_per_epoch

    # ---------------------- geometry & bins ---------------------------
        # --- instrumentation helpers (inside EpisodeWrapper) ---
    
    def _dbg(self, *args: Any) -> None:
        if self.debug:
            print("[EpisodeWrapper]", *args)

    def _build_geometry(self, image_meta_lookup: Optional[Dict[int, Any]]) -> None:
        if image_meta_lookup is None:
            return
        centers: Dict[int, np.ndarray] = {}
        views: Dict[int, np.ndarray] = {}
        bins: Dict[int, Tuple[int, int]] = {}
        for img_id in self._image_ids:
            meta = image_meta_lookup.get(img_id, None)
            if meta is None:
                continue
            c2w = getattr(meta, "c2w", None)
            if c2w is None:
                continue
            c2w = np.asarray(c2w, dtype=np.float32).reshape(4, 4)
            center = c2w[:3, 3]
            # facing direction = -R[:,2] (for standard camera convention)
            view = -c2w[:3, :3][:, 2]
            view = view / (np.linalg.norm(view) + 1e-8)
            yaw = np.degrees(np.arctan2(view[0], view[2]))  # [-180,180]
            pitch = np.degrees(np.arcsin(np.clip(view[1], -1.0, 1.0)))  # [-90,90]
            yaw_bin = int(np.floor((yaw + 180.0) / 360.0 * max(1, self._yaw_bins)))
            pit_bin = int(np.floor((pitch + 90.0) / 180.0 * max(1, self._pitch_bins)))
            centers[img_id] = center
            views[img_id] = view
            bins[img_id] = (yaw_bin, pit_bin)
        if centers:
            self._centers = centers
            self._view_dirs = views
            self._orient_bins = bins

    def _angle(self, a: int, b: int) -> float:
        va = self._view_dirs.get(a) if self._view_dirs else None
        vb = self._view_dirs.get(b) if self._view_dirs else None
        if va is None or vb is None:
            return 0.0
        cos = float(np.clip(np.dot(va, vb), -1.0, 1.0))
        return float(np.degrees(np.arccos(cos)))

    def _angle_ok(self, seed: int, cand: int) -> Tuple[bool, str, float]:
        if (self._view_dirs is None) or (not self._use_angle_gate):
            return True, "no-geometry", 0.0
        ang = self._angle(seed, cand)
        ok = self._angle_min <= ang <= self._angle_max
        reason = "pass" if ok else f"filtered(angle={ang:.1f})"
        return ok, reason, ang

    # ------------------ candidate discovery & selection ----------------
    def _choose_seed(self) -> int:
        ids = self._image_ids
        n = len(ids)
        if n == 0:
            raise RuntimeError("No images available to choose as seed.")

        def _safe_choice_by_weight(weights: np.ndarray) -> int:
            # ensure non-negative finite weights
            w = np.asarray(weights, dtype=np.float64)
            w[~np.isfinite(w)] = 0.0
            w = np.maximum(w, 0.0)
            s = w.sum()
            if s <= 0.0:
                # fallback: uniform over all ids
                return int(ids[int(self._rng.randint(0, n))])
            # normalize robustly
            w /= s
            # eliminate tiny negative/rounding drift and renormalize
            w = np.clip(w, 0.0, 1.0)
            s2 = w.sum()
            if not np.isfinite(s2) or s2 <= 0.0:
                return int(ids[int(self._rng.randint(0, n))])
            w /= s2
            # guarantee exact sum==1 within fp tolerance
            w[-1] = 1.0 - w[:-1].sum()
            # use numpy choice with p
            return int(ids[int(self._rng.choice(n, p=w))])

        if self.seed_sampling == "round_robin":
            if not hasattr(self, "_seed_rr_order") or not self._seed_rr_order:
                # fallback init
                self._seed_rr_order = list(self._image_ids)
                self._rng.shuffle(self._seed_rr_order)
                self._seed_rr_ptr = 0
            i = self._seed_rr_ptr % len(self._seed_rr_order)
            self._seed_rr_ptr += 1
            return self._seed_rr_order[i]

        if self.seed_sampling == "ray_weighted":
            sizes = np.array([int(self._indices_by_image[i].numel()) for i in ids], dtype=np.float64)
            return _safe_choice_by_weight(sizes)

        if self.seed_sampling == "inverse":
            sizes = np.array([int(self._indices_by_image[i].numel()) for i in ids], dtype=np.float64)
            inv = 1.0 / np.maximum(sizes, 1.0)  # avoid div-by-zero
            return _safe_choice_by_weight(inv)

        # default: uniform random among images
        return int(ids[int(self._rng.randint(0, n))])

    def _firsthop_candidates(self, seed_img: int) -> Tuple[List[int], Dict[int, str]]:
        explained: Dict[int, str] = {}
        if seed_img not in self.neighbors:
            return [], explained
        cand = [c for c in self.neighbors.get(seed_img, []) if c != seed_img]
        out: List[int] = []
        for c in cand:
            ok, why, _ang = self._angle_ok(seed_img, c)
            if ok:
                out.append(c)
            explained[c] = why
        return out, explained

    def _augment_with_near_bound(
        self, seed_img: int, candidates: List[int], explained: Dict[int, str], need: int
    ) -> List[int]:
        if need <= 0 or not self._view_dirs:
            return candidates
        # Add neighbors just outside the angle band, closest-to-band first
        ref = [seed_img] + list(candidates)
        seed_set = list(set(ref))
        seed0 = seed_set[0] if seed_set else None
        mid_angle = 0.5 * (self._angle_min + self._angle_max)

        def min_sep(c: int) -> float:
            return min(self._angle(c, r) for r in ref) if ref else 180.0

        def median_score(c: int) -> float:
            if seed0 is None:
                return abs(min_sep(c) - mid_angle)
            return abs(self._angle(seed0, c) - mid_angle)

        pool = set(candidates)
        while pool and len(candidates) < (len(ref) + need):
            # Try adding the closest-to-band candidate
            extras = []
            for cid in list(pool):
                ok, why, ang = self._angle_ok(seed_img, cid)
                if ok:
                    pool.remove(cid)
                    continue
                # prefer near-bound rejections
                dev = min(abs(ang - self._angle_min), abs(ang - self._angle_max))
                if why.startswith("filtered("):
                    extras.append((float(dev), cid, float(ang)))
            extras.sort(key=lambda x: x[0])
            added = []
            for _, cid, _ in extras:
                pool.remove(cid)
                candidates.append(cid)
                added.append(cid)
                if len(added) >= need:
                    break
            if not added:
                break
        if self.debug and need > 0:
            self._dbg(f"augmented candidates with near-bound rejects: need={need}")
        return candidates

    def _bin_of(self, img_id: int) -> Optional[Tuple[int, int]]:
        return self._orient_bins.get(img_id) if self._orient_bins is not None else None

    def _select_with_bins(
        self,
        seed_set: List[int],
        candidates: List[int],
        limit: int,
        per_bin_cap: int,
        preference: Optional[str] = None,
    ) -> List[int]:
        if limit <= 0 or not candidates:
            return []
        pref = (preference or self.selection_preference).lower()
        if self._view_dirs is None:
            chosen: List[int] = []
            per_bin = defaultdict(int)
            pool = list(candidates)
            self._rng.shuffle(pool)
            for cid in pool:
                b = self._bin_of(cid)
                if b is None:
                    if len(chosen) < limit:
                        chosen.append(cid)
                else:
                    if per_bin[b] < per_bin_cap and len(chosen) < limit:
                        chosen.append(cid)
                        per_bin[b] += 1
                if len(chosen) >= limit:
                    break
            return chosen

        # with geometry: apply preference
        ref = list(seed_set)

        def min_sep(c: int) -> float:
            return min(self._angle(c, r) for r in ref) if ref else 180.0

        def median_score(c: int) -> float:
            if not ref:
                return 90.0
            seed0 = ref[0]
            return abs(self._angle(seed0, c) - 0.5 * (self._angle_min + self._angle_max))

        per_bin = defaultdict(int)
        chosen: List[int] = []
        pool = set(candidates)
        while pool and len(chosen) < limit:
            best_c = None
            best_score = None
            if pref == "farthest":
                # greedy max-min angle (diverse)
                for cid in pool:
                    b = self._bin_of(cid)
                    if b is not None and per_bin[b] >= per_bin_cap:
                        continue
                    score = -min_sep(cid)  # maximize min sep
                    if best_score is None or score < best_score:
                        best_score, best_c = score, cid
            elif pref == "closest":
                # cluster near ref
                for cid in pool:
                    b = self._bin_of(cid)
                    if b is not None and per_bin[b] >= per_bin_cap:
                        continue
                    score = min_sep(cid)  # minimize min sep
                    if best_score is None or score < best_score:
                        best_score, best_c = score, cid
            else:  # 'median' default
                for cid in pool:
                    b = self._bin_of(cid)
                    if b is not None and per_bin[b] >= per_bin_cap:
                        continue
                    score = median_score(cid)
                    if best_score is None or score < best_score:
                        best_score, best_c = score, cid
            if best_c is None:
                break
            chosen.append(best_c)
            per_bin[self._bin_of(best_c)] += 1
            pool.remove(best_c)
        return chosen

    # --------------------- allocation & sampling ----------------------
    def _alloc_counts_by_capacity(
        self,
        image_ids: List[int],
        target_rays: int,
        forbid_indices: Optional[Tensor] = None,
    ) -> List[int]:
        if not image_ids:
            return []
        # Prebucket forbids per image to avoid repeated torch.isin
        forbid_by_img: Dict[int, Tensor] = {}
        if forbid_indices is not None and forbid_indices.numel() > 0:
            f_imgs = self._img_of_ray.index_select(0, forbid_indices).tolist()
            buckets: Dict[int, List[int]] = defaultdict(list)
            for px, img in zip(forbid_indices.tolist(), f_imgs):
                buckets[int(img)].append(int(px))
            for img, pxs in buckets.items():
                forbid_by_img[img] = torch.tensor(pxs, dtype=torch.long, device=self._rays.device)
        caps: List[int] = []
        for img in image_ids:
            pool = self._indices_by_image[img]
            if img in forbid_by_img:
                allowed = int((~torch.isin(pool, forbid_by_img[img])).sum().item())
            else:
                allowed = int(pool.numel())
            caps.append(max(0, allowed))
        total_cap = sum(caps)
        effective = min(target_rays, total_cap)
        m = len(image_ids)
        if m == 0 or effective <= 0:
            return [0] * m
        base = effective // m
        alloc = [min(c, base) for c in caps]
        rem = effective - sum(alloc)
        if rem > 0:
            order = list(range(m))
            self._rng.shuffle(order)
            j = 0
            while rem > 0:
                i = order[j % m]
                if alloc[i] < caps[i]:
                    alloc[i] += 1
                    rem -= 1
                j += 1
        return alloc

    def _gather_side_via_rings(
        self, image_ids: List[int], counts: List[int], split: str, target_total: int
    ) -> Tensor:
        chunks: List[Tensor] = []
        assert self._rings is not None
        contributed: List[int] = []
        for img, take in zip(image_ids, counts):
            if take <= 0:
                continue
            sel = self._rings.take(img, take, split=split)  # parity-first
            if sel.numel() > 0:
                chunks.append(sel)
                contributed.append(img)
        out = torch.cat(chunks, dim=0) if chunks else torch.empty(0, dtype=torch.long)
        deficit = target_total - int(out.numel())
        if deficit <= 0:
            return out
        # Phase 2: top-up WITHOUT replacement from full per-image pool (both parities), still pixel-disjoint
        added: List[Tensor] = []
        for img, take in zip(image_ids, counts):
            if deficit <= 0 or take <= 0:
                continue
            pool = self._indices_by_image[img]
            if pool.numel() == 0:
                continue
            forbid = out if out.numel() > 0 else None
            if forbid is not None:
                mask = ~torch.isin(pool, forbid)
                pool = pool[mask]
            if pool.numel() == 0:
                continue
            k = min(deficit, pool.numel())
            idx = torch.randperm(pool.numel(), generator=self._gen, device=pool.device)[:k] if hasattr(self, "_gen") else torch.randperm(pool.numel(), device=pool.device)[:k]
            added.append(pool.index_select(0, idx))
            deficit -= k
            contributed.append(img)
        if added:
            out = torch.cat([out] + added, dim=0)
        # Phase 3: last resort WITH REPLACEMENT from already-contributing images
        if deficit > 0 and contributed:
            repl_pools = [self._indices_by_image[i] for i in set(contributed) if self._indices_by_image[i].numel() > 0]
            if repl_pools:
                concat = torch.cat(repl_pools, dim=0)
                idx = torch.randint(low=0, high=concat.numel(), size=(deficit,), device=concat.device, generator=self._gen) if hasattr(self, "_gen") else torch.randint(0, concat.numel(), (deficit,), device=concat.device)
                out = torch.cat([out, concat.index_select(0, idx)], dim=0)
                warnings.warn(f"[{split}] capacity shortfall: topped up {deficit} rays WITH REPLACEMENT from same images.", RuntimeWarning)
        return out

    def _sample_from_image_rand(
        self, image_id: int, n: int, forbid: Optional[Tensor], strict: bool
    ) -> Tensor:
        pool = self._indices_by_image[image_id]
        if forbid is not None and forbid.numel() > 0:
            mask = ~torch.isin(pool, forbid)
            pool = pool[mask]
        if pool.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        if pool.numel() >= n:
            perm = torch.randperm(pool.numel(), generator=self._gen, device=pool.device)[:n] if hasattr(self, '_gen') else torch.randperm(pool.numel(), device=pool.device)[:n]
            return pool.index_select(0, perm)
        if strict:
            return pool
        idx = torch.randint(low=0, high=pool.numel(), size=(n,), dtype=torch.long, generator=self._gen, device=pool.device) if hasattr(self, '_gen') else torch.randint(low=0, high=pool.numel(), size=(n,), dtype=torch.long, device=pool.device)
        return pool.index_select(0, idx)

    def _gather_side_rand(
        self,
        image_ids: List[int],
        counts: List[int],
        forbid: Optional[Tensor],
        strict: bool,
        target_total: int,
        split="support"
    ) -> Tensor:
        chunks: List[Tensor] = []
        for img, take in zip(image_ids, counts):
            if take <= 0:
                continue
            sel = self._sample_from_image_rand(img, take, forbid=forbid, strict=strict)
            if sel.numel() > 0:
                chunks.append(sel)
        out = torch.cat(chunks, dim=0) if chunks else torch.empty(0, dtype=torch.long)
        deficit = target_total - int(out.numel())
        if deficit > 0:
            top_counts = self._alloc_counts_by_capacity(image_ids, deficit, forbid)
            top_chunks = []
            for img, add in zip(image_ids, top_counts):
                if add <= 0:
                    continue
                sel = self._sample_from_image_rand(
                    img, add, forbid=forbid, strict=strict
                )
                if sel.numel() > 0:
                    top_chunks.append(sel)
            if top_chunks:
                out = torch.cat([out] + top_chunks, dim=0)
        if out.numel() != target_total:
            deficit = int(target_total - out.numel())
            # last resort replacement from same images
            if deficit > 0:
                pools = [self._indices_by_image[i] for i in image_ids if self._indices_by_image[i].numel() > 0]
                if pools:
                    concat = torch.cat(pools, dim=0)
                    idx = torch.randint(0, concat.numel(), (deficit,), device=concat.device, generator=self._gen) if hasattr(self, '_gen') else torch.randint(0, concat.numel(), (deficit,), device=concat.device)
                    rep = concat.index_select(0, idx)
                    out = torch.cat([out, rep], dim=0)
                    warnings.warn(f"[{split}] capacity shortfall: topped up {deficit} rays WITH REPLACEMENT.", RuntimeWarning)
        return out

    # ------------------------- episode building -----------------------
    def _pack(self, idx: Tensor) -> Dict[str, Tensor]:
        idx = idx.long().view(-1)
        return {
            "rays": self._rays.index_select(0, idx),
            "rgbs": self._rgbs.index_select(0, idx),
            "indices": idx,
        }

    def _build_image_sets(self) -> Tuple[int, List[int], List[int], List[int]]:
        tries = 0
        mcs = int(getattr(self, "min_coverage_support", 0) or 0)
        mcq = int(getattr(self, "min_coverage_query",   0) or 0)

        while True:
            tries += 1
            seed_img = self._choose_seed()

            # 1-hop candidates (may be empty)
            cand_1hop, explained = self._firsthop_candidates(seed_img)

            # If min coverage requested for SUPPORT, try near-bound augmentation
            if mcs > 0 and len(cand_1hop) < mcs:
                cand_1hop = self._augment_with_near_bound(
                    seed_img, cand_1hop, explained, mcs - len(cand_1hop)
                )

            # Absolute fallback: if still no candidates, widen to "all others"
            if not cand_1hop:
                all_others = [i for i in self._image_ids if i != seed_img]
                if all_others:
                    warnings.warn("[policy] no 1-hop candidates; widened to all images (bins will still cap).",
                                RuntimeWarning)
                    cand_1hop = all_others

            # -------------------- SUPPORT --------------------
            support_imgs: List[int] = [seed_img]
            more_sup = self._select_with_bins(
                seed_set=support_imgs,
                candidates=cand_1hop,
                limit=max(0, self.max_support_images - 1),
                per_bin_cap=self._max_per_orient_bin,
                preference=self.selection_preference_support,
            )
            support_imgs.extend(more_sup)

            # -------------------- QUERY-A --------------------
            policy = self.policy
            if policy == "same_image_only":
                query_imgs = support_imgs.copy()
            else:
                disjoint = [c for c in cand_1hop if c not in set(support_imgs)]
                if policy == "strict_image_disjoint":
                    query_pool = disjoint
                else:
                    # prefer disjoint; if empty, allow same images (pixel-level disjoint later)
                    query_pool = disjoint if disjoint else support_imgs.copy()
                    if not disjoint:
                        warnings.warn("[policy] no disjoint query images; allowing same images (pixel-level disjoint).",
                                    RuntimeWarning)

                ref_for_query = list(set(support_imgs))
                query_imgs = self._select_with_bins(
                    seed_set=ref_for_query,
                    candidates=query_pool,
                    limit=self.max_query_images,
                    per_bin_cap=self._max_per_orient_bin,
                    preference=self.selection_preference_query,
                )

                # If a minimum coverage was requested for QUERY and we’re short (and policy allows),
                # try topping up from support (still pixel-level disjoint at ray level).
                if mcq > 0 and len(query_imgs) < mcq and policy != "strict_image_disjoint":
                    topup_src = [iid for iid in support_imgs if iid not in query_imgs]
                    needed = mcq - len(query_imgs)
                    if needed > 0 and topup_src:
                        warnings.warn("[policy] query disjoint coverage too low → adding images from support "
                                    "(pixel-level disjoint).", RuntimeWarning)
                        topup = self._select_with_bins(
                            seed_set=list(set(support_imgs + query_imgs)),
                            candidates=topup_src,
                            limit=needed,
                            per_bin_cap=self._max_per_orient_bin,
                            preference=self.selection_preference_query,
                        )
                        query_imgs.extend(topup)

                # Last soft guard: ensure at least one query image unless strict disjoint forbids it
                if not query_imgs:
                    if policy == "strict_image_disjoint":
                        # keep empty; episode will retry another seed
                        pass
                    else:
                        # allow same image as seed; ray-level disjointness handled later
                        warnings.warn("[policy] empty query pool; using seed image for query (pixel-level disjoint).",
                                    RuntimeWarning)
                        query_imgs = [seed_img]

            # -------------------- QUERY-HOLDOUT --------------------
            query_holdout: List[int] = []
            if getattr(self, "query_holdout_images", 0) > 0:
                pool = [c for c in cand_1hop if c not in set(support_imgs + query_imgs)]
                if pool:
                    query_holdout = self._select_with_bins(
                        seed_set=list(set(support_imgs + query_imgs)),
                        candidates=pool,
                        limit=self.query_holdout_images,
                        per_bin_cap=self._max_per_orient_bin,
                        preference=self.selection_preference,  # ok to keep shared pref here
                    )

            # -------------------- Sanity & exit --------------------
            # If query_rays == 0, we can accept empty query; otherwise require at least one image
            query_required = (self.query_rays or 0) > 0
            if support_imgs and ((query_imgs or query_holdout) or not query_required):
                return seed_img, support_imgs, query_imgs, query_holdout

            if tries >= self.max_seed_tries:
                raise RuntimeError(f"Failed to build episode image sets after {tries} tries.")

    # ----------------------------- getitem ----------------------------
    def __getitem__(self, index: int) -> Dict[str, Dict[str, Tensor]]:
        # Per-episode generator seeded from worker + epoch + index
        if not hasattr(self, '_episode_serial'):
            self._episode_serial = 0
        self._episode_serial += 1
        base = torch.initial_seed()
        gseed = int((base + self._epoch * 0x9E3779B1 + index + self._episode_serial) & 0x7FFFFFFF)
        self._gen = torch.Generator(device=self._rays.device)
        self._gen.manual_seed(gseed)
        np.random.seed(gseed & 0xFFFFFFFF)
        random_seed = gseed & 0xFFFFFFFF
        try:
            import random as _py_random
            _py_random.seed(random_seed)
        except Exception:
            pass

        # Choose images either from precomputed index or build on the fly
        if (
            self._precompute
            and self._episode_index is not None
            and len(self._episode_index) > 0
        ):
            idx = index % len(self._episode_index)
            seed_img, support_imgs, query_imgs, query_holdout_imgs = (
                self._episode_index.get(idx)
            )
        else:
            seed_img, support_imgs, query_imgs, query_holdout_imgs = (
                self._build_image_sets()
            )

        # ----- Allocate rays (capacity-aware) -----
        sup_counts = self._alloc_counts_by_capacity(
            support_imgs, self.support_rays, forbid_indices=None
        )
        imgs_overlap = bool(set(support_imgs) & set(query_imgs))

        # Gather SUPPORT
        if self._use_ring_buffers:
            support_idx = self._gather_side_via_rings(
                support_imgs,
                sup_counts,
                split="support",
                target_total=self.support_rays,
            )
        else:
            support_idx = self._gather_side_rand(
                support_imgs,
                sup_counts,
                forbid=None,
                strict=False,
                target_total=self.support_rays,
                split= 'support'
            )

        # If user requests Query-B only (no Query-A), ensure query uses only held-out images
        query_idx = torch.empty(0, dtype=torch.long)
        if self.max_query_images == 0:
            # When Query-A is disabled, we allocate the full query budget to Query-B (handled below)
            query_imgs = []
        else:
            # QUERY-A: pixel-disjoint from SUPPORT if images overlap
            qry_forbid = (
                support_idx if (not self._use_ring_buffers and imgs_overlap) else None
            )
            qry_counts = self._alloc_counts_by_capacity(
                query_imgs, self.query_rays, forbid_indices=qry_forbid
            )
            if self._use_ring_buffers:
                query_idx = self._gather_side_via_rings(
                    query_imgs,
                    qry_counts,
                    split="query",
                    target_total=self.query_rays,
                )
            else:
                # If policy is "strict_image_disjoint" we disallow replacement top-ups across support images;
                # otherwise we allow pixel-level disjointness and, as last resort, replacement.
                query_idx = self._gather_side_rand(
                    query_imgs,
                    qry_counts,
                    forbid=qry_forbid,
                    strict=(self.policy == "strict_image_disjoint"),
                    target_total=self.query_rays,
                    split='quey'
                )

        # Optional: QUERY-B (held-out views) — also used when Query-A is disabled
        out_query_holdout = None
        if (
            self.query_holdout_images > 0 and len(query_holdout_imgs) > 0
        ) or self.max_query_images == 0:
            # Determine target rays for Query-B: if A disabled, full budget; else (by default) half-budget
            qh_target = (
                self.query_rays
                if self.max_query_images == 0
                else max(0, min(self.query_rays, self.query_rays // 2))
            )

            # Capacity check: if strictly disjoint images cannot support qh_target, either relax or warn+fail
            holdout_pool = query_holdout_imgs.copy()
            if self.policy == "strict_image_disjoint":
                cap = sum(int(self._indices_by_image[i].numel()) for i in holdout_pool)
                if cap < qh_target:
                    raise RuntimeError(
                        f"Insufficient rays in held-out images to meet Query-B target ({cap} < {qh_target}) and strict_image_disjoint is set."
                    )
                # Relax: allow topping up from SUPPORT images, but keep pixel-level disjointness
                extra_src = [i for i in support_imgs if i not in holdout_pool]
                holdout_pool.extend(extra_src)

            # Allocate and gather
            qh_counts = self._alloc_counts_by_capacity(
                holdout_pool,
                qh_target,
                forbid_indices=None if self._use_ring_buffers else support_idx,
            )
            if self._use_ring_buffers:
                qh_idx = self._gather_side_via_rings(
                    holdout_pool, qh_counts, split="query", target_total=qh_target
                )
            else:
                qh_idx = self._gather_side_rand(
                    holdout_pool,
                    qh_counts,
                    forbid=support_idx,
                    strict=True,
                    target_total=qh_target,
                    split='query'
                )
            out_query_holdout = self._pack(qh_idx)

        # --------- Pack & return ---------
        if self.debug:
            self._dbg(
                f"seed={seed_img} support={support_imgs} query={query_imgs}"
                + (f" qholdout={query_holdout_imgs}" if self.query_holdout_images > 0 else "")
            )
        out = {
            "support": self._pack(support_idx),
            "query": self._pack(query_idx) if query_idx.numel() > 0 else None,
        }
        if out_query_holdout is not None:
            out["query_holdout"] = out_query_holdout
        return out


# ----------------------------------------------------------------------
# Neighbor building from SfM points
# ----------------------------------------------------------------------
def _build_neighbors_from_points3D(
    points3D: Iterable[Any], allowed_ids: set[int]
) -> Dict[int, List[int]]:
    # points3D: objects with .image_ids iterable
    pair_count: Dict[Tuple[int, int], int] = defaultdict(int)
    for p in points3D:
        ids = [int(i) for i in getattr(p, "image_ids", []) if int(i) in allowed_ids]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if a > b:
                    a, b = b, a
                pair_count[(a, b)] += 1
    neighbors = {i: [] for i in allowed_ids}
    for (a, b), _cnt in pair_count.items():
        if a in neighbors and b in neighbors:
            neighbors[a].append(b)
            neighbors[b].append(a)
    for k in list(neighbors.keys()):
        neighbors[k] = list(sorted(set(neighbors[k])))
    return neighbors
