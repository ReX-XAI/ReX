from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch as tt

ROOT = Path(__file__).resolve().parent
EXTERNAL_REX_DIR = ROOT.parent
sys.path.insert(0, str(EXTERNAL_REX_DIR))

from rex_xai.input.input_data import Data  # noqa: E402
from rex_xai.input.config import CausalArgs, Strategy, Queue, Distribution  # noqa: E402
from rex_xai.responsibility.prediction import Prediction  # noqa: E402
from rex_xai.explanation.rex import predict_target, calculate_responsibility  # noqa: E402
from rex_xai.explanation.explanation import Explanation  # noqa: E402

# mode="tabular": a flat feature vector, not an image.


def _make_tabular_prediction_func(predict, threshold: float = 0.5):
    def prediction_func(x, target=None):
        arr = x.detach().cpu().numpy() if isinstance(x, tt.Tensor) else np.asarray(x)
        arr = arr.reshape(-1, arr.shape[-1]).astype(np.float32)
        probs = np.asarray(predict(arr)).reshape(-1)
        preds = []
        for p in probs:
            cls = 1 if p >= threshold else 0
            conf = float(p if cls == 1 else 1.0 - p)
            pr = Prediction(cls, conf)
            if target is not None:
                pr.target = target.classification
                pr.target_confidence = float(p if target.classification == 1 else 1.0 - p)
            preds.append(pr)
        return preds
    return prediction_func


def _make_tabular_data(x: np.ndarray, baseline: np.ndarray, device: str = "cpu") -> Data:
    n = len(x)
    data = Data(x.astype(np.float32), model_shape=(1, n), device=device, mode="tabular", process=True)
    baseline_t = tt.from_numpy(baseline.astype(np.float32)).to(device)
    data.mask_value = lambda mask, d: tt.where(mask, d, baseline_t)
    return data


def _make_tabular_args(iters=32, tree_depth=10, min_box_size=1, queue_len=6, weighted=True,
                        concentrate=True, confidence_filter=0.9, responsibility_style="multiplicative",
                        seed=42, min_conf=0.9, chunk_size=1, batch_size=32) -> CausalArgs:
    args = CausalArgs()
    args.iters = iters
    args.tree_depth = tree_depth
    args.min_box_size = min_box_size
    args.queue_len = queue_len
    args.queue_style = Queue.Intersection
    args.weighted = weighted
    args.concentrate = concentrate
    args.confidence_filter = confidence_filter
    args.responsibility_style = responsibility_style
    args.distribution = Distribution.Uniform
    args.distribution_args = None
    args.seed = seed
    args.use_bounding_box = False
    args.strategy = Strategy.Global
    args.minimum_confidence_threshold = min_conf
    args.chunk_size = chunk_size
    args.batch_size = batch_size
    args.progress_bar = False
    args.negative_responsibility = False
    return args


def _explain_tabular_raw(predict, x: np.ndarray, baseline: np.ndarray, args: CausalArgs):
    n = len(x)
    prediction_func = _make_tabular_prediction_func(predict)
    data = _make_tabular_data(x, baseline)
    data.target = predict_target(data, args, prediction_func)
    maps, run_stats = calculate_responsibility(data, args, prediction_func)

    resp_map = maps.get(data.target.classification)
    resp = np.asarray(resp_map[0]) if resp_map is not None else np.zeros(n, dtype=np.float32)

    explanation = Explanation(maps, prediction_func, data, args, run_stats)
    explanation.extract()

    if explanation.sufficiency_mask is not None:
        sufficient_mask = explanation.sufficiency_mask.detach().cpu().numpy().reshape(-1)[:n]
    else:
        sufficient_mask = np.zeros(n, dtype=bool)

    return resp, sufficient_mask, explanation.sufficiency_confidence


def detect_onehot_groups(feat_names: list[str], X: np.ndarray) -> dict[str, list[int]]:
    by_prefix: dict[str, list[int]] = {}
    for j, name in enumerate(feat_names):
        if "_" not in name:
            continue
        by_prefix.setdefault(name.rsplit("_", 1)[0], []).append(j)

    groups = {}
    for prefix, cols in by_prefix.items():
        if len(cols) < 2:
            continue
        sub = X[:, cols]
        if not np.isfinite(sub).all():
            continue
        is_binary = np.all((sub == 0) | (sub == 1))
        row_sums = sub.sum(axis=1)
        sums_valid = np.all(np.isclose(row_sums, 1.0) | np.isclose(row_sums, 0.0))
        if is_binary and sums_valid:
            groups[prefix] = cols
    return groups


def build_units(n_features: int, groups: dict[str, list[int]]) -> list[list[int]]:
    grouped = set(c for cols in groups.values() for c in cols)
    units = [[j] for j in range(n_features) if j not in grouped]
    units.extend(groups.values())
    return units


def _explain_tabular_grouped(predict, x, baseline, units, args):
    n_units = len(units)
    unit_x = np.ones(n_units, dtype=np.float32)
    unit_baseline = np.zeros(n_units, dtype=np.float32)

    def unit_predict(rows):
        rows = np.atleast_2d(rows)
        full = np.tile(baseline, (rows.shape[0], 1)).astype(np.float32)
        for j, cols in enumerate(units):
            reveal = rows[:, j] > 0.5
            for r in np.where(reveal)[0]:
                full[r, cols] = x[cols]
        return predict(full)

    unit_resp, unit_suff, conf = _explain_tabular_raw(unit_predict, unit_x, unit_baseline, args)

    n_features = sum(len(u) for u in units)
    resp = np.zeros(n_features, dtype=np.float32)
    sufficient_mask = np.zeros(n_features, dtype=bool)
    for j, cols in enumerate(units):
        resp[cols] = unit_resp[j]
        if unit_suff[j]:
            sufficient_mask[cols] = True
    return resp, sufficient_mask, conf


def mean_baseline(X: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.nanmean(X, axis=0)).astype(np.float32)


def decision_boundary_baseline(predict, X_train: np.ndarray, groups: dict[str, list[int]],
                                tol: float = 1e-3, max_iters: int = 60) -> np.ndarray:
    mean_bl = mean_baseline(X_train)
    preds = predict(X_train)
    mean_bl_p = float(predict(mean_bl[None, :])[0])
    opposite_mask = (preds < 0.5) if mean_bl_p >= 0.5 else (preds >= 0.5)
    other_mean = np.nan_to_num(np.nanmean(X_train[opposite_mask], axis=0)).astype(np.float32)

    onehot_cols = set(c for cols in groups.values() for c in cols)
    non_oh = np.array([j for j in range(len(mean_bl)) if j not in onehot_cols])

    def at(a):
        row = mean_bl.copy()
        if len(non_oh):
            row[non_oh] = (1 - a) * mean_bl[non_oh] + a * other_mean[non_oh]
        for cols in groups.values():
            row[cols] = 1.0 / len(cols)
        return row.astype(np.float32)

    def p_at(a):
        return float(predict(at(a)[None, :])[0])

    below = p_at(0.0) < 0.5
    lo, hi = 0.0, 1.0
    while (p_at(hi) < 0.5) == below and hi <= 8.0:
        lo, hi = hi, hi * 2
    bl = at(0.0)
    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        bl = at(mid)
        p = p_at(mid)
        if abs(p - 0.5) < tol:
            break
        if (p < 0.5) == below:
            lo = mid
        else:
            hi = mid
    return bl.astype(np.float32)


def effective_min_conf(min_conf: float, orig_conf: float, abs_ceiling: float | None) -> float:
    if abs_ceiling is None:
        return min_conf
    return min(min_conf, abs_ceiling / orig_conf)


def necessity_prune(predict, x: np.ndarray, baseline: np.ndarray, suff_idx, resp: np.ndarray,
                     target_conf_needed: float, cls: int, units: list[list[int]],
                     tolerance: float = 0.03) -> list[int]:
    suff_idx = set(suff_idx)
    full_idx = list(suff_idx)
    row_full = baseline.copy()
    if full_idx:
        row_full[full_idx] = x[full_idx]
    p_full = float(predict(row_full[None, :])[0])
    achieved_conf = p_full if cls == 1 else 1.0 - p_full
    target_conf_needed = max(target_conf_needed, achieved_conf - tolerance)

    active_units = [cols for cols in units if any(j in suff_idx for j in cols)]

    def unit_resp(cols):
        return max(resp[j] for j in cols)

    order = sorted(active_units, key=unit_resp)
    changed = True
    while changed:
        changed = False
        for cols in list(order):
            if cols not in active_units:
                continue
            trial_units = [u for u in active_units if u != cols]
            trial_idx = [j for u in trial_units for j in u]
            row = baseline.copy()
            if trial_idx:
                row[trial_idx] = x[trial_idx]
            p = float(predict(row[None, :])[0])
            conf = p if cls == 1 else 1.0 - p
            if conf >= target_conf_needed:
                active_units = trial_units
                changed = True

    if not active_units:
        all_units = [cols for cols in units if any(j in suff_idx for j in cols)] or units
        active_units = [max(all_units, key=unit_resp)]

    return [j for u in active_units for j in u]


def _display_line(j: int, x: np.ndarray, baseline: np.ndarray, resp: np.ndarray,
                   feat_names: list[str], name_to_idx: dict[str, int],
                   measured_suffix: str) -> str:
    name = feat_names[j]
    val = x[j]
    if np.isnan(val):
        measured_idx = name_to_idx.get(f"{name}{measured_suffix}")
        if measured_idx is not None and x[measured_idx] == 0:
            return (f"  {name:35s} NOT MEASURED for this row "
                    f"(missingness itself is the signal)  resp={resp[j]:.3f}")
        return f"  {name:35s} value unavailable (NaN, no companion flag found)  resp={resp[j]:.3f}"
    return f"  {name:35s} value={val:.3f}  baseline={baseline[j]:.3f}  resp={resp[j]:.3f}"


@dataclass
class TReXResult:
    feature_indices: list[int]
    feature_names: list[str]
    responsibility: np.ndarray
    prediction: float
    predicted_class: int
    sufficiency_confidence: float | None
    baseline: np.ndarray
    x: np.ndarray
    _measured_suffix: str = field(default="_measured", repr=False)

    def report(self) -> str:
        name_to_idx = {n: j for j, n in enumerate(self.feature_names)}
        lines = [f"prediction: P(class=1)={self.prediction:.3f}  predicted_class={self.predicted_class}  "
                 f"k={len(self.feature_indices)}"]
        for j in sorted(self.feature_indices, key=lambda j: -self.responsibility[j]):
            lines.append(_display_line(j, self.x, self.baseline, self.responsibility,
                                        self.feature_names, name_to_idx, self._measured_suffix))
        return "\n".join(lines)


class TReX:

    def __init__(self, X_train: np.ndarray, feat_names: list[str], predict,
                 measured_suffix: str = "_measured"):
        self.feat_names = feat_names
        self.predict = predict
        self.measured_suffix = measured_suffix
        self.groups = detect_onehot_groups(feat_names, X_train)
        self.units = build_units(len(feat_names), self.groups)
        self.baseline = decision_boundary_baseline(predict, X_train, self.groups)

    def explain(self, x: np.ndarray, iters: int = 32, min_conf: float = 0.95,
                abs_ceiling: float | None = 0.75, prune_tolerance: float = 0.01,
                seed: int = 1, resp_floor: float = 0.0) -> TReXResult:
        p = float(self.predict(x[None, :])[0])
        cls = 1 if p >= 0.5 else 0
        orig_conf = p if cls == 1 else 1.0 - p
        mc = effective_min_conf(min_conf, orig_conf, abs_ceiling)

        args = _make_tabular_args(iters=iters, min_conf=mc, seed=seed)
        resp, suff_mask, conf = _explain_tabular_grouped(self.predict, x, self.baseline, self.units, args)
        suff_idx = set(np.where(suff_mask)[0])
        if suff_idx:
            cur_max = max(resp[j] for j in suff_idx)
            floor = resp_floor * cur_max
            if floor > 0:
                for cols in self.units:
                    if any(j in suff_idx for j in cols):
                        continue
                    if max(resp[j] for j in cols) >= floor:
                        suff_idx |= set(cols)
        pruned_idx = necessity_prune(self.predict, x, self.baseline, suff_idx, resp,
                                      mc * orig_conf, cls, self.units, tolerance=prune_tolerance)

        return TReXResult(feature_indices=pruned_idx, feature_names=self.feat_names,
                           responsibility=resp, prediction=p, predicted_class=cls,
                           sufficiency_confidence=conf, baseline=self.baseline, x=x,
                           _measured_suffix=self.measured_suffix)

    def explain_stable(self, x: np.ndarray, seeds: list[int] = (1, 2, 3, 4, 5), **kwargs) -> TReXResult:
        union_idx: set[int] = set()
        last_res = None
        for s in seeds:
            res = self.explain(x, seed=s, **kwargs)
            union_idx |= set(res.feature_indices)
            last_res = res
        return TReXResult(feature_indices=sorted(union_idx), feature_names=last_res.feature_names,
                           responsibility=last_res.responsibility, prediction=last_res.prediction,
                           predicted_class=last_res.predicted_class,
                           sufficiency_confidence=last_res.sufficiency_confidence,
                           baseline=last_res.baseline, x=last_res.x,
                           _measured_suffix=last_res._measured_suffix)
