"""ANN predictor of the GUI: the seed ensembles of the paper in plain numpy (no scikit-learn at run time).

Since author decision D56 the ANN of each target is the mean of five networks that differ only in their random
initialization (random_state 42, 1, 2, 3, 4; 40 networks for the eight targets). The arrays in
assets/models/ann_weights.npz are the fitted StandardScaler and MLPRegressor of every member
(gui/train_deployment_models.py; seeds, gate tier and provenance in assets/models/model_metadata.json). The forward pass
repeats scikit-learn's operations in the same order (x - mean, / scale, then per layer x @ W + b with tanh or relu on
the hidden layers and the identity on the output), so it returns the same numbers as the fitted pipelines.
Inverse transforms: V = exp(z) - 1 for the forces, delta = exp(z) H_tot for the displacements, T1 = z. The prediction of
a target is the arithmetic mean of its five members in physical units. As in the paper, the base shear at the target
displacement of every member is capped at the V_max of the member with the same seed before the mean, so the mean
never exceeds the predicted V_max. Intervals: [y (1 - q), y (1 + q)] with the split-conformal relative half-width q of
the ANN at 90 % nominal coverage (paper Table 7).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .frame import INPUT_ORDER, FrameInput, derived, feature_vector

MODEL_DIR = Path(__file__).resolve().parents[1] / 'assets' / 'models'
TARGETS = ['Vy', 'dy', 'Vmax', 'dmax', 'V_at_dt', 'dt', 'du', 'T1']
HAZARDS = ['DD-2', 'DD-1']


class ANNBundle:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.meta = json.loads((Path(model_dir) / 'model_metadata.json').read_text(encoding='utf-8'))
        assert self.meta['input_order'] == INPUT_ORDER, 'input order of the models differs from the GUI'
        arr = np.load(Path(model_dir) / 'ann_weights.npz')
        self.seeds = list(self.meta['ensemble']['seeds'])
        self.layers = {}                                  # target -> list of members in seed order
        for t in TARGETS:
            n = self.meta['targets_meta'][t]['n_layers']
            act = self.meta['targets_meta'][t]['activation']
            self.layers[t] = [dict(mean=arr[f'{t}__s{s}__mean'], scale=arr[f'{t}__s{s}__scale'],
                                   W=[arr[f'{t}__s{s}__W{k}'] for k in range(n)],
                                   b=[arr[f'{t}__s{s}__b{k}'] for k in range(n)], act=act) for s in self.seeds]
        self.q = {t: float(self.meta['conformal']['q'][t]) for t in TARGETS}
        self.transform = {t: self.meta['targets_meta'][t]['transform'] for t in TARGETS}
        self.tier = self.meta['gate']['tier']

    @staticmethod
    def _forward(L: dict, X: np.ndarray) -> np.ndarray:
        a = np.array(X, dtype=np.float64, copy=True)
        a -= L['mean']
        a /= L['scale']
        n = len(L['W'])
        for i in range(n):
            a = a @ L['W'][i]
            a += L['b'][i]
            if i != n - 1:
                if L['act'] == 'tanh':
                    np.tanh(a, out=a)
                elif L['act'] == 'relu':
                    np.maximum(a, 0, out=a)
                else:
                    raise ValueError(L['act'])
        return a.ravel()

    def _inverse(self, t: str, z: np.ndarray, H: np.ndarray) -> np.ndarray:
        tt = self.transform[t]
        if tt == 'log1p':
            return np.expm1(z)
        if tt == 'lognormH':
            return np.exp(z) * H
        return z

    def predict_rows(self, X: np.ndarray) -> tuple[dict, np.ndarray]:
        """Predictions in physical units for the rows of X (21 inputs in the paper order). Returns (pred, capped): each
        target is the mean of its five members; the base shear at the target displacement of every member is capped at
        the V_max of the member with the same seed before the mean; capped marks the rows where the cap lowered at least
        one member."""
        X = np.asarray(X, dtype=np.float64)
        H = X[:, INPUT_ORDER.index('Htot')]
        members = {t: np.vstack([self._inverse(t, self._forward(L, X), H) for L in self.layers[t]]) for t in TARGETS}
        capped = (members['V_at_dt'] > members['Vmax']).any(axis=0)
        members['V_at_dt'] = np.minimum(members['V_at_dt'], members['Vmax'])
        pred = {t: members[t].mean(axis=0) for t in TARGETS}
        return pred, capped


@dataclass
class FramePrediction:
    frame: FrameInput
    derived: dict
    values: dict = field(default_factory=dict)      # hazard -> target -> value
    intervals: dict = field(default_factory=dict)   # hazard -> target -> (lo, hi)
    capped: dict = field(default_factory=dict)      # hazard -> bool
    warnings: list = field(default_factory=list)


def predict_frame(bundle: ANNBundle, f: FrameInput) -> FramePrediction:
    der = derived(f)
    X = np.array([feature_vector(f, h, der) for h in HAZARDS])
    pred, capped = bundle.predict_rows(X)
    out = FramePrediction(frame=f, derived=der)
    for i, h in enumerate(HAZARDS):
        out.values[h] = {t: float(pred[t][i]) for t in TARGETS}
        out.intervals[h] = {t: (out.values[h][t] * (1 - bundle.q[t]), out.values[h][t] * (1 + bundle.q[t]))
                            for t in TARGETS}
        out.capped[h] = bool(capped[i])
        if out.values[h]['Vy'] > out.values[h]['Vmax']:
            out.warnings.append(f'{h}: the predicted yield base shear exceeds the predicted peak base shear.')
        if out.capped[h]:
            out.warnings.append(f'{h}: the base shear at the target displacement was capped at the peak base shear in '
                                'at least one of the five networks.')
    return out
