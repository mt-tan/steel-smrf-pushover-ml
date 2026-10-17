"""Validity checks of the GUI: the closed-form screens of the dataset generator and the Table 2 range check.

Closed-form screens = the sampler tier of generator v0.19 (generator_v019/filters.py, _evaluate_all_pairs_v019, with the
arguments of generator_v019/sampling.py), evaluated for the chosen column-beam pair:
  compactness (high ductility, TBEC 2018 Table 9.3) of the column (Ca from the tributary 1.2G + 1.6Q load of an
    interior base column) and of the beam;
  strong column-weak beam at a first-floor interior joint (TBEC 2018 Eq. 9.7 to 9.9b: ratio > 1.0, 1.1 Ry on the beam
    side, Muv from the hinge shear under 1.2G + 0.5Q, Puc from the tributary load of combination (4.11));
  column axial-load ratio P/Pc <= 0.50 (Annex 5C; P = interior base column under G(1 + 0.2 SDS) + 0.30Q, Pc = expected
    compressive strength about the weaker axis, TBEC 2018 Table 5.1 Fye, K = 1, L = h1);
  beam depth below column depth;
  beam gravity strength (TSSDC 2018 5.3.1: 1.4G and 1.2G + 1.6Q with the beam self weight);
  geometric buildability (clear lengths above 0.20 m);
  total height H_tot <= 28 m (single-mode pushover applicability, TBEC 2018).
The strength-and-drift design of the dataset (elastic frame analysis with second-order amplification, every joint, the
DD-2 gravity state of the nonlinear model) and MP1 >= 0.70 need a structural analysis and are not checked here.
"""
from __future__ import annotations

import math

from .frame import CATALOGUE, E_STEEL, FYE_T51, G_ACC, GRADES, LIVE_LOAD_N, RHO_STEEL, FrameInput, derived

MIN_CLEAR_LENGTH = 0.20      # m
H_MAX = 28.0                 # m
CPR, BEAM_FACTOR, LH = 1.2, 1.1, 0.0
PHI_B, SPAN_DIVISOR = 0.90, 12.0
NOTE = ('Predictions are validated only for frames inside the ranges that pass every screen; the strength-and-drift '
        'design, the gravity-state checks and MP1 >= 0.70 require a structural analysis and are not checked here.')

# SDS of the site at DD-2 per soil class: SS = 1.053 times Fs of TBEC 2018 Table 2.1 (linear interpolation), as
# generator_v019/tbdy_seismic.py computes it
_SS_DD2 = 1.053
_SS_ANCH = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
_FS = {'ZA': [0.8] * 6, 'ZB': [0.9] * 6, 'ZC': [1.3, 1.3, 1.2, 1.2, 1.2, 1.2], 'ZD': [1.6, 1.4, 1.2, 1.1, 1.0, 1.0],
       'ZE': [2.4, 1.7, 1.3, 1.1, 0.9, 0.8]}


def _interp(x, xs, ys):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for k in range(1, len(xs)):
        if x <= xs[k]:
            t = (x - xs[k - 1]) / (xs[k] - xs[k - 1])
            return ys[k - 1] + t * (ys[k] - ys[k - 1])
    return ys[-1]


def sds_dd2(soil: str) -> float:
    return _SS_DD2 * _interp(_SS_DD2, _SS_ANCH, _FS[soil])


def _fcr(Fy, slender):
    """TSSDC 2018 8.2: Fe = pi^2 E/(Lc/i)^2; Fy/Fe <= 2.25: 0.658^(Fy/Fe) Fy, else 0.877 Fe."""
    Fe = math.pi ** 2 * E_STEEL / slender ** 2
    return (0.658 ** (Fy / Fe)) * Fy if Fy / Fe <= 2.25 else 0.877 * Fe


def tributary_axial(f: FrameInput) -> float:
    """Interior base-column axial estimate under 1.2G + 1.6Q [kN] (Ca of the compactness screen)."""
    w_u = 1.2 * f.G_line + 1.6 * f.Q_line
    trib = f.L_bay if f.n_bay >= 2 else f.L_bay / 2.0
    return w_u * trib * f.n_story


def screens(f: FrameInput) -> list:
    """List of dicts: name, criterion, value, limit, passed."""
    c, b = CATALOGUE[f.col_profile], CATALOGUE[f.beam_profile]
    Fy_MPa, Ry = GRADES[f.grade]
    Fy = Fy_MPa * 1e3
    Fye = FYE_T51[f.grade] * Fy
    Pu = tributary_axial(f)
    sqEF = math.sqrt(E_STEEL / Fy)
    lam_hd_f = 0.30 * sqEF
    ez = 0.2 * sds_dd2(f.soil)
    sw_b = RHO_STEEL * G_ACC * b['A']
    ns = int(f.n_story)
    # compactness, column (with its axial load) and beam
    Ca = abs(Pu) / (0.90 * Fy * c['A'])
    lam_hd_w = (2.45 * sqEF * (1.0 - 0.93 * Ca) if Ca <= 0.125
                else max(0.77 * sqEF * (2.93 - Ca), 1.49 * sqEF))
    lf_c, lw_c = c['bf'] / (2 * c['tf']), (c['d'] - 2 * c['tf'] - 2 * c['r']) / c['tw']
    lf_b, lw_b = b['bf'] / (2 * b['tf']), (b['d'] - 2 * b['tf'] - 2 * b['r']) / b['tw']
    col_ok = (lf_c <= lam_hd_f) and (lw_c <= lam_hd_w)
    beam_ok = (lf_b <= lam_hd_f) and (lw_b <= 2.45 * sqEF)
    # strong column-weak beam, first-floor interior joint
    dc, db = c['d'], b['d']
    Lclear = f.L_bay - dc
    Ln = Lclear - 2.0 * LH
    a = LH + dc / 2.0
    wS1 = (1.2 + ez) * (f.G_line + sw_b) + 1.0 * f.Q_line
    Puc_lo, Puc_up = wS1 * f.L_bay * ns, wS1 * f.L_bay * (ns - 1)
    Vg = (1.2 * (f.G_line + sw_b) + 0.5 * f.Q_line) * Lclear / 2.0
    Mpr = CPR * Ry * Fy * b['Wpl']
    Mpb = BEAM_FACTOR * Ry * Fy * b['Wpl']
    VE = 2.0 * Mpr / Ln
    SMpb = (Mpb + abs(VE + Vg) * a) + (Mpb + abs(VE - Vg) * a)
    Li_lo, Li_up = f.h1 / 2.0, f.h_typ / 2.0
    g_lo = Li_lo / (Li_lo - db / 2.0)
    g_up = Li_up / (Li_up - db / 2.0)
    SMpc = c['Wpl'] * (Fy - Puc_lo / c['A']) * g_lo + c['Wpl'] * (Fy - Puc_up / c['A']) * g_up
    ratio = SMpc / SMpb
    # axial-load ratio (Annex 5C)
    P6 = (f.G_line * (1.0 + ez) + LIVE_LOAD_N * f.Q_line) * f.L_bay * ns
    i_min = math.sqrt(min(c['Iz'], c['I']) / c['A'])
    p_pc = P6 / (_fcr(Fye, f.h1 / i_min) * c['A'])
    # beam gravity strength, TSSDC 2018 5.3.1 (1) and (2b)
    wu = max(1.4 * (f.G_line + sw_b), 1.2 * (f.G_line + sw_b) + 1.6 * f.Q_line)
    Mu = wu * (f.L_bay - dc) ** 2 / SPAN_DIVISOR
    phiMp = PHI_B * b['Wpl'] * Fy
    # buildability
    clear_beam, clear_col = f.L_bay - dc, f.h_typ - db
    Htot = derived(f)['Htot']
    return [
        dict(name='Compactness, column', criterion='high ductility, TBEC 2018 Table 9.3',
             value=f'b/2tf {lf_c:.2f}, h/tw {lw_c:.1f}', limit=f'{lam_hd_f:.2f}, {lam_hd_w:.1f}', passed=bool(col_ok)),
        dict(name='Compactness, beam', criterion='high ductility, TBEC 2018 Table 9.3',
             value=f'b/2tf {lf_b:.2f}, h/tw {lw_b:.1f}', limit=f'{lam_hd_f:.2f}, {2.45 * sqEF:.1f}', passed=bool(beam_ok)),
        dict(name='Strong column-weak beam', criterion='ratio > 1.0 (first-floor interior joint)',
             value=f'{ratio:.3f}', limit='> 1.0', passed=bool(ratio > 1.0)),
        dict(name='Column axial-load ratio', criterion='P/Pc <= 0.50 (gravity state, TBEC 2018 Annex 5C)',
             value=f'{p_pc:.3f}', limit='<= 0.50', passed=bool(p_pc <= 0.50)),
        dict(name='Depth hierarchy', criterion='beam depth below column depth',
             value=f'{db * 1e3:.0f} mm / {dc * 1e3:.0f} mm', limit='d_beam < d_column', passed=bool(dc > db)),
        dict(name='Beam gravity strength', criterion='Mu <= 0.90 Wpl Fy under 1.4G and 1.2G + 1.6Q',
             value=f'{Mu:.1f} kNm', limit=f'<= {phiMp:.1f} kNm', passed=bool(Mu <= phiMp)),
        dict(name='Buildability', criterion='clear lengths above 0.20 m',
             value=f'{clear_beam:.2f} m, {clear_col:.2f} m', limit='> 0.20 m',
             passed=bool(clear_beam > MIN_CLEAR_LENGTH and clear_col > MIN_CLEAR_LENGTH)),
        dict(name='Total height', criterion='H_tot <= 28 m (single-mode pushover)',
             value=f'{Htot:.2f} m', limit='<= 28 m', passed=bool(Htot <= H_MAX)),
    ]


RANGE_KEYS = ['n_story', 'n_bay', 'L_bay', 'h1', 'h_typ', 'Htot', 'B_tot', 'G_line', 'Q_line', 'M_building', 'd_col',
              'bf_col', 'tf_col', 'tw_col', 'd_beam', 'bf_beam', 'tf_beam', 'tw_beam']


def range_check(f: FrameInput, realized: dict, der: dict | None = None) -> list:
    """Every numeric input against the realized ranges of the database (Table 2). List of dicts: key, value, lo, hi,
    inside."""
    der = der or derived(f)
    vals = dict(n_story=f.n_story, n_bay=f.n_bay, L_bay=f.L_bay, h1=f.h1, h_typ=f.h_typ, G_line=f.G_line,
                Q_line=f.Q_line, **der)
    out = []
    for k in RANGE_KEYS:
        lo, hi = realized[k]
        v = float(vals[k])
        out.append(dict(key=k, value=v, lo=lo, hi=hi, inside=bool(lo <= v <= hi)))
    return out
