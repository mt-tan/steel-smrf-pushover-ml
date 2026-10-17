"""Pushover Response Predictor for Steel Moment Frames: desktop entry point.

    python gui/app.py

Left: frame inputs (geometry, loads, steel grade, soil class, column and beam profiles), the derived model inputs
(read-only) and the validity checks (range check against the database of the paper, closed-form screens of the dataset
generator). Right: the eight characteristic quantities for DD-2 and DD-1 with their 90 % split-conformal intervals, a
plot of the predicted characteristic points, joined for each hazard level by a piecewise-linear curve in the order of
displacement (core/curve.py; not an analysis result), and the validity note. CSV export of the results.
The predictions are those of the ANN of the paper, the mean of five networks per quantity (assets/models/, seeds and
gate tier in model_metadata.json); no OpenSees analysis is run.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
from core.curve import LEGEND as CURVE_LEGEND, curve_points  # noqa: E402
from core.frame import FrameInput  # noqa: E402
from core.predictor import HAZARDS, TARGETS, ANNBundle, predict_frame  # noqa: E402
from core.screens import NOTE, range_check, screens  # noqa: E402
from ui import theme as T  # noqa: E402

APP_TITLE = 'Pushover Response Predictor for Steel Moment Frames'
APP_VERSION = '1.0 (Paper 1)'
SPEC = json.loads((APP_DIR / 'assets' / 'gui_spec.json').read_text(encoding='utf-8'))
ITALIC_SUBS = {'y', 't', 'u', 'δt'}
PAD = 10


class Symbol(ctk.CTkFrame):
    """Symbol of the paper notation: italic base, subscript at 75 % (italic for variable subscripts), unit."""

    def __init__(self, master, key, size=T.SIZE_TEXT, unit=True, weight='normal', color=T.TEXT, fg='transparent'):
        super().__init__(master, fg_color=fg)
        base, sub, u = T.SYMBOLS[key]
        sub_size = round(size * T.SUB_RATIO)
        ctk.CTkLabel(self, text=base, text_color=color, fg_color='transparent', height=size + 8,
                     font=ctk.CTkFont(family=T.FAMILY, size=size, weight=weight, slant='italic')).grid(
            row=0, column=0, sticky='s')
        col = 1
        if sub:
            ctk.CTkLabel(self, text=sub, text_color=color, fg_color='transparent', height=sub_size + 4,
                         font=ctk.CTkFont(family=T.FAMILY, size=sub_size, weight=weight,
                                          slant='italic' if sub in ITALIC_SUBS else 'roman')).grid(
                row=0, column=col, sticky='s', pady=(0, 1))
            col += 1
        if unit and u:
            ctk.CTkLabel(self, text=f' [{u}]', text_color=color, fg_color='transparent', height=size + 8,
                         font=ctk.CTkFont(family=T.FAMILY, size=size, weight=weight)).grid(row=0, column=col, sticky='s')


class Rich(ctk.CTkFrame):
    """One line of text in pieces: (text, style) with style n (normal), i (italic), s (subscript), si (italic
    subscript), b (bold)."""

    def __init__(self, master, parts, size=T.SIZE_TEXT, color=T.TEXT, fg='transparent'):
        super().__init__(master, fg_color=fg)
        sub = round(size * T.SUB_RATIO)
        for k, (txt, st) in enumerate(parts):
            sz = sub if st in ('s', 'si') else size
            fnt = ctk.CTkFont(family=T.FAMILY, size=sz, weight='bold' if st == 'b' else 'normal',
                              slant='italic' if st in ('i', 'si') else 'roman')
            ctk.CTkLabel(self, text=txt, font=fnt, text_color=color, fg_color='transparent',
                         height=(sz + 4) if st in ('s', 'si') else (sz + 8)).grid(
                row=0, column=k, sticky='s', pady=(0, 1) if st in ('s', 'si') else 0)


class App(ctk.CTk):
    def __init__(self, hidden: bool = False):
        ctk.set_appearance_mode('light')
        ctk.set_default_color_theme('blue')
        super().__init__()
        if hidden:
            self.withdraw()
        self.title(f'{APP_TITLE} - TBEC 2018')
        self.geometry(f'{T.WINDOW[0]}x{T.WINDOW[1]}')
        self.resizable(False, False)
        self.configure(fg_color=T.BG)
        self.bundle = ANNBundle()
        self.result = None
        self.vars = {}
        self._build()
        self.load_default()

    # ------------------------------------------------------------------ layout
    def _build(self):
        f_text, f_head = T.font(T.SIZE_TEXT), T.font(T.SIZE_HEAD, 'bold')
        head = ctk.CTkFrame(self, fg_color=T.BG)
        head.pack(fill='x', padx=PAD, pady=(6, 4))
        ctk.CTkLabel(head, text=APP_TITLE, font=T.font(T.SIZE_TITLE, 'bold'), text_color=T.HEAD, height=32).pack(
            anchor='w')
        ctk.CTkLabel(head, text='Frames screened against TBEC 2018 and TSSDC 2018: eight response quantities with 90 % '
                     'intervals', font=f_text, text_color=T.MUTED, height=26).pack(anchor='w')
        body = ctk.CTkFrame(self, fg_color=T.BG)
        body.pack(fill='both', expand=True, padx=PAD, pady=(0, PAD))
        left = ctk.CTkFrame(body, fg_color=T.PANEL, border_color=T.BORDER, border_width=1, width=400)
        left.pack(side='left', fill='y', padx=(0, PAD))
        left.pack_propagate(False)
        right = ctk.CTkFrame(body, fg_color=T.PANEL, border_color=T.BORDER, border_width=1)
        right.pack(side='left', fill='both', expand=True)

        # ---- inputs
        grid = ctk.CTkFrame(left, fg_color='transparent')
        grid.pack(fill='x', padx=PAD, pady=(PAD, 0))
        ctk.CTkLabel(grid, text='Frame', font=f_head, text_color=T.HEAD).grid(row=0, column=0, columnspan=2, sticky='w')
        rows = [('n_story', 'menu', [str(v) for v in SPEC['story_counts']]),
                ('n_bay', 'menu', [str(v) for v in SPEC['bay_counts']]),
                ('L_bay', 'entry', None), ('h1', 'entry', None), ('h_typ', 'entry', None),
                ('G_line', 'entry', None), ('Q_line', 'entry', None),
                ('grade', 'menu', SPEC['grades']), ('soil', 'menu', SPEC['soils']),
                ('col_profile', 'menu', SPEC['column_profiles']), ('beam_profile', 'menu', SPEC['beam_profiles'])]
        names = {'grade': 'Steel grade', 'soil': 'Soil class', 'col_profile': 'Column', 'beam_profile': 'Beam'}
        for i, (key, kind, values) in enumerate(rows, start=1):
            if key in names:
                ctk.CTkLabel(grid, text=names[key], font=f_text, text_color=T.TEXT, height=28).grid(
                    row=i, column=0, sticky='w', pady=0)
            else:
                Symbol(grid, key).grid(row=i, column=0, sticky='w', pady=0)
            var = ctk.StringVar()
            self.vars[key] = var
            if kind == 'menu':
                w = ctk.CTkOptionMenu(grid, values=values, variable=var, font=f_text, dropdown_font=f_text, width=170,
                                      height=28, command=lambda _v: self.update_all())
            else:
                w = ctk.CTkEntry(grid, textvariable=var, font=f_text, width=170, height=28)
                w.bind('<Return>', lambda _e: self.update_all())
                w.bind('<FocusOut>', lambda _e: self.update_all())
            w.grid(row=i, column=1, sticky='e', padx=(8, 0), pady=1 if i > 1 else (2, 1))
        grid.grid_columnconfigure(0, weight=1)

        # ---- derived inputs (read-only)
        der = ctk.CTkFrame(left, fg_color='transparent')
        der.pack(fill='x', padx=PAD, pady=(2, 0))
        ctk.CTkLabel(der, text='Derived inputs', font=f_head, text_color=T.HEAD, height=28).grid(
            row=0, column=0, columnspan=4, sticky='w')
        self.der_labels = {}
        for key, r, c in (('Htot', 1, 0), ('B_tot', 1, 2), ('M_building', 2, 0)):
            Symbol(der, key).grid(row=r, column=c, sticky='w', padx=(0 if c == 0 else 14, 0))
            lab = ctk.CTkLabel(der, text='', font=f_text, text_color=T.TEXT, height=26)
            lab.grid(row=r, column=c + 1, sticky='e', padx=(6, 0))
            self.der_labels[key] = lab
        Rich(der, [('d', 'i'), (' / ', 'n'), ('b', 'i'), (' / ', 'n'), ('t', 'i'), ('f', 's'), (' / ', 'n'),
                   ('t', 'i'), ('w', 's'), (' [mm]', 'n')], color=T.MUTED).grid(row=3, column=0, columnspan=4,
                                                                                sticky='w')
        for i, (lab_txt, key) in enumerate([('Column', 'col'), ('Beam', 'beam')]):
            ctk.CTkLabel(der, text=lab_txt, font=f_text, text_color=T.TEXT, height=26).grid(row=4 + i, column=0,
                                                                                           sticky='w')
            lab = ctk.CTkLabel(der, text='', font=f_text, text_color=T.TEXT, height=26)
            lab.grid(row=4 + i, column=1, columnspan=3, sticky='e', padx=(6, 0))
            self.der_labels[key] = lab
        der.grid_columnconfigure(1, weight=1)
        der.grid_columnconfigure(3, weight=1)

        # ---- validity
        val = ctk.CTkFrame(left, fg_color='transparent')
        val.pack(fill='x', padx=PAD, pady=(2, 0))
        ctk.CTkLabel(val, text='Validity checks', font=f_head, text_color=T.HEAD, height=28).grid(
            row=0, column=0, columnspan=2, sticky='w')
        self.range_label = ctk.CTkLabel(val, text='', font=f_text, text_color=T.TEXT, height=26, justify='left',
                                        wraplength=370, anchor='w')
        self.range_label.grid(row=1, column=0, columnspan=2, sticky='w')
        self.screen_labels = []
        short = [[('Compactness, column', 'n')], [('Compactness, beam', 'n')], [('Strong column-weak beam', 'n')],
                 [('Axial load ', 'n'), ('P', 'i'), ('/', 'n'), ('P', 'i'), ('c', 's')], [('Depth hierarchy', 'n')],
                 [('Beam gravity strength', 'n')], [('Buildability', 'n')], [('Total height', 'n')]]
        for i, parts in enumerate(short):
            Rich(val, parts).grid(row=2 + i, column=0, sticky='w')
            lab = ctk.CTkLabel(val, text='', font=T.font(T.SIZE_TEXT, 'bold'), height=25)
            lab.grid(row=2 + i, column=1, sticky='e')
            self.screen_labels.append(lab)
        val.grid_columnconfigure(0, weight=1)

        # ---- results table
        tab = ctk.CTkFrame(right, fg_color='transparent')
        tab.pack(fill='x', padx=PAD, pady=(PAD, 0))
        heads = ['Quantity', 'DD-2', '90 % interval', 'DD-1', '90 % interval']
        widths = [120, 90, 190, 90, 190]
        for j, (h, wd) in enumerate(zip(heads, widths)):
            ctk.CTkLabel(tab, text=h, font=T.font(T.SIZE_TEXT, 'bold'), text_color='white', fg_color=T.HEAD,
                         width=wd, height=32, anchor='w' if j == 0 else 'e', padx=6).grid(row=0, column=j,
                                                                                          sticky='ew')
        self.cells = {}
        for i, t in enumerate(TARGETS, start=1):
            bg = '#F2F4F7' if i % 2 == 0 else T.PANEL
            cell = ctk.CTkFrame(tab, fg_color=bg, corner_radius=0, height=31, width=widths[0])
            cell.grid(row=i, column=0, sticky='nsew')
            Symbol(cell, t, fg=bg).pack(side='left', padx=(4, 0))
            for j in range(1, 5):
                lab = ctk.CTkLabel(tab, text='', font=f_text, text_color=T.TEXT, fg_color=bg, width=widths[j],
                                   height=31, anchor='e', padx=6)
                lab.grid(row=i, column=j, sticky='ew')
                self.cells[(t, j)] = lab
        self.warn_label = ctk.CTkLabel(right, text='', font=f_text, text_color=T.WARN, justify='left',
                                       wraplength=680, anchor='w')
        self.warn_label.pack(fill='x', padx=PAD, pady=(4, 0))

        # ---- plot of the characteristic points
        from customtkinter.windows.widgets.scaling import ScalingTracker
        s = ScalingTracker.get_widget_scaling(self)
        self.fig = Figure(figsize=(690 / 72, 332 / 72), dpi=72 * s, facecolor='white')
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(padx=PAD, pady=(4, 0))

        self.note_label = ctk.CTkLabel(right, text=NOTE, font=f_text, text_color=T.MUTED, justify='left',
                                       wraplength=690, anchor='w')
        self.note_label.pack(fill='x', padx=PAD, pady=(6, 0))
        tier = self.bundle.tier
        self.status = ctk.CTkLabel(right, text=f'ANN of the paper, five networks each (gate {tier}); no structural analysis is run.',
                                   font=f_text, text_color=T.MUTED, anchor='w', wraplength=690, justify='left')
        self.status.pack(fill='x', padx=PAD, pady=(4, 0))
        btns = ctk.CTkFrame(right, fg_color='transparent')
        btns.pack(fill='x', padx=PAD, pady=(8, PAD))
        ctk.CTkButton(btns, text='Predict', font=f_text, width=130, height=36, command=self.update_all).pack(
            side='left')
        ctk.CTkButton(btns, text='Default example', font=f_text, width=190, height=36, fg_color=T.MUTED,
                      command=self.load_default).pack(side='left', padx=10)
        ctk.CTkButton(btns, text='Export CSV', font=f_text, width=150, height=36, fg_color=T.MUTED,
                      command=self.export_dialog).pack(side='left')

    # ------------------------------------------------------------------ logic
    def load_default(self):
        d = SPEC['default']
        for k in ('n_story', 'n_bay'):
            self.vars[k].set(str(d[k]))
        for k in ('L_bay', 'h1', 'h_typ'):
            self.vars[k].set(f'{d[k]:.3f}')
        for k in ('G_line', 'Q_line'):
            self.vars[k].set(f'{d[k]:.2f}')
        for k in ('grade', 'soil', 'col_profile', 'beam_profile'):
            self.vars[k].set(d[k])
        self.update_all()

    def frame_input(self) -> FrameInput:
        v = {k: self.vars[k].get().strip() for k in self.vars}
        return FrameInput(n_story=int(v['n_story']), n_bay=int(v['n_bay']), L_bay=float(v['L_bay']),
                          h1=float(v['h1']), h_typ=float(v['h_typ']), G_line=float(v['G_line']),
                          Q_line=float(v['Q_line']), grade=v['grade'], soil=v['soil'], col_profile=v['col_profile'],
                          beam_profile=v['beam_profile'])

    def update_all(self):
        try:
            f = self.frame_input()
        except ValueError:
            self.status.configure(text='Enter numbers for the bay width, story heights and line loads.',
                                  text_color=T.FAIL)
            return
        res = predict_frame(self.bundle, f)
        self.result = res
        self.screen_rows = screens(f)
        self.range_rows = range_check(f, SPEC['realized_ranges'], res.derived)
        der = res.derived
        self.der_labels['Htot'].configure(text=f"{der['Htot']:.2f}")
        self.der_labels['B_tot'].configure(text=f"{der['B_tot']:.2f}")
        self.der_labels['M_building'].configure(text=f"{der['M_building']:.1f}")
        self.der_labels['col'].configure(text=f"{der['d_col']:.0f} / {der['bf_col']:.0f} / {der['tf_col']:g} / "
                                              f"{der['tw_col']:g}")
        self.der_labels['beam'].configure(text=f"{der['d_beam']:.0f} / {der['bf_beam']:.0f} / {der['tf_beam']:g} / "
                                               f"{der['tw_beam']:g}")
        outside = [r for r in self.range_rows if not r['inside']]
        if outside:
            self.range_label.configure(text='Outside the database ranges (Table 2): ' + ', '.join(
                T.SYMBOLS[r['key']][0] + T.SYMBOLS[r['key']][1] for r in outside), text_color=T.FAIL)
        else:
            self.range_label.configure(text='All inputs inside the database ranges', text_color=T.PASS)
        for lab, s in zip(self.screen_labels, self.screen_rows):
            lab.configure(text='pass' if s['passed'] else 'fail', text_color=T.PASS if s['passed'] else T.FAIL)
        for t in TARGETS:
            for j, h in ((1, 'DD-2'), (3, 'DD-1')):
                lo, hi = res.intervals[h][t]
                self.cells[(t, j)].configure(text=T.fmt(t, res.values[h][t]))
                self.cells[(t, j + 1)].configure(text=f'{T.fmt(t, lo)} to {T.fmt(t, hi)}')
        warn = list(res.warnings)
        for h in HAZARDS:                                # warnings of the plotted curve (core/curve.py)
            warn += [f'{h}: {w}.' for w in curve_points(res.values[h])[2]]
        if outside or not all(s['passed'] for s in self.screen_rows):
            warn.append('The frame is outside the validated domain: see the validity checks.')
        self.warn_label.configure(text=' '.join(warn) if warn else 'No warning for this frame.',
                                  text_color=T.WARN if warn else T.PASS)
        self.status.configure(text=f'ANN of the paper, five networks each (gate {self.bundle.tier}); no structural analysis is run.',
                              text_color=T.MUTED)
        self._plot(res)

    def _plot(self, res):
        ax = self.ax
        ax.clear()
        sz = T.SIZE_TEXT
        mk = dict(yield_=('o', '#3E6DB5'), peak=('s', '#1F3A5F'), ultimate=('D', '#6B4C9A'))
        tcol = {'DD-2': '#CC3311', 'DD-1': '#EE7733'}
        # piecewise-linear curve of each hazard level through its predicted points, in increasing displacement, dashed
        # in the colour of that level's target marker and under every marker (style of the companion RC tool)
        self.curve_lines, self.target_markers = {}, {}
        for h in HAZARDS:
            pts, _, _ = curve_points(res.values[h])
            self.curve_lines[h], = ax.plot([p[0] * 1000 for p in pts], [p[1] for p in pts], '--', color=tcol[h], lw=1.8,
                                           zorder=1)
        # capacity points of both records: DD-1 as an open ring under the filled DD-2 marker (they nearly coincide)
        for h, filled, ms, z in (('DD-1', False, 17, 2), ('DD-2', True, 10, 3)):
            v = res.values[h]
            pts = [('yield_', v['dy'], v['Vy']), ('peak', v['dmax'], v['Vmax']), ('ultimate', v['du'], 0.8 * v['Vmax'])]
            for name, d, V in pts:
                m, c = mk[name]
                ax.plot(d * 1000, V, m, ms=ms, mfc=c if filled else 'none', mec=c, mew=1.6, zorder=z)
        for h, m in (('DD-2', '^'), ('DD-1', 'v')):
            v = res.values[h]
            self.target_markers[h], = ax.plot(v['dt'] * 1000, v['V_at_dt'], m, ms=12, mfc=tcol[h], mec=tcol[h], zorder=4)
        ax.set_xlabel('roof displacement [mm]', fontsize=sz)
        ax.set_ylabel('base shear [kN]', fontsize=sz)
        ax.tick_params(labelsize=sz)
        dmax_all = max(max(res.values[h]['du'], res.values[h]['dt']) for h in HAZARDS) * 1000
        vtop = max(max(res.values[h][t] for t in ('Vmax', 'Vy', 'V_at_dt')) for h in HAZARDS)
        ax.set_xlim(0, 1.08 * dmax_all)
        ax.set_ylim(0, 1.52 * vtop)                     # headroom for the curve legend above the points
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.grid(True, ls='--', lw=0.6, alpha=0.5)
        handles = [Line2D([], [], ls='', marker='o', ms=10, color='#3E6DB5', label='yield'),
                   Line2D([], [], ls='', marker='s', ms=10, color='#1F3A5F', label='peak'),
                   Line2D([], [], ls='', marker='D', ms=9, color='#6B4C9A', label='ultimate'),
                   Line2D([], [], ls='', marker='^', ms=12, color='#CC3311', label='target DD-2'),
                   Line2D([], [], ls='', marker='v', ms=12, color='#EE7733', label='target DD-1'),
                   Line2D([], [], ls='', marker='o', ms=15, mfc='none', mec='0.35', mew=1.6, label='DD-1 record')]
        leg = ax.legend(handles=handles, fontsize=sz, loc='lower right', ncol=2, frameon=True, framealpha=0.95,
                        handletextpad=0.2, columnspacing=0.8, borderpad=0.3, labelspacing=0.25)
        ax.add_artist(leg)
        both = tuple(Line2D([], [], ls='--', lw=1.8, color=tcol[h]) for h in HAZARDS)
        ax.legend([both], [CURVE_LEGEND.replace(' (not', '\n(not')], handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)},
                  fontsize=sz, loc='upper left', frameon=True, framealpha=0.95, handlelength=3.2, handletextpad=0.4,
                  borderpad=0.3)
        self.fig.tight_layout(pad=0.4)
        self.canvas.draw()

    # ------------------------------------------------------------------ export
    def export_rows(self) -> list:
        res = self.result
        rows = [['section', 'quantity', 'hazard', 'value', 'interval_low', 'interval_high', 'unit']]
        for k, v in res.frame.as_dict().items():
            rows.append(['input', k, '', v, '', '', ''])
        for k, v in res.derived.items():
            rows.append(['derived input', k, '', v, '', '', ''])
        units = {'Vy': 'kN', 'dy': 'm', 'Vmax': 'kN', 'dmax': 'm', 'V_at_dt': 'kN', 'dt': 'm', 'du': 'm', 'T1': 's'}
        for h in HAZARDS:
            for t in TARGETS:
                lo, hi = res.intervals[h][t]
                rows.append(['prediction', t, h, res.values[h][t], lo, hi, units[t]])
        for s in self.screen_rows:
            rows.append(['screen', s['name'], '', 'pass' if s['passed'] else 'fail', s['value'], s['limit'], ''])
        for r in self.range_rows:
            rows.append(['range check', r['key'], '', r['value'], r['lo'], r['hi'], 'inside' if r['inside'] else
                         'outside'])
        rows.append(['note', NOTE, '', '', '', '', ''])
        rows.append(['models', f"ANN of the paper, mean of five networks per quantity (seeds "
                     f"{', '.join(map(str, self.bundle.seeds))}), deployment gate {self.bundle.tier}, 90 % "
                     'split-conformal intervals', '', '', '', '', ''])
        return rows

    def export_csv(self, path):
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            csv.writer(fh).writerows(self.export_rows())

    def export_dialog(self):
        p = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV', '*.csv')],
                                         initialfile='steel_frame_prediction.csv')
        if p:
            self.export_csv(p)
            self.status.configure(text=f'Results written to {p}', text_color=T.PASS)


if __name__ == '__main__':
    App().mainloop()
