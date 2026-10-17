"""Colours, fonts and the notation of the GUI.

Font sizes are in logical pixels (CustomTkinter scales them with the window). The smallest text is 20 px in a window of
1,140 px: a screenshot of the full window placed at 6.3 in keeps every text at 8 pt or more (the paper figure minimum
is 7.85 pt printed). Subscripts are drawn at 75 % of their symbol, as mathtext does in the paper figures.
"""
BG = '#F4F6F9'
PANEL = '#FFFFFF'
BORDER = '#D5DAE1'
HEAD = '#1F3A5F'           # table header fill of the manuscript tables
TEXT = '#1B1F24'
MUTED = '#5A6472'
PASS = '#2E7D32'
FAIL = '#C62828'
WARN = '#B26A00'
ACCENT = '#3E6DB5'
FAMILY = 'Arial'

SIZE_TITLE = 26
SIZE_HEAD = 22
SIZE_TEXT = 20
SUB_RATIO = 0.75
WINDOW = (1140, 915)

# symbol (base, subscript) and unit of every quantity shown; units of the displacements in mm in the GUI
SYMBOLS = {
    'n_story': ('N', 'story', ''), 'n_bay': ('N', 'bay', ''), 'L_bay': ('L', 'bay', 'm'), 'h1': ('h', '1', 'm'),
    'h_typ': ('h', 'typ', 'm'), 'Htot': ('H', 'tot', 'm'), 'B_tot': ('B', 'tot', 'm'), 'G_line': ('G', '', 'kN/m'),
    'Q_line': ('Q', '', 'kN/m'), 'M_building': ('M', '', 't'),
    'd_col': ('d', 'column', 'mm'), 'bf_col': ('b', 'column', 'mm'), 'tf_col': ('t', 'f,column', 'mm'),
    'tw_col': ('t', 'w,column', 'mm'), 'd_beam': ('d', 'beam', 'mm'), 'bf_beam': ('b', 'beam', 'mm'),
    'tf_beam': ('t', 'f,beam', 'mm'), 'tw_beam': ('t', 'w,beam', 'mm'),
    'Vy': ('V', 'y', 'kN'), 'dy': ('δ', 'y', 'mm'), 'Vmax': ('V', 'max', 'kN'), 'dmax': ('δ', 'max', 'mm'),
    'V_at_dt': ('V', 'δt', 'kN'), 'dt': ('δ', 't', 'mm'), 'du': ('δ', 'u', 'mm'), 'T1': ('T', '1', 's'),
}
DISPLAY_SCALE = {'dy': 1000.0, 'dmax': 1000.0, 'dt': 1000.0, 'du': 1000.0}     # m -> mm
DECIMALS = {'Vy': 0, 'Vmax': 0, 'V_at_dt': 0, 'dy': 1, 'dmax': 1, 'dt': 1, 'du': 1, 'T1': 3}


def font(size=SIZE_TEXT, weight='normal'):
    import customtkinter as ctk
    return ctk.CTkFont(family=FAMILY, size=size, weight=weight)


def fmt(t: str, v: float) -> str:
    return f'{v * DISPLAY_SCALE.get(t, 1.0):,.{DECIMALS[t]}f}'
