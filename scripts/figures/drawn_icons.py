"""Vector drawing primitives for the hand-coded Figure 1 overview.

Everything is built from matplotlib patches and paths in millimetre coordinates
(1 data unit = 1 mm on the printed figure). No raster images are used.
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Ellipse, Polygon, Wedge
from scipy.interpolate import splev, splprep

INK = "#3A3A3A"          # thin dark outline
LW = 0.35                # outline width in pt

RED, ORANGE, PURPLE, BLUE, GREEN = "#C44E52", "#DD8452", "#8172B3", "#4C72B0", "#55A868"
PEACH, LILAC, CONNECT = "#F6C7A8", "#D9C6E6", "#BDBDBD"


# ---------------------------------------------------------------- colour helpers
def tint(c: str, f: float) -> tuple:
    """Mix colour with white (f>0) or black (f<0)."""
    r = np.array(to_rgb(c))
    return tuple(r + (1 - r) * f) if f >= 0 else tuple(r * (1 + f))


# ---------------------------------------------------------------- geometry helpers
def rot(xy: np.ndarray, ang_deg: float, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    a = np.deg2rad(ang_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (xy - [cx, cy]) @ R.T + [cx, cy]


def smooth_closed(pts, n: int = 240, s: float = 0.0) -> np.ndarray:
    """Periodic cubic spline through control points."""
    p = np.asarray(pts, float)
    p = np.vstack([p, p[:1]])
    tck, _ = splprep([p[:, 0], p[:, 1]], s=s, per=True)
    x, y = splev(np.linspace(0, 1, n, endpoint=False), tck)
    return np.column_stack([x, y])


def blob_xy(cx, cy, r, bumps=(), wobble=0.0, ripple=(0, 0.0), seed=0, n=360,
            sx=1.0, sy=1.0, ang=0.0) -> np.ndarray:
    """Closed organic outline: circle + random low harmonics + gaussian bumps (pseudopods).

    bumps: (angle_deg, height, width_deg) - positive height pushes the outline out.
    ripple: (k, amplitude) high-frequency membrane ruffling.
    """
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rr = np.full(n, float(r))
    rng = np.random.default_rng(seed)
    if wobble:
        for k in range(2, 7):
            rr += r * wobble * rng.normal() / k * np.cos(k * th + rng.uniform(0, 2 * np.pi))
    k, a = ripple
    if a:
        rr += r * a * np.sin(k * th + rng.uniform(0, 2 * np.pi)) * (0.6 + 0.4 * np.cos(3 * th))
    for b_ang, h, w in bumps:
        d = np.angle(np.exp(1j * (th - np.deg2rad(b_ang))))
        rr += h * np.exp(-(d / np.deg2rad(w)) ** 2)
    xy = np.column_stack([sx * rr * np.cos(th), sy * rr * np.sin(th)])
    return rot(xy, ang) + [cx, cy]


def capsule_xy(cx, cy, L, W, ang, n=40) -> np.ndarray:
    """Rod (bacillus) outline: rectangle with semicircular caps."""
    r = W / 2
    half = L / 2 - r
    t1 = np.linspace(-np.pi / 2, np.pi / 2, n)
    t2 = np.linspace(np.pi / 2, 3 * np.pi / 2, n)
    right = np.column_stack([half + r * np.cos(t1), r * np.sin(t1)])
    left = np.column_stack([-half + r * np.cos(t2), r * np.sin(t2)])
    return rot(np.vstack([right, left]), ang) + [cx, cy]


def add(ax, artist, clip=None):
    ax.add_patch(artist)
    if clip is not None:
        artist.set_clip_path(clip)
    return artist


def poly(ax, xy, fc, ec=INK, lw=LW, z=2, clip=None, **kw):
    return add(ax, Polygon(xy, closed=True, fc=fc, ec=ec, lw=lw, zorder=z, joinstyle="round", **kw), clip)


def dot(ax, x, y, r, fc, ec="none", lw=LW, z=3, clip=None, **kw):
    return add(ax, Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=z, **kw), clip)


def line(ax, xy, c=INK, lw=LW, z=3, clip=None, **kw):
    xy = np.asarray(xy)
    (ln,) = ax.plot(xy[:, 0], xy[:, 1], color=c, lw=lw, zorder=z, solid_capstyle="round", **kw)
    if clip is not None:
        ln.set_clip_path(clip)
    return ln


def bezier(p0, p1, p2, p3, n=80) -> np.ndarray:
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = map(np.asarray, (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def connector(ax, a, b, lw=0.8, z=0.5, arrow=False):
    """Horizontal-tangent S curve from a to b."""
    dx = (b[0] - a[0]) * 0.55
    xy = bezier(a, (a[0] + dx, a[1]), (b[0] - dx, b[1]), b)
    line(ax, xy, c=CONNECT, lw=lw, z=z)
    if arrow:
        x, y = b
        poly(ax, [(x + 0.2, y), (x - 1.6, y + 0.9), (x - 1.6, y - 0.9)], CONNECT, ec="none", z=z)


# ---------------------------------------------------------------- cells
def bacterium(ax, cx, cy, L, W, ang, clip=None, z=4, pili=True, lps=0, seed=0):
    """Rod bacterium with pili; optional LPS fragments shed nearby."""
    rng = np.random.default_rng(seed)
    out = capsule_xy(cx, cy, L, W, ang)
    if pili:
        idx = np.linspace(0, len(out) - 1, 26).astype(int)
        cen = np.array([cx, cy])
        for i in idx:
            p = out[i]
            v = p - cen
            v = v / np.linalg.norm(v)
            q = p + v * W * rng.uniform(0.25, 0.45)
            line(ax, [p, q], c=tint(RED, -0.35), lw=0.3, z=z - 0.1, clip=clip)
    poly(ax, out, tint(RED, 0.25), z=z, clip=clip)
    poly(ax, capsule_xy(cx, cy, L * 0.8, W * 0.38, ang) + rot(np.array([[0, W * 0.15]]), ang),
         tint(RED, 0.55), ec="none", z=z + 0.1, clip=clip)
    for _ in range(lps):
        a = np.deg2rad(rng.uniform(0, 360))
        d = rng.uniform(0.75, 1.0) * L * 0.85
        lps_fragment(ax, cx + d * np.cos(a), cy + d * np.sin(a) * 0.8, 0.55, rng.uniform(0, 360), clip=clip, z=z)


def lps_fragment(ax, x, y, s, ang, clip=None, z=4):
    """Small branched glycolipid glyph: lipid stub + two sugar chains."""
    pts = rot(np.array([[0, 0], [s, 0], [1.6 * s, 0.55 * s], [1.6 * s, -0.55 * s]]), ang) + [x, y]
    line(ax, [pts[0], pts[1]], c=RED, lw=0.6, z=z, clip=clip)
    line(ax, [pts[1], pts[2]], c=RED, lw=0.6, z=z, clip=clip)
    line(ax, [pts[1], pts[3]], c=RED, lw=0.6, z=z, clip=clip)
    for p in pts[2:]:
        dot(ax, p[0], p[1], 0.28 * s / 0.55 * 0.55, RED, z=z + 0.1, clip=clip)


def nucleus(ax, x, y, r, fc, ang=0, clip=None, z=5, kidney=True, seed=1):
    bumps = [(ang, -0.35 * r, 35)] if kidney else []
    poly(ax, blob_xy(x, y, r, bumps=bumps, wobble=0.08, seed=seed), fc, z=z, clip=clip)
    dot(ax, x - 0.25 * r, y + 0.2 * r, 0.18 * r, tint(fc, -0.15), z=z + 0.1, clip=clip)


def granules(ax, cx, cy, r, n, fc, seed=0, clip=None, z=4.5, size=0.16, avoid=None):
    rng = np.random.default_rng(seed)
    k = 0
    while k < n:
        a, d = rng.uniform(0, 2 * np.pi), r * np.sqrt(rng.uniform(0.05, 1))
        x, y = cx + d * np.cos(a), cy + d * np.sin(a)
        avoids = [] if avoid is None else ([avoid] if np.isscalar(avoid[0]) else avoid)
        if any(np.hypot(x - a[0], y - a[1]) < a[2] for a in avoids):
            continue
        dot(ax, x, y, size * rng.uniform(0.7, 1.2), fc, z=z, clip=clip)
        k += 1


def kupffer_cell(ax, cx, cy, r, clip=None, z=6, seed=3, phagosome=True, ang=0):
    """Star-shaped resident macrophage with long tapering processes."""
    bumps = [(ang + 10, 1.6 * r, 11), (ang + 62, 1.0 * r, 10), (ang + 112, 1.7 * r, 10),
             (ang + 158, 0.9 * r, 12), (ang + 205, 1.5 * r, 11), (ang + 250, 0.8 * r, 11),
             (ang + 292, 1.6 * r, 10), (ang + 335, 0.7 * r, 12),
             (ang + 85, 0.45 * r, 18), (ang + 230, 0.45 * r, 16)]
    out = blob_xy(cx, cy, r, bumps=bumps, wobble=0.06, ripple=(23, 0.03), seed=seed, n=720)
    poly(ax, out, tint(BLUE, 0.35), z=z, clip=clip)
    nucleus(ax, cx - 0.42 * r, cy - 0.12 * r, 0.4 * r, tint(BLUE, -0.05), ang=ang + 20, clip=clip, z=z + 0.2)
    px, py, pr = cx + 0.5 * r, cy + 0.25 * r, 0.46 * r
    granules(ax, cx, cy, 0.85 * r, 14, tint(BLUE, -0.25), seed=seed, clip=clip, z=z + 0.1,
             size=0.06 * r, avoid=[(cx - 0.42 * r, cy - 0.12 * r, 0.55 * r)]
             + ([(px, py, pr * 1.3)] if phagosome else []))
    if phagosome:
        dot(ax, px, py, pr, tint(BLUE, 0.75), ec=INK, z=z + 0.3, clip=clip)
        dot(ax, px, py, pr * 0.86, "none", ec=tint(BLUE, 0.2), lw=0.3, z=z + 0.35, clip=clip)
        bacterium(ax, px, py, pr * 1.25, pr * 0.5, 30, clip=clip, z=z + 0.4, pili=False)


def recruited_macrophage(ax, cx, cy, r, clip=None, z=6, seed=5):
    """Round monocyte-derived macrophage with ruffled membrane."""
    out = blob_xy(cx, cy, r, wobble=0.08, ripple=(19, 0.07), seed=seed,
                  bumps=[(40, 0.25 * r, 14), (200, 0.2 * r, 18)])
    poly(ax, out, tint(GREEN, 0.4), z=z, clip=clip)
    nx, ny = cx + 0.1 * r, cy - 0.05 * r
    granules(ax, cx, cy, 0.85 * r, 16, tint(GREEN, -0.25), seed=seed, clip=clip, z=z + 0.1,
             size=0.055 * r, avoid=(nx, ny, 0.52 * r))
    nucleus(ax, nx, ny, 0.42 * r, tint(GREEN, -0.05), ang=200, clip=clip, z=z + 0.2, kidney=True, seed=8)


def hepatocyte(ax, cx, cy, r, clip=None, z=3, seed=0, ruptured=False):
    """Polygonal hepatocyte; ruptured: torn membrane on the right with fragments and leaking contents."""
    rng = np.random.default_rng(seed)
    ang = np.deg2rad(np.arange(6) * 60 + 30 + rng.uniform(-6, 6))
    pts = np.column_stack([r * np.cos(ang), r * np.sin(ang)]) * rng.uniform(0.92, 1.05, (6, 1))
    out = smooth_closed(pts + [cx, cy], n=360, s=r * 0.05)
    if ruptured:     # swollen, irregular necrotic cell
        out = blob_xy(cx, cy, r, wobble=0.16, ripple=(11, 0.025), seed=seed + 2, n=360,
                      bumps=[(120, 0.08 * r, 30), (230, -0.06 * r, 25)])
    if not ruptured:
        poly(ax, out, PEACH, z=z, clip=clip)
    else:
        th = np.mod(np.arctan2(out[:, 1] - cy, out[:, 0] - cx), 2 * np.pi)
        order = np.argsort(th)
        out, th = out[order], th[order]
        gap = np.deg2rad(30)
        rim = out[(th > gap) & (th < 2 * np.pi - gap)]
        a, b = rim[-1], rim[0]                       # lower and upper ends of the tear
        n = 14
        t = np.linspace(0, 1, n)
        jag = a + (b - a) * t[:, None]
        jag[:, 0] -= 0.16 * r * np.sin(np.pi * t)    # tear bows inward
        jag[1:-1, 0] += rng.uniform(-0.06, 0.06, n - 2) * r
        poly(ax, np.vstack([rim, jag]), PEACH, ec="none", z=z, clip=clip)
        line(ax, rim, c=INK, lw=LW, z=z + 0.1, clip=clip)
        for e, s in ((b, 1), (a, -1)):                # curled membrane flaps
            line(ax, bezier(e, e + [0.12 * r, s * 0.02 * r], e + [0.2 * r, s * 0.1 * r],
                            e + [0.16 * r, s * 0.18 * r], n=20), lw=LW, z=z + 0.1, clip=clip)
        for k, (fx, fy, fa) in enumerate(((1.28, 0.42, 70), (1.42, -0.30, -50), (1.18, -0.02, 100))):
            c = np.array([cx + fx * r, cy + fy * r])
            d = rot(np.array([[-0.08 * r, 0], [0, 0.03 * r], [0.08 * r, 0]]), fa) + c
            line(ax, bezier(d[0], d[1], d[1], d[2], n=12), lw=LW, z=z + 0.1, clip=clip)
    nx = cx - (0.12 * r if ruptured else 0)
    poly(ax, blob_xy(nx, cy, 0.3 * r, wobble=0.06, seed=seed), tint(PEACH, -0.13), z=z + 0.2, clip=clip)
    dot(ax, nx - 0.08 * r, cy + 0.06 * r, 0.07 * r, tint(PEACH, -0.3), z=z + 0.3, clip=clip)
    granules(ax, cx - (0.12 * r if ruptured else 0), cy, 0.6 * r, 5 if not ruptured else 9,
             tint(PEACH, -0.25), seed=seed + 11, clip=clip, z=z + 0.1, size=0.035 * r,
             avoid=(nx, cy, 0.38 * r))


def rbc(ax, x, y, a, b, ang, clip=None, z=4):
    add(ax, Ellipse((x, y), 2 * a, 2 * b, angle=ang, fc=tint(RED, 0.1), ec=INK, lw=LW, zorder=z), clip)
    add(ax, Ellipse((x, y), 1.0 * a, 0.75 * b, angle=ang, fc=tint(RED, 0.4), ec="none", zorder=z + 0.1), clip)


def cytokine(ax, x, y, r, z=4):
    dot(ax, x, y, r, tint(PURPLE, 0.15), ec=INK, z=z)
    dot(ax, x - 0.3 * r, y + 0.3 * r, 0.28 * r, tint(PURPLE, 0.6), z=z + 0.1)


def split_ring(ax, cx, cy, r, w, z=3):
    """Three-colour ring: red (PAMP), orange (DAMP), purple (cytokine) attribution."""
    for c, t0, t1 in ((ORANGE, -25, 85), (RED, 95, 205), (PURPLE, 215, 325)):
        add(ax, Wedge((cx, cy), r, t0, t1, width=w, fc=c, ec=INK, lw=LW, zorder=z))


# ---------------------------------------------------------------- organism and organ
LIVER_CTRL = [(0.0, 0.60), (0.06, 0.88), (0.24, 1.0), (0.48, 0.97), (0.70, 0.95), (0.90, 0.93),
              (1.0, 0.86), (0.92, 0.76), (0.74, 0.64), (0.56, 0.44), (0.44, 0.22), (0.30, 0.06),
              (0.14, 0.14), (0.03, 0.34)]


def liver_xy(x0, y0, w, h) -> np.ndarray:
    p = np.array(LIVER_CTRL) * [w, h] + [x0, y0]
    return smooth_closed(p, n=300)


def liver(ax, x0, y0, w, h, fc=None, z=3):
    fc = fc or tint(RED, -0.08)
    poly(ax, liver_xy(x0, y0, w, h), fc, z=z)
    hl = smooth_closed(np.array([(0.12, 0.62), (0.22, 0.86), (0.40, 0.90), (0.30, 0.72), (0.18, 0.52)])
                       * [w, h] + [x0, y0], n=120)
    poly(ax, hl, tint(RED, 0.3), ec="none", z=z + 0.1)
    groove = bezier((x0 + 0.50 * w, y0 + 0.96 * h), (x0 + 0.46 * w, y0 + 0.8 * h),
                    (x0 + 0.47 * w, y0 + 0.62 * h), (x0 + 0.43 * w, y0 + 0.45 * h))
    line(ax, groove, c=tint(RED, -0.45), lw=0.45, z=z + 0.2)


def mouse(ax, x, y, s=1.0, z=3):
    """Side-view mouse facing right; (x, y) is the body centre, s the scale (body ~ 12 mm long)."""
    grey, pink = "#E4E4E6", "#EDB5B8"
    tail = bezier((x - 5.6 * s, y - 1.2 * s), (x - 9.5 * s, y - 2.8 * s), (x - 7 * s, y - 5.2 * s),
                  (x - 11.5 * s, y - 4.6 * s))
    line(ax, tail, c=INK, lw=1.3, z=z - 0.2)
    line(ax, tail, c=pink, lw=0.8, z=z - 0.1)
    for fx in (-3.2, 2.2):
        add(ax, Ellipse((x + fx * s, y - 3.0 * s), 1.6 * s, 0.7 * s, fc=pink, ec=INK, lw=LW, zorder=z - 0.05))
    body = blob_xy(x, y, 3.3 * s, sx=1.75, sy=1.0, wobble=0.0,
                   bumps=[(25, 0.5 * s, 35), (200, 0.3 * s, 40)], seed=2)
    poly(ax, body, grey, z=z)
    head = smooth_closed(np.array([(3.4, 2.2), (5.6, 2.6), (7.6, 1.2), (9.0, -0.2), (7.6, -0.9),
                                   (5.4, -1.6), (3.6, -0.6)]) * s + [x, y], n=200)
    poly(ax, head, grey, z=z + 0.1)
    dot(ax, x + 5.4 * s, y + 2.9 * s, 1.15 * s, grey, ec=INK, z=z + 0.2)
    dot(ax, x + 5.5 * s, y + 2.9 * s, 0.7 * s, pink, z=z + 0.3)
    dot(ax, x + 7.1 * s, y + 0.8 * s, 0.32 * s, INK, z=z + 0.3)
    dot(ax, x + 9.0 * s, y - 0.2 * s, 0.32 * s, pink, ec=INK, lw=0.25, z=z + 0.3)
    for dy in (0.5, -0.1, -0.7):
        line(ax, [(x + 8.6 * s, y - 0.3 * s), (x + 10.4 * s, y + dy * s)], c="#8C8C8C", lw=0.25, z=z + 0.3)


# ---------------------------------------------------------------- data pictograms (centred at x, y)
def plate(ax, x, y, z=3):
    """Multi-well culture plate with macrophages in each well (stimulus references); 20 x 9 mm."""
    from matplotlib.patches import FancyBboxPatch
    add(ax, FancyBboxPatch((x - 10, y - 4.5), 20, 9, boxstyle="round,pad=0,rounding_size=1.2",
                           fc="#E9ECF1", ec=INK, lw=LW, zorder=z))
    add(ax, FancyBboxPatch((x - 9.1, y - 3.7), 18.2, 7.4, boxstyle="round,pad=0,rounding_size=0.8",
                           fc="#F6F7F9", ec=tint(INK, 0.5), lw=0.3, zorder=z + 0.1))
    for i in range(4):
        for j in range(2):
            wx, wy = x - 6.6 + i * 4.4, y + 1.7 - j * 3.4
            dot(ax, wx, wy, 1.45, tint(RED, 0.8), ec=INK, z=z + 0.2)
            poly(ax, blob_xy(wx, wy, 0.6, bumps=[(30, 0.3, 22), (150, 0.25, 22), (260, 0.3, 22)],
                             wobble=0.1, seed=i + 3 * j), tint(BLUE, 0.35), lw=0.25, z=z + 0.3)
            dot(ax, wx - 0.1, wy - 0.05, 0.25, tint(BLUE, -0.1), z=z + 0.4)
    return x - 10, x + 10


def tubes_sorter(ax, x, y, z=3):
    """Three sorted-cell tubes beside a cell sorter; 22 x 9 mm."""
    for k, c in enumerate((BLUE, GREEN, PURPLE)):
        tx = x - 9.6 + k * 3.0
        body = np.array([(tx - 1.0, y + 3.6), (tx + 1.0, y + 3.6), (tx + 1.0, y - 2.2), (tx, y - 4.5),
                         (tx - 1.0, y - 2.2)])
        liquid = np.array([(tx - 1.0, y + 0.2), (tx + 1.0, y + 0.2), (tx + 1.0, y - 2.2), (tx, y - 4.5),
                           (tx - 1.0, y - 2.2)])
        poly(ax, body, "#F4F6F9", z=z)
        poly(ax, liquid, tint(c, 0.55), ec="none", z=z + 0.1)
        poly(ax, body, "none", z=z + 0.2)
        for dx, dy in ((-0.4, -0.6), (0.35, -1.4), (-0.2, -2.4), (0.3, -0.2)):
            dot(ax, tx + dx, y + dy, 0.22, tint(c, -0.2), z=z + 0.3)
        poly(ax, np.array([(tx - 1.2, y + 3.6), (tx + 1.2, y + 3.6), (tx + 1.2, y + 4.4), (tx - 1.2, y + 4.4)]),
             tint(c, 0.25), z=z + 0.2)
    # sorter: box with front panel, sample port and laser window
    bx = x + 1.0
    poly(ax, np.array([(bx, y - 4.0), (bx + 8.6, y - 4.0), (bx + 8.6, y + 3.0), (bx, y + 3.0)]), "#DCDFE5", z=z)
    poly(ax, np.array([(bx, y + 3.0), (bx + 8.6, y + 3.0), (bx + 9.8, y + 4.3), (bx + 1.2, y + 4.3)]),
         "#EEF0F3", z=z)
    poly(ax, np.array([(bx + 8.6, y - 4.0), (bx + 9.8, y - 2.7), (bx + 9.8, y + 4.3), (bx + 8.6, y + 3.0)]),
         "#BFC4CC", z=z)
    poly(ax, np.array([(bx + 0.8, y - 0.2), (bx + 7.8, y - 0.2), (bx + 7.8, y + 0.5), (bx + 0.8, y + 0.5)]),
         BLUE, lw=0.25, z=z + 0.1)
    poly(ax, np.array([(bx + 1.2, y - 3.2), (bx + 3.8, y - 3.2), (bx + 3.8, y - 1.0), (bx + 1.2, y - 1.0)]),
         "#5D6470", lw=0.25, z=z + 0.1)
    for k in range(3):
        dot(ax, bx + 5.4 + k * 0.9, y + 1.8, 0.28, (GREEN, ORANGE, RED)[k], z=z + 0.1)
    return x - 10.8, bx + 9.8


def single_cell(ax, x, y, z=3):
    """Microfluidic droplet with one cell and a small embedding with three clusters; 22 x 9 mm."""
    cx = x - 6.0
    for dy in (1.1, -1.1):
        line(ax, [(cx - 5.0, y + dy), (cx + 5.0, y + dy)], c=INK, lw=LW, z=z)
    poly(ax, np.array([(cx - 5.0, y - 1.1), (cx + 5.0, y - 1.1), (cx + 5.0, y + 1.1), (cx - 5.0, y + 1.1)]),
         "#EEF3F8", ec="none", z=z - 0.1)
    for dx in (-3.7, 3.7):
        dot(ax, cx + dx, y, 0.75, tint(BLUE, 0.8), ec=INK, z=z + 0.1)
    dot(ax, cx, y, 2.0, tint(BLUE, 0.85), ec=INK, z=z + 0.2)
    poly(ax, blob_xy(cx, y, 0.85, bumps=[(20, 0.45, 15), (140, 0.4, 15), (250, 0.45, 15)], seed=7),
         tint(BLUE, 0.2), lw=0.25, z=z + 0.3)
    dot(ax, cx - 0.1, y - 0.05, 0.35, tint(BLUE, -0.1), z=z + 0.4)
    ax0 = x + 1.5
    line(ax, [(ax0, y + 4.3), (ax0, y - 4.0), (ax0 + 9.5, y - 4.0)], c=INK, lw=0.4, z=z)
    rng = np.random.default_rng(11)
    for (mx, my, c) in ((3.2, 1.6, BLUE), (7.4, 2.2, GREEN), (5.8, -1.9, PURPLE)):
        pts = rng.normal(0, 0.7, (18, 2))
        for px, py in pts:
            dot(ax, ax0 + mx + px, y + my + py * 0.8, 0.3, c, z=z + 0.1)
    return cx - 5.0, ax0 + 9.5


def spatial(ax, x, y, z=3):
    """Liver section with coloured transcript spots; 22 x 9 mm."""
    w, h = 21, 9.5
    out = liver_xy(x - w / 2, y - h / 2, w, h)
    poly(ax, out, "#F4DCE2", z=z)
    clip = ax.patches[-1]
    rng = np.random.default_rng(4)
    for gx in np.arange(x - w / 2 + 0.8, x + w / 2, 1.05):
        for gy in np.arange(y - h / 2 + 0.6, y + h / 2, 1.05):
            u = (gx - (x - w / 2)) / w
            p = rng.uniform()
            if p < 0.55:
                continue
            c = RED if u < 0.35 + rng.normal(0, 0.08) else (ORANGE if u < 0.62 + rng.normal(0, 0.08) else PURPLE)
            dot(ax, gx, gy, 0.3, c, z=z + 0.1, clip=clip)
    vein = bezier((x - 4, y - 1), (x - 1, y + 0.5), (x + 2, y - 1.2), (x + 5, y + 0.4))
    line(ax, vein, c="#E7B9C3", lw=1.2, z=z + 0.05, clip=clip)
    poly(ax, out, "none", z=z + 0.2)
    return x - w / 2, x + w / 2


def proteome(ax, x, y, z=3):
    """Mass-spectrum peak trace; 22 x 9 mm."""
    xs = np.linspace(x - 10, x + 10, 600)
    peaks = [(-8.0, 1.6), (-6.4, 5.2), (-5.2, 3.4), (-2.6, 1.4), (-0.9, 2.6), (0.6, 8.0), (3.3, 1.1),
             (4.8, 4.0), (5.6, 1.6), (7.6, 2.8), (8.6, 1.0)]
    ys = np.full_like(xs, y - 4.2)
    for px, ph in peaks:
        ys += ph * np.exp(-((xs - (x + px)) / 0.22) ** 2)
    poly(ax, np.vstack([np.column_stack([xs, ys]), [(xs[-1], y - 4.2), (xs[0], y - 4.2)]]),
         tint(BLUE, 0.7), ec="none", z=z)
    line(ax, np.column_stack([xs, ys]), c=BLUE, lw=0.5, z=z + 0.1)
    line(ax, [(x - 10.5, y - 4.25), (x + 10.5, y - 4.25)], c=INK, lw=0.45, z=z + 0.2)
    return x - 10.5, x + 10.5
