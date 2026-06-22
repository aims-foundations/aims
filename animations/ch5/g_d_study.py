"""
AIMS Chapter 5 Animation: Generalizability and Decision Studies
G-study estimates variance components; D-study trades items vs raters
against a target reliability.

Run with:
    PATH="/lfs/local/0/sttruong/miniconda3/bin:$PATH" \
    manim -qh --disable_caching --media_dir media/ch5 animations/ch5/g_d_study.py GandDStudy
"""

from manim import *
import numpy as np

# ── design tokens ────────────────────────────────────────────────
ACCENT = "#FFD966"
BG = "#0f0f0f"
TEXT2 = "#aaaaaa"
PAL = ["#5B8DEE", "#45BF7C", "#F0A35C", "#E8637A", "#B07CD8"]
AXIS_CLR = "#888888"

# ── synthesized variance components (person p, item i, rater r) ──
# Tuned so the G = 0.90 contour sweeps the grid diagonally: adding raters
# clears the target while items alone (at one rater) never do — i.e. the
# person x rater interaction dominates the measurement error.
SIG_P = 1.0      # person variance (the signal)
SIG_I = 0.18     # item main effect
SIG_R = 0.22     # rater main effect
SIG_PI = 0.18    # person x item
SIG_PR = 0.35    # person x rater (dominant — favors adding raters)
SIG_PIR = 0.10   # three-way + residual

# heatmap grid axes
N_ITEMS = [5, 10, 20, 35, 50]
N_RATERS = [1, 2, 3, 4, 5, 6]
TARGET_G = 0.90


def reliability(n_i, n_r):
    """Generalizability coefficient for a person x item x rater design."""
    err = (
        SIG_PI / n_i
        + SIG_PR / n_r
        + SIG_PIR / (n_i * n_r)
    )
    return SIG_P / (SIG_P + err)


class GandDStudy(Scene):
    """G-study estimates components; D-study designs the test plan."""

    def construct(self):
        self.camera.background_color = BG
        self.play_title()
        self.play_g_study()
        self.play_d_study()
        self.play_tradeoff()
        self.play_takeaway()

    # ================================================================
    #  Title
    # ================================================================
    def play_title(self):
        title = Text("Designing for Generalizability", font_size=44,
                     color=WHITE, weight=BOLD)
        subtitle = Text("From variance components to a test plan",
                        font_size=24, color=TEXT2)
        subtitle.next_to(title, DOWN, buff=0.35)
        line = Line(LEFT * 2.5, RIGHT * 2.5, color=ACCENT,
                    stroke_width=1.5)
        line.next_to(subtitle, DOWN, buff=0.3)
        group = VGroup(title, subtitle, line)

        self.play(FadeIn(title, shift=UP * 0.2), run_time=0.8)
        self.play(FadeIn(subtitle), Create(line), run_time=0.8)
        self.wait(2.0)
        self.play(FadeOut(group, shift=UP * 0.4), run_time=0.7)

    # ================================================================
    #  Act 1 — G-study: estimate the variance components
    # ================================================================
    def play_g_study(self):
        header = Text("G-study: estimate the components", font_size=28,
                      color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.35)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.5)

        # ── a small labeled "cube" of three facets ──
        origin = LEFT * 4.3 + DOWN * 0.4
        ax_len = 1.5
        # three facet arrows: model (p), item (i), rater (r)
        arr_p = Arrow(origin, origin + RIGHT * ax_len, color=PAL[0],
                      stroke_width=4, buff=0,
                      max_tip_length_to_length_ratio=0.18)
        arr_i = Arrow(origin, origin + UP * ax_len, color=PAL[1],
                      stroke_width=4, buff=0,
                      max_tip_length_to_length_ratio=0.18)
        arr_r = Arrow(origin, origin + (UP + RIGHT) * 0.62 * ax_len,
                      color=PAL[3], stroke_width=4, buff=0,
                      max_tip_length_to_length_ratio=0.18)
        lbl_p = Text("model", font_size=18, color=PAL[0])
        lbl_p.next_to(arr_p, DOWN, buff=0.12)
        lbl_i = Text("item", font_size=18, color=PAL[1])
        lbl_i.next_to(arr_i, UP, buff=0.12)
        lbl_r = Text("rater", font_size=18, color=PAL[3])
        lbl_r.next_to(arr_r.get_end(), RIGHT, buff=0.1)

        cube = VGroup(arr_p, arr_i, arr_r, lbl_p, lbl_i, lbl_r)
        crossed = Text("crossed design", font_size=16, color=TEXT2)
        crossed.next_to(cube, DOWN, buff=0.55)

        self.play(GrowArrow(arr_p), GrowArrow(arr_i), GrowArrow(arr_r),
                  run_time=0.8)
        self.play(FadeIn(lbl_p), FadeIn(lbl_i), FadeIn(lbl_r),
                  FadeIn(crossed), run_time=0.6)
        self.wait(1.0)

        # ── bar chart of the seven variance components ──
        comps = [
            (r"\sigma_p^2", SIG_P, PAL[0]),
            (r"\sigma_i^2", SIG_I, PAL[1]),
            (r"\sigma_r^2", SIG_R, PAL[2]),
            (r"\sigma_{pi}^2", SIG_PI, PAL[1]),
            (r"\sigma_{pr}^2", SIG_PR, PAL[3]),
            (r"\sigma_{ir}^2", 0.15, PAL[2]),
            (r"\sigma_{pir}^2", SIG_PIR, PAL[4]),
        ]
        chart_base = DOWN * 1.9 + RIGHT * 1.0
        bar_w = 0.55
        gap = 0.78
        max_h = 2.6
        max_v = max(v for _, v, _ in comps)

        bars = VGroup()
        labels = VGroup()
        for k, (tex, val, color) in enumerate(comps):
            h = max_h * val / max_v
            x = chart_base + RIGHT * (k * gap)
            bar = Rectangle(width=bar_w, height=h, color=color,
                            fill_color=color, fill_opacity=0.85,
                            stroke_width=1)
            bar.move_to(x + UP * (h / 2))
            lbl = MathTex(tex, font_size=22, color=color)
            lbl.next_to(bar, UP, buff=0.1)
            bars.add(bar)
            labels.add(lbl)

        baseline = Line(
            bars[0].get_corner(DL) + LEFT * 0.15,
            bars[-1].get_corner(DR) + RIGHT * 0.15,
            color=AXIS_CLR, stroke_width=1.5,
        )
        caption = Text("estimated from data", font_size=18, color=TEXT2)
        caption.next_to(baseline, DOWN, buff=0.25)

        self.play(Create(baseline), run_time=0.4)
        self.play(
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars],
                        lag_ratio=0.12),
            run_time=1.6,
        )
        self.play(LaggedStart(*[FadeIn(l) for l in labels],
                              lag_ratio=0.12), run_time=1.0)
        self.play(FadeIn(caption, shift=UP * 0.1), run_time=0.5)
        self.wait(1.5)

        # highlight the dominant person x rater interaction
        box = SurroundingRectangle(VGroup(bars[4], labels[4]),
                                   color=ACCENT, stroke_width=2.5,
                                   buff=0.08)
        note = Text("rater interaction dominates", font_size=16,
                    color=ACCENT)
        note.next_to(box, RIGHT, buff=0.3).shift(UP * 0.6)
        self.play(Create(box), FadeIn(note), run_time=0.7)
        self.wait(2.0)

        self.play(FadeOut(VGroup(
            header, cube, crossed, bars, labels, baseline, caption,
            box, note,
        )), run_time=0.8)

    # ================================================================
    #  Act 2 — D-study: how many items and raters?
    # ================================================================
    def _build_heatmap(self):
        """Return (cells VGroup, indexer dict) for the D-study grid."""
        cell = 0.9
        x0 = -3.2
        y0 = -2.0
        lo = ManimColor("#2a2a55")
        hi = ManimColor(PAL[1])
        cells = VGroup()
        cell_at = {}
        for ri, n_r in enumerate(N_RATERS):
            for ii, n_i in enumerate(N_ITEMS):
                g = reliability(n_i, n_r)
                t = np.clip((g - 0.55) / (0.98 - 0.55), 0, 1)
                color = interpolate_color(lo, hi, t)
                sq = Square(side_length=cell, stroke_width=1,
                            stroke_color="#111111",
                            fill_color=color, fill_opacity=0.95)
                sq.move_to(RIGHT * (x0 + ii * cell)
                           + UP * (y0 + ri * cell))
                cells.add(sq)
                cell_at[(ii, ri)] = sq
        return cells, cell_at, cell, x0, y0

    def play_d_study(self):
        header = Text("D-study: how many items and raters?",
                      font_size=28, color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.35)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.5)

        cells, cell_at, cell, x0, y0 = self._build_heatmap()
        self.heatmap = cells
        self.cell_at = cell_at

        # axis labels
        x_axis_lbl = Text("number of items", font_size=18, color=AXIS_CLR)
        x_axis_lbl.next_to(cells, DOWN, buff=0.45)
        y_axis_lbl = Text("number of raters", font_size=18, color=AXIS_CLR)
        y_axis_lbl.rotate(PI / 2)
        y_axis_lbl.next_to(cells, LEFT, buff=0.45)

        x_ticks = VGroup()
        for ii, n_i in enumerate(N_ITEMS):
            t = Text(str(n_i), font_size=16, color=TEXT2)
            t.move_to(RIGHT * (x0 + ii * cell) + UP * (y0 - cell * 0.72))
            x_ticks.add(t)
        y_ticks = VGroup()
        for ri, n_r in enumerate(N_RATERS):
            t = Text(str(n_r), font_size=16, color=TEXT2)
            t.move_to(RIGHT * (x0 - cell * 0.72) + UP * (y0 + ri * cell))
            y_ticks.add(t)

        self.play(
            LaggedStart(*[FadeIn(c, scale=0.7) for c in cells],
                        lag_ratio=0.015),
            run_time=1.8,
        )
        self.play(FadeIn(x_axis_lbl), FadeIn(y_axis_lbl),
                  FadeIn(x_ticks), FadeIn(y_ticks), run_time=0.6)
        self.wait(1.0)

        # ── G = 0.90 contour as a highlighted set of cells ──
        # for each rater row, find the first item count that clears target
        contour_cells = VGroup()
        for ri, n_r in enumerate(N_RATERS):
            for ii, n_i in enumerate(N_ITEMS):
                if reliability(n_i, n_r) >= TARGET_G:
                    contour_cells.add(SurroundingRectangle(
                        cell_at[(ii, ri)], color=ACCENT,
                        stroke_width=2.5, buff=-0.02,
                    ))
                    break
        contour_lbl = MathTex(r"G = 0.90", font_size=24, color=ACCENT)
        contour_lbl.next_to(cells, RIGHT, buff=0.55).shift(UP * 0.6)

        self.play(
            LaggedStart(*[Create(c) for c in contour_cells],
                        lag_ratio=0.15),
            FadeIn(contour_lbl),
            run_time=1.6,
        )
        self.contour_lbl = contour_lbl
        caption = Text("cells past the line meet the target",
                       font_size=16, color=TEXT2)
        caption.next_to(cells, RIGHT, buff=0.55).shift(DOWN * 0.2)
        self.play(FadeIn(caption), run_time=0.5)
        self.wait(2.0)

        self.contour_cells = contour_cells
        self.d_static = VGroup(header, x_axis_lbl, y_axis_lbl,
                               x_ticks, y_ticks, caption)
        self.play(FadeOut(caption), FadeOut(contour_lbl), run_time=0.5)

    # ================================================================
    #  Act 3 — Trade-off: add raters vs add items
    # ================================================================
    def play_tradeoff(self):
        header = Text("Trade items against raters", font_size=28,
                      color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.35)
        self.play(
            ReplacementTransform(self.d_static[0], header),
            run_time=0.6,
        )

        cell_at = self.cell_at

        # coefficient reminder
        coef = MathTex(
            r"G = \frac{\sigma_p^2}{\sigma_p^2 + \sigma_{pi}^2/n_i + \sigma_{pr}^2/n_r}",
            font_size=26, color=ACCENT,
        )
        coef.next_to(self.heatmap, RIGHT, buff=0.55).shift(UP * 1.1)
        self.play(Write(coef), run_time=1.0)
        self.wait(1.0)

        # start point: few items, few raters (bottom-left-ish)
        start = (0, 0)  # n_i=5, n_r=1
        start_dot = Dot(cell_at[start].get_center(), color=WHITE,
                        radius=0.1)
        start_lbl = Text("start", font_size=16, color=WHITE)
        start_lbl.next_to(start_dot, DOWN, buff=0.12)
        self.play(FadeIn(start_dot, scale=0.5), FadeIn(start_lbl),
                  run_time=0.5)
        self.wait(0.5)

        # candidate A: move RIGHT (more items) until target met at n_r=1
        a_target = None
        for ii, n_i in enumerate(N_ITEMS):
            if reliability(n_i, N_RATERS[0]) >= TARGET_G:
                a_target = (ii, 0)
                break
        # candidate B: move UP (more raters) until target met at n_i=5
        b_target = None
        for ri, n_r in enumerate(N_RATERS):
            if reliability(N_ITEMS[0], n_r) >= TARGET_G:
                b_target = (0, ri)
                break

        dot_a = Dot(cell_at[start].get_center(), color=PAL[0], radius=0.09)
        dot_b = Dot(cell_at[start].get_center(), color=PAL[3], radius=0.09)
        self.add(dot_a, dot_b)

        if a_target is not None:
            path_a = Arrow(
                cell_at[start].get_center(),
                cell_at[a_target].get_center(),
                color=PAL[0], stroke_width=3, buff=0.1,
                max_tip_length_to_length_ratio=0.12,
            )
            lbl_a = Text("more items", font_size=16, color=PAL[0])
            lbl_a.next_to(path_a, UP, buff=0.1)
            self.play(GrowArrow(path_a),
                      dot_a.animate.move_to(cell_at[a_target].get_center()),
                      FadeIn(lbl_a), run_time=1.2)
        else:
            # never reaches target on this row — show it stalls at the edge
            path_a = Arrow(
                cell_at[start].get_center(),
                cell_at[(len(N_ITEMS) - 1, 0)].get_center(),
                color=PAL[0], stroke_width=3, buff=0.1,
                max_tip_length_to_length_ratio=0.12,
            )
            lbl_a = Text("items alone fall short", font_size=16,
                         color=PAL[0])
            lbl_a.next_to(path_a, UP, buff=0.1)
            self.play(GrowArrow(path_a),
                      dot_a.animate.move_to(
                          cell_at[(len(N_ITEMS) - 1, 0)].get_center()),
                      FadeIn(lbl_a), run_time=1.2)
        self.wait(0.8)

        b_end = b_target if b_target is not None else (0, len(N_RATERS) - 1)
        path_b = Arrow(
            cell_at[start].get_center(),
            cell_at[b_end].get_center(),
            color=PAL[3], stroke_width=3, buff=0.1,
            max_tip_length_to_length_ratio=0.14,
        )
        lbl_b = Text("more raters", font_size=16, color=PAL[3])
        lbl_b.next_to(path_b, LEFT, buff=0.1)
        self.play(GrowArrow(path_b),
                  dot_b.animate.move_to(cell_at[b_end].get_center()),
                  FadeIn(lbl_b), run_time=1.2)
        self.wait(1.0)

        # mark the cheaper path in ACCENT — rater variance dominates here,
        # so adding raters reaches the target sooner
        cheaper = Text("rater variance dominates: adding raters is cheaper",
                       font_size=18, color=ACCENT)
        cheaper.next_to(self.heatmap, DOWN, buff=1.1)
        glow = SurroundingRectangle(path_b, color=ACCENT,
                                    stroke_width=2, buff=0.05)
        self.play(Create(glow), path_b.animate.set_color(ACCENT),
                  lbl_b.animate.set_color(ACCENT),
                  FadeIn(cheaper, shift=UP * 0.1), run_time=0.9)
        self.wait(2.5)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)

    # ================================================================
    #  Takeaway
    # ================================================================
    def play_takeaway(self):
        heading = Text("Trade facets against each other",
                       font_size=36, color=WHITE, weight=BOLD)
        heading.shift(UP * 1.0)
        sub = Text("to hit a target reliability.", font_size=30,
                   color=TEXT2)
        sub.next_to(heading, DOWN, buff=0.3)

        rows = []
        items = [
            ("G-study", PAL[1], "estimate the variance components"),
            ("D-study", PAL[0], "choose items and raters for a target"),
            ("Trade-off", PAL[3], "add raters when rater variance dominates"),
        ]
        for label, color, desc in items:
            lbl = Text(label, font_size=24, color=color, weight=BOLD)
            dash = Text("—", font_size=22, color=TEXT2)
            desc_mob = Text(desc, font_size=20, color=TEXT2)
            row = VGroup(lbl, dash, desc_mob).arrange(RIGHT, buff=0.2)
            rows.append(row)
        row_group = VGroup(*rows).arrange(DOWN, buff=0.4,
                                          aligned_edge=LEFT)
        row_group.next_to(sub, DOWN, buff=0.55)

        line = Line(LEFT * 2, RIGHT * 2, color=ACCENT, stroke_width=1.5)
        line.next_to(row_group, DOWN, buff=0.45)

        source = Text("AIMS — Chapter 5: Reliability",
                      font_size=18, color=ManimColor("#444444"))
        source.next_to(line, DOWN, buff=0.3)

        self.play(FadeIn(heading, shift=DOWN * 0.15), run_time=0.7)
        self.play(FadeIn(sub), run_time=0.4)
        for row in rows:
            self.play(FadeIn(row, shift=RIGHT * 0.15), run_time=0.45)
            self.wait(0.3)
        self.play(Create(line), FadeIn(source), run_time=0.5)
        self.wait(3.5)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
