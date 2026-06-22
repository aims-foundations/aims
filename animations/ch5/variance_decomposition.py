"""
AIMS Chapter 5 Animation: Decomposing the Variance
The four-term variance decomposition of a benchmark score, and reliability
as the person (signal) share of that variance.

Run with:
    PATH="/lfs/local/0/sttruong/miniconda3/bin:$PATH" \
    manim -qh --disable_caching --media_dir media/ch5 \
        animations/ch5/variance_decomposition.py VarianceDecomposition
"""

from manim import *
import numpy as np

# ── design tokens ────────────────────────────────────────────────
ACCENT = "#FFD966"
BG = "#0f0f0f"
TEXT2 = "#aaaaaa"
PAL = ["#5B8DEE", "#45BF7C", "#F0A35C", "#E8637A", "#B07CD8"]
AXIS_CLR = "#888888"


class VarianceDecomposition(Scene):
    """Where does a benchmark score's variability come from?"""

    def construct(self):
        self.camera.background_color = BG
        self.play_title()
        self.play_response_model()
        self.play_decomposition()
        self.play_reliability()
        self.play_takeaway()

    # ================================================================
    #  Title
    # ================================================================
    def play_title(self):
        title = Text("Decomposing the Variance", font_size=44,
                     color=WHITE, weight=BOLD)
        sub = Text("Where does a benchmark score's variability come from?",
                   font_size=24, color=TEXT2)
        sub.next_to(title, DOWN, buff=0.35)
        line = Line(LEFT * 2.5, RIGHT * 2.5, color=ACCENT,
                    stroke_width=1.5)
        line.next_to(sub, DOWN, buff=0.3)
        group = VGroup(title, sub, line)

        self.play(FadeIn(title, shift=UP * 0.2), run_time=0.8)
        self.play(FadeIn(sub), Create(line), run_time=0.8)
        self.wait(2.0)
        self.play(FadeOut(group, shift=UP * 0.4), run_time=0.7)

    # ================================================================
    #  Act 1: Response matrix → one cell → log-odds model
    # ================================================================
    def play_response_model(self):
        header = Text("The Response Model", font_size=28,
                      color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.35)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.5)

        # ── Build a small response matrix Y_{ij} as a grid of cells ──
        n_rows, n_cols = 4, 5
        rng = np.random.default_rng(7)
        # 1 = correct (green), 0 = incorrect (dark)
        Y = rng.integers(0, 2, size=(n_rows, n_cols))
        # guarantee the highlighted cell (1,2) is "correct" for the model
        Y[1, 2] = 1

        cell = 0.6
        cells = VGroup()
        cell_map = {}
        for i in range(n_rows):
            for j in range(n_cols):
                correct = Y[i, j] == 1
                sq = Square(side_length=cell)
                sq.set_stroke(BG, width=2)
                sq.set_fill(PAL[1] if correct else "#3a3a3a",
                            opacity=1.0)
                sq.move_to(np.array([j * cell, -i * cell, 0]))
                cells.add(sq)
                cell_map[(i, j)] = sq
        cells.move_to(LEFT * 2.7 + UP * 0.2)

        matrix_lbl = MathTex(r"Y_{ij}", font_size=30, color=TEXT2)
        matrix_lbl.next_to(cells, UP, buff=0.25)
        row_lbl = Text("models", font_size=16, color=TEXT2)
        row_lbl.rotate(PI / 2).next_to(cells, LEFT, buff=0.2)
        col_lbl = Text("items", font_size=16, color=TEXT2)
        col_lbl.next_to(cells, DOWN, buff=0.2)

        self.play(
            LaggedStart(*[FadeIn(c, scale=0.6) for c in cells],
                        lag_ratio=0.03),
            FadeIn(matrix_lbl), FadeIn(row_lbl), FadeIn(col_lbl),
            run_time=1.4,
        )
        self.wait(1.0)

        # ── Pull ONE cell out ──
        target = cell_map[(1, 2)]
        big = target.copy()
        big.generate_target()
        big.target.scale(1.9).move_to(RIGHT * 2.7 + UP * 1.4)
        ring = SurroundingRectangle(target, color=ACCENT, buff=0.04,
                                    stroke_width=2.5)
        self.play(Create(ring), run_time=0.4)
        self.play(MoveToTarget(big), run_time=0.9)

        cell_tag = MathTex(r"Y_{ij}", font_size=26, color=ACCENT)
        cell_tag.next_to(big, UP, buff=0.15)
        self.play(FadeIn(cell_tag), run_time=0.4)

        # ── Log-odds model for that cell ──
        model = MathTex(
            r"\eta_{ij} = \theta_i - \beta_j + \gamma_{ij}",
            font_size=34, color=ACCENT,
        )
        model.move_to(RIGHT * 2.7 + DOWN * 0.2)
        self.play(Write(model), run_time=1.1)
        self.wait(0.6)

        # ── Annotate the three terms (small TEXT2 labels) ──
        ann = VGroup(
            MathTex(r"\theta_i", r"\;\;\mathrm{ability}",
                    font_size=22).set_color_by_tex(r"\theta_i", PAL[0]),
            MathTex(r"\beta_j", r"\;\;\mathrm{difficulty}",
                    font_size=22).set_color_by_tex(r"\beta_j", PAL[2]),
            MathTex(r"\gamma_{ij}", r"\;\;\mathrm{person}\times\mathrm{item}",
                    font_size=22).set_color_by_tex(r"\gamma_{ij}", PAL[4]),
        )
        for a in ann:
            a[1].set_color(TEXT2)
        ann.arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        ann.next_to(model, DOWN, buff=0.4)
        self.play(LaggedStart(*[FadeIn(a, shift=RIGHT * 0.1) for a in ann],
                              lag_ratio=0.3), run_time=1.3)
        self.wait(2.5)

        self.play(FadeOut(VGroup(
            header, cells, matrix_lbl, row_lbl, col_lbl, ring,
            big, cell_tag, model, ann,
        )), run_time=0.8)

    # ================================================================
    #  Act 2: Four-term stacked bar separating out of the total
    # ================================================================
    def play_decomposition(self):
        header = Text("Four Sources of Variance", font_size=28,
                      color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.35)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.5)

        # The equation
        eqn = MathTex(
            r"\mathrm{Var}(Y) = V_{\mathrm{person}} + V_{\mathrm{item}}"
            r" + V_{\mathrm{int}} + V_{\mathrm{resid}}",
            font_size=30, color=ACCENT,
        )
        eqn.next_to(header, DOWN, buff=0.35)
        self.play(Write(eqn), run_time=1.2)
        self.wait(0.8)

        # Stacked bar geometry. For a single binary trial the person
        # (signal) share is SMALL relative to item + residual.
        total_w = 9.0
        height = 0.85
        # fractions: person small; item moderate; interaction small; residual large
        fracs = [0.14, 0.26, 0.12, 0.48]
        names = [
            r"V_{\mathrm{person}}",
            r"V_{\mathrm{item}}",
            r"V_{\mathrm{int}}",
            r"V_{\mathrm{resid}}",
        ]
        sublabels = ["signal", "item difficulty",
                     "interaction", "Bernoulli noise"]
        colors = [PAL[0], PAL[2], PAL[4], PAL[3]]

        bar_y = 0.5
        segs = VGroup()
        x_left = -total_w / 2
        x = x_left
        seg_list = []
        for frac, col in zip(fracs, colors):
            w = frac * total_w
            seg = Rectangle(width=w, height=height,
                            stroke_color=BG, stroke_width=2,
                            fill_color=col, fill_opacity=1.0)
            seg.move_to(np.array([x + w / 2, bar_y, 0]))
            segs.add(seg)
            seg_list.append(seg)
            x += w

        total_lbl = MathTex(r"\mathrm{Var}(Y)", font_size=24, color=TEXT2)
        total_lbl.next_to(segs, UP, buff=0.2)

        # Reveal the bar as a single total, then color-split.
        whole = Rectangle(width=total_w, height=height,
                          stroke_color=AXIS_CLR, stroke_width=2,
                          fill_color="#444444", fill_opacity=1.0)
        whole.move_to(np.array([0, bar_y, 0]))
        self.play(GrowFromEdge(whole, LEFT), FadeIn(total_lbl),
                  run_time=0.9)
        self.wait(0.5)
        self.play(FadeOut(whole), FadeIn(segs), run_time=0.8)

        # Build labels + brackets under each segment.
        labels = VGroup()
        for seg, name, sub, col in zip(seg_list, names, sublabels, colors):
            brace = Line(
                seg.get_corner(DL) + DOWN * 0.08,
                seg.get_corner(DR) + DOWN * 0.08,
                color=col, stroke_width=2.5,
            )
            tex = MathTex(name, font_size=22, color=col)
            tex.next_to(brace, DOWN, buff=0.12)
            subt = Text(sub, font_size=13, color=TEXT2)
            subt.next_to(tex, DOWN, buff=0.08)
            labels.add(VGroup(brace, tex, subt))
        self.play(LaggedStart(*[FadeIn(l, shift=DOWN * 0.1) for l in labels],
                              lag_ratio=0.2), run_time=1.4)
        self.wait(1.0)

        # Animate the segments separating apart, then back together.
        gaps = [LEFT * 0.45, LEFT * 0.15, RIGHT * 0.15, RIGHT * 0.45]
        anims = []
        for grp_idx, (seg, lab) in enumerate(zip(seg_list, labels)):
            anims.append(seg.animate.shift(gaps[grp_idx] + UP * 0.0))
            anims.append(lab.animate.shift(gaps[grp_idx]))
        self.play(*anims, run_time=1.0)
        self.wait(0.5)

        note = Text(
            "A single binary trial: the signal is only a sliver",
            font_size=22, color=ACCENT,
        )
        note.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(note, shift=UP * 0.1), run_time=0.6)
        self.wait(2.5)

        # Re-merge segments for handoff to Act 3 (store reassembled state).
        anims_back = []
        for grp_idx, (seg, lab) in enumerate(zip(seg_list, labels)):
            anims_back.append(seg.animate.shift(-gaps[grp_idx]))
            anims_back.append(lab.animate.shift(-gaps[grp_idx]))
        self.play(*anims_back, FadeOut(note), run_time=0.8)

        self.decomp_keep = VGroup(eqn, segs, total_lbl, labels)
        self.decomp_segs = seg_list
        self.decomp_labels = labels
        self.decomp_header = header

    # ================================================================
    #  Act 3: Highlight person segment → reliability coefficient
    # ================================================================
    def play_reliability(self):
        new_header = Text("Reliability = Signal Share", font_size=28,
                          color=WHITE, weight=BOLD)
        new_header.to_edge(UP, buff=0.35)
        self.play(
            FadeTransform(self.decomp_header, new_header),
            run_time=0.6,
        )

        # Dim everything except V_person (segment 0).
        dim_anims = []
        for idx, seg in enumerate(self.decomp_segs):
            if idx == 0:
                dim_anims.append(seg.animate.set_fill(opacity=1.0))
            else:
                dim_anims.append(seg.animate.set_fill(opacity=0.18))
        for idx, lab in enumerate(self.decomp_labels):
            if idx != 0:
                dim_anims.append(lab.animate.set_opacity(0.25))
        self.play(*dim_anims, run_time=0.9)

        # Ring around the person segment.
        ring = SurroundingRectangle(self.decomp_segs[0], color=ACCENT,
                                    buff=0.05, stroke_width=2.5)
        self.play(Create(ring), run_time=0.5)
        self.wait(0.5)

        # Reliability / generalizability coefficient.
        rho = MathTex(
            r"\rho = \frac{V_{\mathrm{person}}}{\mathrm{Var}(Y)}",
            font_size=40, color=ACCENT,
        )
        rho.move_to(DOWN * 1.4)
        self.play(Write(rho), run_time=1.1)

        caption = Text(
            "the signal share — genuine differences between models",
            font_size=22, color=TEXT2,
        )
        caption.next_to(rho, DOWN, buff=0.35)
        self.play(FadeIn(caption, shift=UP * 0.1), run_time=0.6)
        self.wait(3.0)

        self.play(FadeOut(VGroup(
            new_header, self.decomp_keep, ring, rho, caption,
        )), run_time=0.8)

    # ================================================================
    #  Takeaway
    # ================================================================
    def play_takeaway(self):
        msg = Text(
            "Reliability is the share of variance that is signal.",
            font_size=34, color=WHITE, weight=BOLD,
        )
        line = Line(LEFT * 2.5, RIGHT * 2.5, color=ACCENT,
                    stroke_width=1.5)
        line.next_to(msg, DOWN, buff=0.4)
        source = Text("AIMS — Chapter 5: Reliability",
                      font_size=18, color=ManimColor("#444444"))
        source.next_to(line, DOWN, buff=0.4)

        self.play(FadeIn(msg, shift=UP * 0.15), run_time=0.9)
        self.play(Create(line), run_time=0.5)
        self.play(FadeIn(source), run_time=0.5)
        self.wait(3.5)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
