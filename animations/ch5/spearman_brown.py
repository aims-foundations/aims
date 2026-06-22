"""
AIMS Chapter 5 Animation: Back to Classical Coefficients
Cronbach's alpha & split-half recover the same reliability; Spearman-Brown
shows longer tests are more reliable (with diminishing returns); SEM bounds
how much a score would wobble on re-testing.

Run with:
    PATH="/lfs/local/0/sttruong/miniconda3/bin:$PATH" \
    manim -qh --disable_caching --media_dir media/ch5 animations/ch5/spearman_brown.py SpearmanBrown
"""

from manim import *
import numpy as np

# ── design tokens ────────────────────────────────────────────────
ACCENT = "#FFD966"
BG = "#0f0f0f"
TEXT2 = "#aaaaaa"
PAL = ["#5B8DEE", "#45BF7C", "#F0A35C", "#E8637A", "#B07CD8"]
AXIS_CLR = "#888888"


def spearman_brown(k, rho1):
    """Reliability of a test k times as long, given base reliability rho1."""
    return (k * rho1) / (1 + (k - 1) * rho1)


class SpearmanBrown(Scene):
    """Classical reliability coefficients, test length, and SEM."""

    def construct(self):
        self.camera.background_color = BG
        self.play_title()
        self.play_coefficients()
        self.play_spearman_brown()
        self.play_sem()
        self.play_takeaway()

    # ================================================================
    #  Title
    # ================================================================
    def play_title(self):
        title = Text("Back to Classical Coefficients", font_size=44,
                     color=WHITE, weight=BOLD)
        subtitle = Text("When all you have is total scores",
                        font_size=24, color=TEXT2)
        subtitle.next_to(title, DOWN, buff=0.35)
        line = Line(LEFT * 2.5, RIGHT * 2.5, color=ACCENT,
                    stroke_width=1.5)
        line.next_to(subtitle, DOWN, buff=0.3)
        group = VGroup(title, subtitle, line)

        self.play(FadeIn(title, shift=UP * 0.2), run_time=0.8)
        self.play(FadeIn(subtitle), Create(line), run_time=0.6)
        self.wait(1.6)
        self.play(FadeOut(group, shift=UP * 0.4), run_time=0.7)

    # ================================================================
    #  Act 1: Classical coefficients recover the same reliability
    # ================================================================
    def play_coefficients(self):
        header = Text("Same Reliability, Two Estimators", font_size=28,
                      color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.4)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.5)

        note = Text(
            "From total scores alone, Cronbach's "
            "α and split-half\nrecover the same person-signal share.",
            font_size=22, color=TEXT2, line_spacing=0.9,
        )
        note.next_to(header, DOWN, buff=0.5)
        self.play(FadeIn(note, shift=UP * 0.1), run_time=0.6)
        self.wait(1.5)

        alpha = MathTex(
            r"\alpha = \frac{k}{k-1}\left(1 - "
            r"\frac{\sum_j \sigma_j^2}{\sigma_X^2}\right)",
            font_size=40, color=ACCENT,
        )
        alpha.next_to(note, DOWN, buff=0.7)
        self.play(Write(alpha), run_time=1.2)
        self.wait(1.2)

        caption = Text("the person (signal) share, recovered from scores",
                       font_size=20, color=TEXT2)
        caption.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(caption, shift=UP * 0.1), run_time=0.5)
        self.wait(3.0)

        self.play(FadeOut(VGroup(header, note, alpha, caption)),
                  run_time=0.7)

    # ================================================================
    #  Act 2: Spearman-Brown — longer test, more reliable
    # ================================================================
    def play_spearman_brown(self):
        header = Text("Spearman–Brown: Lengthen the Test",
                      font_size=28, color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.4)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.5)

        rho1 = 0.1  # base single-item reliability

        ax = Axes(
            x_range=[0, 40, 10], y_range=[0, 1.0, 0.25],
            x_length=8.5, y_length=4.0,
            axis_config={
                "color": AXIS_CLR, "include_numbers": True,
                "font_size": 20,
            },
            tips=False,
        )
        x_lab = ax.get_x_axis_label(
            Text("number of items (k)", font_size=18, color=AXIS_CLR),
            edge=RIGHT, direction=DOWN,
        ).shift(DOWN * 0.35)
        y_lab = ax.get_y_axis_label(
            MathTex(r"\rho_k", font_size=26, color=AXIS_CLR),
            edge=UP, direction=LEFT,
        ).shift(UP * 0.3 + LEFT * 0.1)
        ax_group = VGroup(ax, x_lab, y_lab)
        ax_group.move_to(DOWN * 0.5)
        self.play(Create(ax), FadeIn(x_lab), FadeIn(y_lab), run_time=0.8)

        # Asymptote line at rho = 1
        asymptote = DashedLine(
            ax.c2p(0, 1.0), ax.c2p(40, 1.0),
            color=TEXT2, stroke_width=1.5, dash_length=0.08,
        )
        asym_lbl = MathTex(r"\rho \to 1", font_size=20, color=TEXT2)
        asym_lbl.next_to(ax.c2p(40, 1.0), UP + LEFT, buff=0.1)
        self.play(Create(asymptote), FadeIn(asym_lbl), run_time=0.6)

        # Spearman-Brown curve, drawn as the test lengthens
        curve = ax.plot(
            lambda k: spearman_brown(k, rho1),
            x_range=[1, 40, 0.2], color=PAL[1], stroke_width=4,
        )
        self.play(Create(curve), run_time=2.5)

        formula = MathTex(
            r"\rho_k = \frac{k\,\rho_1}{1 + (k-1)\rho_1}",
            font_size=34, color=ACCENT,
        )
        formula.next_to(header, DOWN, buff=0.2).shift(RIGHT * 2.4)
        self.play(Write(formula), run_time=1.0)
        self.wait(1.0)

        # Mark diminishing returns: a dot early and a dot late
        k_early, k_late = 8, 32
        dot_early = Dot(
            ax.c2p(k_early, spearman_brown(k_early, rho1)),
            color=PAL[3], radius=0.07,
        )
        dot_late = Dot(
            ax.c2p(k_late, spearman_brown(k_late, rho1)),
            color=PAL[3], radius=0.07,
        )
        self.play(FadeIn(dot_early), FadeIn(dot_late), run_time=0.6)

        flatten = Text("diminishing returns", font_size=18, color=PAL[3])
        flatten.next_to(dot_late, DOWN + RIGHT, buff=0.15)
        self.play(FadeIn(flatten, shift=UP * 0.1), run_time=0.5)
        self.wait(1.0)

        caption = Text("more items, more signal",
                       font_size=20, color=TEXT2)
        caption.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(caption, shift=UP * 0.1), run_time=0.5)
        self.wait(2.5)

        self.play(FadeOut(VGroup(
            header, ax_group, asymptote, asym_lbl, curve, formula,
            dot_early, dot_late, flatten, caption,
        )), run_time=0.7)

    # ================================================================
    #  Act 3: Standard error of measurement
    # ================================================================
    def play_sem(self):
        header = Text("Standard Error of Measurement", font_size=28,
                      color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.4)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.5)

        formula = MathTex(
            r"\mathrm{SEM} = \sigma_X\sqrt{1-\rho}",
            font_size=40, color=ACCENT,
        )
        formula.next_to(header, DOWN, buff=0.5)
        self.play(Write(formula), run_time=1.0)
        self.wait(1.0)

        # Number line for the observed score
        nl = NumberLine(
            x_range=[0, 100, 20], length=9,
            color=AXIS_CLR, include_numbers=True, font_size=20,
        )
        nl.move_to(DOWN * 0.3)
        nl_lbl = Text("observed score", font_size=18, color=AXIS_CLR)
        nl_lbl.next_to(nl, DOWN, buff=0.55)
        self.play(Create(nl), FadeIn(nl_lbl), run_time=0.8)

        # Point estimate with a shaded +/- SEM band
        x_hat, sem = 72, 6
        band = Rectangle(
            width=nl.n2p(x_hat + sem)[0] - nl.n2p(x_hat - sem)[0],
            height=0.7,
            stroke_width=0, fill_color=PAL[0], fill_opacity=0.28,
        )
        band.move_to(nl.n2p(x_hat))
        err_bar = Line(
            nl.n2p(x_hat - sem), nl.n2p(x_hat + sem),
            color=PAL[0], stroke_width=3,
        )
        cap_l = Line(UP * 0.12, DOWN * 0.12, color=PAL[0],
                     stroke_width=3).move_to(nl.n2p(x_hat - sem))
        cap_r = Line(UP * 0.12, DOWN * 0.12, color=PAL[0],
                     stroke_width=3).move_to(nl.n2p(x_hat + sem))
        point = Dot(nl.n2p(x_hat), color=ACCENT, radius=0.09)

        point_lbl = MathTex(r"\hat{X} = 72", font_size=24, color=ACCENT)
        point_lbl.next_to(point, UP, buff=0.45)
        band_lbl = MathTex(r"\pm\,\mathrm{SEM}", font_size=22, color=PAL[0])
        band_lbl.next_to(band, UP, buff=0.12)

        self.play(FadeIn(band), Create(err_bar),
                  Create(cap_l), Create(cap_r), run_time=0.8)
        self.play(FadeIn(point), FadeIn(point_lbl),
                  FadeIn(band_lbl), run_time=0.6)
        self.wait(1.5)

        caption = Text(
            "how much a score would wobble on re-testing",
            font_size=20, color=TEXT2,
        )
        caption.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(caption, shift=UP * 0.1), run_time=0.5)
        self.wait(3.0)

        self.play(FadeOut(VGroup(
            header, formula, nl, nl_lbl, band, err_bar,
            cap_l, cap_r, point, point_lbl, band_lbl, caption,
        )), run_time=0.7)

    # ================================================================
    #  Takeaway
    # ================================================================
    def play_takeaway(self):
        heading = Text(
            "Lengthen the test to raise reliability",
            font_size=34, color=WHITE, weight=BOLD,
        )
        heading.shift(UP * 0.5)
        sub = Text("— with diminishing returns.",
                   font_size=28, color=ACCENT)
        sub.next_to(heading, DOWN, buff=0.4)

        line = Line(LEFT * 2.5, RIGHT * 2.5, color=ACCENT,
                    stroke_width=1.5)
        line.next_to(sub, DOWN, buff=0.5)

        source = Text("AIMS — Chapter 5: Reliability",
                      font_size=18, color=ManimColor("#444444"))
        source.next_to(line, DOWN, buff=0.4)

        self.play(FadeIn(heading, shift=DOWN * 0.15), run_time=0.7)
        self.play(FadeIn(sub, shift=UP * 0.1), run_time=0.6)
        self.play(Create(line), run_time=0.4)
        self.play(FadeIn(source), run_time=0.5)
        self.wait(3.0)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
