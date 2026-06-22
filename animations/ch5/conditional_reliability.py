"""
AIMS Chapter 5 Animation: Conditional Reliability

Reliability is local: precision varies with ability. An easy item pool
informs low/mid-ability test takers well but collapses for frontier
(high-theta) models. A difficulty-matched pool stays reliable across
the whole ability range.

Run with:
    PATH="/lfs/local/0/sttruong/miniconda3/bin:$PATH" \
    manim -qh --disable_caching --media_dir media/ch5 \
        animations/ch5/conditional_reliability.py ConditionalReliability
"""

from manim import *
import numpy as np

# ── design tokens ────────────────────────────────────────────────
ACCENT = "#FFD966"
BG = "#0f0f0f"
TEXT2 = "#aaaaaa"
PAL = ["#5B8DEE", "#45BF7C", "#F0A35C", "#E8637A", "#B07CD8"]
AXIS_CLR = "#888888"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def info_at(theta, betas):
    """Test information I(theta) = sum_j p_j(1-p_j) over 2PL/1PL items."""
    total = 0.0
    for b in betas:
        p = sigmoid(theta - b)
        total += p * (1.0 - p)
    return total


def reliability(theta, betas, const):
    """rho(theta) = I(theta) / (I(theta) + const)."""
    info = info_at(theta, betas)
    return info / (info + const)


# Easy item pool: difficulties clustered low. Information piles up for
# low/mid theta and vanishes for high theta -> reliability collapses.
EASY_BETAS = np.array([-3.0, -2.6, -2.3, -2.0, -1.7, -1.4, -1.1,
                       -0.8, -0.5])
# Matched pool: difficulties spread across the whole ability range, so
# information stays roughly constant -> reliability stays high and flat.
MATCHED_BETAS = np.array([-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5,
                          0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])

EASY_CONST = 0.45
MATCHED_CONST = 0.32


class ConditionalReliability(Scene):
    """Reliability is local: easy pools collapse for frontier models."""

    def construct(self):
        self.camera.background_color = BG
        self.play_title()
        self.play_easy_pool()
        self.play_slide_marker()
        self.play_matched_pool()
        self.play_takeaway()

    # ================================================================
    #  Title
    # ================================================================
    def play_title(self):
        title = Text("Reliability Is Local", font_size=44,
                     color=WHITE, weight=BOLD)
        sub = Text("Precision varies with ability", font_size=24,
                   color=TEXT2)
        sub.next_to(title, DOWN, buff=0.35)
        line = Line(LEFT * 2.5, RIGHT * 2.5, color=ACCENT,
                    stroke_width=1.5)
        line.next_to(sub, DOWN, buff=0.3)
        group = VGroup(title, sub, line)

        self.play(FadeIn(title, shift=UP * 0.2), run_time=0.8)
        self.play(FadeIn(sub), Create(line), run_time=0.7)
        self.wait(1.6)
        self.play(FadeOut(group, shift=UP * 0.4), run_time=0.7)

    # ================================================================
    #  Shared axes builder
    # ================================================================
    def build_axes(self):
        ax = Axes(
            x_range=[-4, 4, 1], y_range=[0, 1.0, 0.25],
            x_length=9.5, y_length=4.2,
            axis_config={
                "color": AXIS_CLR, "include_numbers": True,
                "font_size": 20,
            },
            tips=False,
        )
        x_lab = ax.get_x_axis_label(
            MathTex(r"\theta", font_size=30, color=AXIS_CLR),
            edge=RIGHT, direction=DOWN,
        ).shift(RIGHT * 0.35 + DOWN * 0.1)
        y_lab = ax.get_y_axis_label(
            MathTex(r"\rho(\theta)", font_size=28, color=AXIS_CLR),
            edge=UP, direction=LEFT,
        ).shift(UP * 0.3)
        ax_group = VGroup(ax, x_lab, y_lab)
        ax_group.move_to(DOWN * 0.5)
        return ax, x_lab, y_lab, ax_group

    # ================================================================
    #  Act 1: easy item pool curve
    # ================================================================
    def play_easy_pool(self):
        header = Text("Conditional Reliability", font_size=28,
                      color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.35)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.5)

        ax, x_lab, y_lab, ax_group = self.build_axes()
        self.play(Create(ax), FadeIn(x_lab), FadeIn(y_lab), run_time=0.8)

        easy_curve = ax.plot(
            lambda x: reliability(x, EASY_BETAS, EASY_CONST),
            x_range=[-4, 4, 0.04], color=PAL[2], stroke_width=4,
        )
        easy_lbl = Text("easy item pool", font_size=20, color=PAL[2])
        easy_lbl.next_to(ax.c2p(-2.2, reliability(-2.2, EASY_BETAS,
                                                  EASY_CONST)),
                         UP, buff=0.2)

        self.play(Create(easy_curve), run_time=1.6)
        self.play(FadeIn(easy_lbl, shift=UP * 0.1), run_time=0.5)
        self.wait(2.0)

        # persist for later acts
        self.header = header
        self.ax = ax
        self.ax_group = ax_group
        self.easy_curve = easy_curve
        self.easy_lbl = easy_lbl

    # ================================================================
    #  Act 2: slide marker to high theta, show collapse
    # ================================================================
    def play_slide_marker(self):
        ax = self.ax
        theta = ValueTracker(-3.0)

        def rho_of(t):
            return reliability(t, EASY_BETAS, EASY_CONST)

        dot = always_redraw(lambda: Dot(
            ax.c2p(theta.get_value(), rho_of(theta.get_value())),
            color=ACCENT, radius=0.09,
        ))
        drop = always_redraw(lambda: DashedLine(
            ax.c2p(theta.get_value(), 0),
            ax.c2p(theta.get_value(), rho_of(theta.get_value())),
            color=ACCENT, stroke_width=1.5, dash_length=0.06,
        ))
        readout = always_redraw(lambda: MathTex(
            r"\rho = " + f"{rho_of(theta.get_value()):.2f}",
            font_size=24, color=ACCENT,
        ).next_to(ax.c2p(theta.get_value(),
                         rho_of(theta.get_value())),
                  UR, buff=0.12))

        self.play(FadeIn(dot), FadeIn(drop), FadeIn(readout),
                  run_time=0.5)
        self.wait(0.6)
        # slide from low to high ability
        self.play(theta.animate.set_value(3.2), run_time=3.0,
                  rate_func=smooth)
        self.wait(0.6)

        # frontier region marker
        frontier_line = DashedLine(
            ax.c2p(3.2, 0), ax.c2p(3.2, 1.0),
            color=PAL[3], stroke_width=2, dash_length=0.08,
        )
        frontier_lbl = Text("frontier models", font_size=20,
                            color=PAL[3])
        frontier_lbl.next_to(ax.c2p(3.2, 1.0), UP, buff=0.12)
        frontier_lbl.shift(LEFT * 0.6)
        self.play(Create(frontier_line), FadeIn(frontier_lbl),
                  run_time=0.8)

        caption = Text("easy items barely inform a strong model",
                       font_size=22, color=TEXT2)
        caption.to_edge(DOWN, buff=0.25)
        self.play(FadeIn(caption, shift=UP * 0.1), run_time=0.6)
        self.wait(2.8)

        # fade marker bits but keep curve + axes for the overlay
        self.play(FadeOut(dot), FadeOut(drop), FadeOut(readout),
                  FadeOut(caption), run_time=0.6)

        self.frontier_line = frontier_line
        self.frontier_lbl = frontier_lbl

    # ================================================================
    #  Act 3: overlay matched pool curve
    # ================================================================
    def play_matched_pool(self):
        ax = self.ax

        matched_curve = ax.plot(
            lambda x: reliability(x, MATCHED_BETAS, MATCHED_CONST),
            x_range=[-4, 4, 0.04], color=ACCENT, stroke_width=4,
        )
        matched_lbl = Text("matched pool", font_size=20, color=ACCENT)
        matched_lbl.next_to(
            ax.c2p(1.0, reliability(1.0, MATCHED_BETAS,
                                    MATCHED_CONST)),
            UP, buff=0.2)

        self.play(Create(matched_curve), run_time=1.6)
        self.play(FadeIn(matched_lbl, shift=UP * 0.1), run_time=0.5)
        self.wait(1.0)

        # highlight the gap at high theta
        hi = 3.2
        rho_easy = reliability(hi, EASY_BETAS, EASY_CONST)
        rho_matched = reliability(hi, MATCHED_BETAS, MATCHED_CONST)
        gap = DoubleArrow(
            ax.c2p(hi, rho_easy), ax.c2p(hi, rho_matched),
            color=WHITE, stroke_width=2.5, buff=0.05,
            tip_length=0.18,
        )
        gap_lbl = Text("gap", font_size=20, color=WHITE)
        gap_lbl.next_to(gap, RIGHT, buff=0.12)
        self.play(GrowFromCenter(gap), FadeIn(gap_lbl), run_time=0.8)
        self.wait(3.0)

        self.cleanup_group = VGroup(
            self.header, self.ax_group, self.easy_curve, self.easy_lbl,
            self.frontier_line, self.frontier_lbl,
            matched_curve, matched_lbl, gap, gap_lbl,
        )
        self.play(FadeOut(self.cleanup_group), run_time=0.8)

    # ================================================================
    #  Takeaway
    # ================================================================
    def play_takeaway(self):
        heading = Text(
            "Match item difficulty to the ability range you care about.",
            font_size=30, color=WHITE, weight=BOLD)
        heading.scale_to_fit_width(min(heading.width, 12))
        heading.shift(UP * 0.3)

        line = Line(LEFT * 3, RIGHT * 3, color=ACCENT, stroke_width=1.5)
        line.next_to(heading, DOWN, buff=0.5)

        source = Text("AIMS — Chapter 5: Reliability", font_size=18,
                      color=ManimColor("#444444"))
        source.next_to(line, DOWN, buff=0.4)

        self.play(FadeIn(heading, shift=DOWN * 0.15), run_time=0.8)
        self.play(Create(line), run_time=0.5)
        self.play(FadeIn(source), run_time=0.5)
        self.wait(3.5)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
