"""
AIMS Chapter 5 Animation: Three Estimators, One Estimand
Three routes to the same variance components (plug-in, Bayesian, method of
moments), plus an identifiability coda on replication. True sigma_p^2 = 1.0.

Run with:
    PATH="/lfs/local/0/sttruong/miniconda3/bin:$PATH" \
    manim -qh --disable_caching --media_dir media/ch5 \
        animations/ch5/three_estimators.py ThreeEstimators
"""

from manim import *
import numpy as np

# ── design tokens ────────────────────────────────────────────────
ACCENT = "#FFD966"
BG = "#0f0f0f"
TEXT2 = "#aaaaaa"
PAL = ["#5B8DEE", "#45BF7C", "#F0A35C", "#E8637A", "#B07CD8"]
AXIS_CLR = "#888888"


def gauss(x, mu, sigma):
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (
        sigma * np.sqrt(2 * np.pi)
    )


class ThreeEstimators(Scene):
    """Three routes to the same variance components, plus identifiability."""

    def construct(self):
        self.camera.background_color = BG
        self.play_title()
        self.play_plugin()        # Act 1
        self.play_bayesian()      # Act 2
        self.play_moments()       # Act 3
        self.play_identifiability()  # Act 4
        self.play_takeaway()

    # ================================================================
    #  Title
    # ================================================================
    def play_title(self):
        title = Text("Estimating the Components", font_size=44,
                     color=WHITE, weight=BOLD)
        sub = Text("Three routes to the same variance", font_size=24,
                   color=TEXT2)
        sub.next_to(title, DOWN, buff=0.35)
        line = Line(LEFT * 2.5, RIGHT * 2.5, color=ACCENT,
                    stroke_width=1.5)
        line.next_to(sub, DOWN, buff=0.3)
        g = VGroup(title, sub, line)

        self.play(FadeIn(title, shift=UP * 0.2), run_time=0.8)
        self.play(FadeIn(sub), Create(line), run_time=0.6)
        self.wait(1.6)
        self.play(FadeOut(g, shift=UP * 0.4), run_time=0.7)

    # ================================================================
    #  Act 1: Plug-in (fixed effects)
    # ================================================================
    def play_plugin(self):
        header = Text("Plug-in (fixed effects)", font_size=30,
                      color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.4)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.5)

        # Number line of fitted abilities
        nl = NumberLine(
            x_range=[-3, 3, 1], length=8, color=AXIS_CLR,
            include_numbers=True, font_size=20,
        )
        nl.shift(DOWN * 1.6)
        nl_lbl = MathTex(r"\hat\theta_i", font_size=24, color=AXIS_CLR)
        nl_lbl.next_to(nl, RIGHT, buff=0.2)
        self.play(Create(nl), FadeIn(nl_lbl), run_time=0.7)

        rng = np.random.default_rng(5)
        abil = rng.normal(0, 1.35, 14)  # noisy fitted abilities (overshoot)
        dots = VGroup(*[
            Dot(nl.n2p(a), radius=0.07, color=PAL[0], fill_opacity=0.85)
            for a in abil
        ])
        self.play(LaggedStartMap(FadeIn, dots, lag_ratio=0.06),
                  run_time=1.0)

        # Variance bar chart: true vs naive vs corrected, on a small axis
        bax = Axes(
            x_range=[0, 3, 1], y_range=[0, 1.8, 0.5],
            x_length=3.6, y_length=2.6,
            axis_config={"color": AXIS_CLR, "font_size": 18},
            y_axis_config={"include_numbers": True},
            tips=False,
        )
        bax.to_edge(LEFT, buff=0.7).shift(UP * 0.9)
        # true sigma_p^2 = 1 reference line
        true_y = bax.c2p(0, 1.0)[1]
        true_line = DashedLine(
            bax.c2p(0, 1.0), bax.c2p(3, 1.0),
            color=ACCENT, stroke_width=2, dash_length=0.08,
        )
        true_lbl = MathTex(r"\sigma_p^2 = 1", font_size=20, color=ACCENT)
        true_lbl.next_to(true_line, RIGHT, buff=0.1)

        naive_h, corr_h = 1.55, 1.02
        naive_bar = Rectangle(
            width=0.7, height=bax.c2p(1, naive_h)[1] - bax.c2p(1, 0)[1],
            color=PAL[0], fill_opacity=0.6, stroke_width=1.5,
        )
        naive_bar.move_to(bax.c2p(1, naive_h / 2))
        naive_t = Text("naive", font_size=16, color=PAL[0])
        naive_t.next_to(naive_bar, DOWN, buff=0.12)

        self.play(Create(bax), Create(true_line), FadeIn(true_lbl),
                  run_time=0.7)
        self.play(GrowFromEdge(naive_bar, DOWN), FadeIn(naive_t),
                  run_time=0.7)
        over_lbl = Text("overshoots", font_size=16, color=PAL[3])
        over_lbl.next_to(naive_bar, UP, buff=0.1)
        self.play(FadeIn(over_lbl, shift=UP * 0.1), run_time=0.5)
        self.wait(1.5)

        # Correction formula
        corr_eq = MathTex(
            r"\hat\sigma_p^2 = \mathrm{Var}(\hat\theta) - "
            r"\overline{\mathrm{SE}^2}",
            font_size=28, color=ACCENT,
        )
        corr_eq.to_edge(RIGHT, buff=0.6).shift(UP * 1.7)
        self.play(Write(corr_eq), run_time=0.9)

        corr_bar = Rectangle(
            width=0.7, height=bax.c2p(1, corr_h)[1] - bax.c2p(1, 0)[1],
            color=PAL[1], fill_opacity=0.7, stroke_width=1.5,
        )
        corr_bar.move_to(bax.c2p(2, corr_h / 2))
        corr_t = Text("corrected", font_size=16, color=PAL[1])
        corr_t.next_to(corr_bar, DOWN, buff=0.12)
        self.play(FadeOut(over_lbl), run_time=0.3)
        self.play(GrowFromEdge(corr_bar, DOWN), FadeIn(corr_t),
                  run_time=0.7)
        check = Text("→ lands on 1.0", font_size=16, color=PAL[1])
        check.next_to(corr_eq, DOWN, buff=0.25)
        self.play(FadeIn(check), run_time=0.5)
        self.wait(1.2)

        # Inset: bias decays like 1/M
        iax = Axes(
            x_range=[0, 5, 1], y_range=[0, 1, 0.5],
            x_length=3.2, y_length=1.7,
            axis_config={"color": AXIS_CLR, "font_size": 14},
            tips=False,
        )
        iax.to_edge(RIGHT, buff=0.6).shift(DOWN * 1.4)
        decay = iax.plot(lambda x: 0.9 / (1 + 1.4 * x),
                         x_range=[0.2, 5, 0.05], color=PAL[2],
                         stroke_width=3)
        ix_lbl = Text("items M", font_size=14, color=AXIS_CLR)
        ix_lbl.next_to(iax, DOWN, buff=0.08)
        decay_eq = MathTex(r"\overline{\mathrm{SE}^2} \sim 1/M",
                           font_size=20, color=PAL[2])
        decay_eq.next_to(iax, UP, buff=0.1)
        self.play(Create(iax), FadeIn(ix_lbl), run_time=0.5)
        self.play(Create(decay), FadeIn(decay_eq), run_time=0.8)
        self.wait(1.6)

        self.play(FadeOut(VGroup(
            header, nl, nl_lbl, dots, bax, true_line, true_lbl,
            naive_bar, naive_t, corr_bar, corr_t, corr_eq, check,
            iax, decay, ix_lbl, decay_eq,
        )), run_time=0.7)

    # ================================================================
    #  Act 2: Bayesian (marginal) — prior x likelihood = posterior
    # ================================================================
    def play_bayesian(self):
        header = Text("Bayesian (marginal)", font_size=30,
                      color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.4)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.5)

        ax = Axes(
            x_range=[-3.5, 3.5, 1], y_range=[0, 0.6, 0.2],
            x_length=9, y_length=3.8,
            axis_config={"color": AXIS_CLR, "include_numbers": True,
                         "font_size": 20},
            tips=False,
        )
        x_lab = ax.get_x_axis_label(
            MathTex(r"\theta", font_size=26, color=AXIS_CLR),
            edge=RIGHT, direction=DOWN,
        )
        ax_group = VGroup(ax, x_lab).move_to(DOWN * 0.55)
        self.play(Create(ax), FadeIn(x_lab), run_time=0.7)

        # Prior N(0,1)
        prior = ax.plot(lambda x: gauss(x, 0, 1),
                        x_range=[-3.5, 3.5, 0.05],
                        color=PAL[4], stroke_width=3)
        prior_lbl = Text("Prior N(0,1)", font_size=18, color=PAL[4])
        prior_lbl.next_to(ax.c2p(-1.6, gauss(-1.6, 0, 1)), UP + LEFT,
                          buff=0.1)
        self.play(Create(prior), FadeIn(prior_lbl), run_time=0.9)
        self.wait(0.6)

        # Likelihood
        lmu, lsig = 1.7, 0.65
        lscale = gauss(0, 0, 1) / gauss(lmu, lmu, lsig)
        lik = ax.plot(lambda x: gauss(x, lmu, lsig) * lscale,
                      x_range=[-3.5, 3.5, 0.05], color=PAL[0],
                      stroke_width=3)
        lik_lbl = Text("Likelihood", font_size=18, color=PAL[0])
        lik_lbl.next_to(ax.c2p(lmu, gauss(lmu, lmu, lsig) * lscale),
                        UP + RIGHT, buff=0.1)
        self.play(Create(lik), FadeIn(lik_lbl), run_time=0.9)
        self.wait(0.6)

        # Posterior (shrunk toward center)
        pmu, psig = 1.0, 0.5
        pscale = gauss(0, 0, 1) / gauss(pmu, pmu, psig) * 1.05
        post = ax.plot(lambda x: gauss(x, pmu, psig) * pscale,
                       x_range=[-3.5, 3.5, 0.05], color=PAL[1],
                       stroke_width=3.5)
        post_lbl = Text("Posterior", font_size=18, color=PAL[1])
        post_peak = gauss(pmu, pmu, psig) * pscale
        post_lbl.next_to(ax.c2p(pmu, post_peak), UP, buff=0.12)

        # 95% credible interval shading
        lo, hi = pmu - 1.96 * psig, pmu + 1.96 * psig
        ci = ax.get_area(post, x_range=[lo, hi], color=PAL[1],
                         opacity=0.22)
        self.play(Create(post), FadeIn(post_lbl), run_time=0.9)
        self.play(FadeIn(ci), run_time=0.5)

        # MAP marker (peak)
        map_dot = Dot(ax.c2p(pmu, 0), color=PAL[1], radius=0.08)
        map_lbl = MathTex(r"\mathrm{MAP}", font_size=20, color=PAL[1])
        map_lbl.next_to(map_dot, DOWN, buff=0.12)
        self.play(FadeIn(map_dot), FadeIn(map_lbl), run_time=0.5)

        # Shrinkage arrow likelihood peak -> posterior peak
        shrink = Arrow(ax.c2p(lmu, -0.06), ax.c2p(pmu, -0.06),
                       color=ACCENT, stroke_width=2.5,
                       max_tip_length_to_length_ratio=0.25, buff=0.05)
        shrink_lbl = Text("shrinkage", font_size=16, color=ACCENT)
        shrink_lbl.next_to(shrink, DOWN, buff=0.05)
        self.play(Create(shrink), FadeIn(shrink_lbl), run_time=0.6)

        ci_lbl = Text("95% credible interval", font_size=16, color=PAL[1])
        ci_lbl.to_edge(DOWN, buff=0.25)
        self.play(FadeIn(ci_lbl), run_time=0.4)
        self.wait(2.0)

        self.play(FadeOut(VGroup(
            header, ax_group, prior, prior_lbl, lik, lik_lbl,
            post, post_lbl, ci, map_dot, map_lbl, shrink, shrink_lbl,
            ci_lbl,
        )), run_time=0.7)

    # ================================================================
    #  Act 3: Method of moments (ANOVA)
    # ================================================================
    def play_moments(self):
        header = Text("Method of moments (ANOVA)", font_size=30,
                      color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.4)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.5)

        # Three mean-square boxes
        def ms_box(tex, color):
            box = RoundedRectangle(corner_radius=0.12, width=1.9,
                                   height=1.0, color=color,
                                   stroke_width=2)
            lbl = MathTex(tex, font_size=30, color=color)
            lbl.move_to(box)
            return VGroup(box, lbl)

        b_p = ms_box(r"MS_p", PAL[0])
        b_i = ms_box(r"MS_i", PAL[2])
        b_pi = ms_box(r"MS_{pi}", PAL[3])
        boxes = VGroup(b_p, b_i, b_pi).arrange(RIGHT, buff=0.7)
        boxes.shift(UP * 1.3)
        self.play(LaggedStartMap(FadeIn, boxes, lag_ratio=0.25),
                  run_time=1.0)
        self.wait(0.6)

        # Solved component formula
        solved = MathTex(
            r"\hat V_{\mathrm{person}} = \frac{MS_p - MS_{pi}}{n_i}",
            font_size=34, color=ACCENT,
        )
        solved.shift(DOWN * 0.4)
        # Arrows from MS_p and MS_pi into the formula
        a1 = Arrow(b_p.get_bottom(), solved.get_top() + LEFT * 1.2,
                   color=PAL[0], stroke_width=2.5, buff=0.15,
                   max_tip_length_to_length_ratio=0.15)
        a2 = Arrow(b_pi.get_bottom(), solved.get_top() + RIGHT * 1.2,
                   color=PAL[3], stroke_width=2.5, buff=0.15,
                   max_tip_length_to_length_ratio=0.15)
        self.play(Create(a1), Create(a2), run_time=0.6)
        self.play(Write(solved), run_time=1.0)
        self.wait(1.0)

        cap = Text("no likelihood, no prior — cheapest, bluntest",
                   font_size=22, color=TEXT2)
        cap.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(cap, shift=UP * 0.1), run_time=0.6)
        self.wait(1.8)

        self.play(FadeOut(VGroup(
            header, boxes, solved, a1, a2, cap,
        )), run_time=0.7)

    # ================================================================
    #  Act 4: Identifiability — one trial vs many
    # ================================================================
    def play_identifiability(self):
        header = Text("One trial vs many", font_size=30,
                      color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.4)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.5)

        ax = Axes(
            x_range=[1, 5, 1], y_range=[0.9, 1.9, 0.2],
            x_length=8, y_length=4.0,
            axis_config={"color": AXIS_CLR, "include_numbers": True,
                         "font_size": 20},
            tips=False,
        )
        x_lab = ax.get_x_axis_label(
            Text("trials per cell", font_size=20, color=AXIS_CLR),
            edge=RIGHT, direction=DOWN,
        )
        y_lab = ax.get_y_axis_label(
            MathTex(r"\hat\sigma_{pi}^2", font_size=24, color=AXIS_CLR),
            edge=UP, direction=LEFT,
        )
        ax_group = VGroup(ax, x_lab, y_lab).move_to(DOWN * 0.45)
        self.play(Create(ax), FadeIn(x_lab), FadeIn(y_lab), run_time=0.8)

        # Dashed truth line at 1.0
        truth = DashedLine(ax.c2p(1, 1.0), ax.c2p(5, 1.0),
                           color=ACCENT, stroke_width=2.5,
                           dash_length=0.1)
        truth_lbl = MathTex(r"\mathrm{true}\ \sigma_{pi}^2",
                            font_size=20, color=ACCENT)
        truth_lbl.next_to(ax.c2p(5, 1.0), UR, buff=0.05)
        self.play(Create(truth), FadeIn(truth_lbl), run_time=0.7)

        # Estimate overshoots at 1 trial, converges down to truth
        trials = np.array([1, 2, 3, 4, 5])
        est = 1.0 + 0.8 / trials  # 1.8, 1.4, 1.27, 1.2, 1.16 -> decays
        pts = [ax.c2p(t, e) for t, e in zip(trials, est)]
        dots = VGroup(*[Dot(p, color=PAL[0], radius=0.08) for p in pts])
        curve = VMobject(color=PAL[0], stroke_width=3)
        curve.set_points_smoothly(pts)

        # Highlight the overshoot at one trial
        over_lbl = Text("interaction absorbs\nBernoulli noise",
                        font_size=18, color=PAL[3], line_spacing=0.8)
        over_lbl.next_to(pts[0], RIGHT, buff=0.3)

        self.play(FadeIn(dots[0]), run_time=0.4)
        self.play(FadeIn(over_lbl, shift=RIGHT * 0.1), run_time=0.6)
        self.wait(0.8)
        self.play(Create(curve),
                  LaggedStartMap(FadeIn, dots[1:], lag_ratio=0.2),
                  run_time=1.4)
        self.wait(0.8)

        cap = Text("replication separates interaction from noise",
                   font_size=22, color=TEXT2)
        cap.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(cap, shift=UP * 0.1), run_time=0.6)
        self.wait(2.0)

        self.play(FadeOut(VGroup(
            header, ax_group, truth, truth_lbl, dots, curve,
            over_lbl, cap,
        )), run_time=0.7)

    # ================================================================
    #  Takeaway
    # ================================================================
    def play_takeaway(self):
        heading = Text("Three estimators, one estimand",
                       font_size=36, color=WHITE, weight=BOLD)
        heading.shift(UP * 0.6)
        sub = Text("replication splits signal from noise",
                   font_size=26, color=ACCENT)
        sub.next_to(heading, DOWN, buff=0.4)
        line = Line(LEFT * 2.2, RIGHT * 2.2, color=ACCENT,
                    stroke_width=1.5)
        line.next_to(sub, DOWN, buff=0.4)
        src = Text("AIMS — Chapter 5: Reliability",
                   font_size=18, color=ManimColor("#444444"))
        src.next_to(line, DOWN, buff=0.4)

        self.play(FadeIn(heading, shift=DOWN * 0.15), run_time=0.7)
        self.play(FadeIn(sub, shift=UP * 0.1), run_time=0.6)
        self.play(Create(line), run_time=0.4)
        self.play(FadeIn(src), run_time=0.4)
        self.wait(2.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
