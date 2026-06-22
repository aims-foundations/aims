"""
AIMS Chapter 5 Animation: Leaderboards Flicker
Reliability vs validity — same model, same benchmark, different rank.
Three runs reorder the leaderboard; collapsing to means with 95% CIs
shows that many rank differences sit inside the noise.

Run with:
    PATH="/lfs/local/0/sttruong/miniconda3/bin:$PATH" \
    manim -qh --disable_caching --media_dir media/ch5 animations/ch5/leaderboard_flicker.py LeaderboardFlicker
"""

from manim import *
import numpy as np

# ── design tokens ────────────────────────────────────────────────
ACCENT = "#FFD966"
BG = "#0f0f0f"
TEXT2 = "#aaaaaa"
PAL = ["#5B8DEE", "#45BF7C", "#F0A35C", "#E8637A", "#B07CD8"]
AXIS_CLR = "#888888"


class LeaderboardFlicker(Scene):
    """Leaderboards flicker: reliability before validity."""

    def construct(self):
        self.camera.background_color = BG
        self.play_title()
        self.play_reorder()
        self.play_intervals()
        self.play_takeaway()

    # ================================================================
    #  Title
    # ================================================================
    def play_title(self):
        title = Text("Leaderboards Flicker", font_size=44,
                     color=WHITE, weight=BOLD)
        sub = Text("Same model, same benchmark, different answer",
                   font_size=24, color=TEXT2)
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
    #  Act 1: leaderboard reordering across runs
    # ================================================================
    def play_reorder(self):
        header = Text("One benchmark, three runs", font_size=28,
                      color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.35)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.5)

        # Six models, each pinned to a stable color so the eye can
        # follow a model as it slides between ranks.
        models = [
            ("Atlas-7B", PAL[0]),
            ("Orca-13B", PAL[1]),
            ("Vega-8B", PAL[2]),
            ("Nova-9B", PAL[3]),
            ("Lyra-7B", PAL[4]),
            ("Comet-6B", "#7FBF9F"),
        ]
        n = len(models)

        # Three runs: each a permutation of ranks + jittered scores.
        # Order lists are model indices from rank 1 (top) downward.
        runs = [
            ([0, 1, 2, 3, 4, 5],
             [0.79, 0.78, 0.76, 0.74, 0.73, 0.71]),
            ([1, 0, 3, 2, 5, 4],
             [0.78, 0.77, 0.76, 0.75, 0.73, 0.72]),
            ([0, 3, 1, 5, 2, 4],
             [0.79, 0.77, 0.76, 0.74, 0.73, 0.71]),
        ]

        top_y = 1.95
        row_dy = 0.78
        col_x = -1.4   # x of the model label column
        score_x = 2.7  # x of the score column

        def row_y(rank):
            return top_y - rank * row_dy

        # Run label shown top-right.
        run_lbl = Text("Run 1", font_size=24, color=ACCENT, weight=BOLD)
        run_lbl.to_edge(RIGHT, buff=0.9).shift(UP * 2.6)

        # Build the initial rows for Run 1.
        order0, scores0 = runs[0]
        rank_nums = VGroup()
        rows = {}  # model index -> VGroup(badge, name, score)
        for rank, midx in enumerate(order0):
            name, color = models[midx]
            y = row_y(rank)

            num = Text(f"{rank + 1}", font_size=22, color=TEXT2)
            num.move_to(np.array([col_x - 1.55, y, 0]))
            rank_nums.add(num)

            badge = Dot(point=np.array([col_x - 0.95, y, 0]),
                        radius=0.13, color=color)
            label = Text(name, font_size=24, color=color, weight=BOLD)
            label.move_to(np.array([col_x, y, 0])).align_to(
                np.array([col_x - 0.7, y, 0]), LEFT)
            score = Text(f"{scores0[rank]:.2f}", font_size=24,
                         color=TEXT2)
            score.move_to(np.array([score_x, y, 0]))

            row = VGroup(badge, label, score)
            rows[midx] = row

        self.play(
            FadeIn(run_lbl, shift=LEFT * 0.1),
            LaggedStart(*[FadeIn(rank_nums[i], shift=RIGHT * 0.1)
                          for i in range(n)], lag_ratio=0.06),
            LaggedStart(*[FadeIn(rows[order0[i]], shift=RIGHT * 0.15)
                          for i in range(n)], lag_ratio=0.08),
            run_time=1.4,
        )
        self.wait(1.6)

        note = Text("Watch the ranks shuffle", font_size=22,
                    color=TEXT2)
        note.to_edge(DOWN, buff=0.25)
        self.play(FadeIn(note, shift=UP * 0.1), run_time=0.5)

        # Animate Run 2 and Run 3: each model slides to its new rank
        # and its score text updates.
        for ri in (1, 2):
            order, scores = runs[ri]
            new_lbl = Text(f"Run {ri + 1}", font_size=24,
                           color=ACCENT, weight=BOLD)
            new_lbl.move_to(run_lbl)

            anims = [Transform(run_lbl, new_lbl)]
            for rank, midx in enumerate(order):
                y = row_y(rank)
                row = rows[midx]
                # Move the whole row (badge + name + score) to new y.
                anims.append(row.animate.shift(
                    np.array([0, y - row[0].get_center()[1], 0])))
                # Update the score text in place.
                new_score = Text(f"{scores[rank]:.2f}", font_size=24,
                                 color=TEXT2)
                new_score.move_to(np.array([score_x, y, 0]))
                anims.append(Transform(row[2], new_score))

            self.play(*anims, run_time=1.7)
            self.wait(1.3)

        self.wait(0.6)

        self.reorder_group = VGroup(
            header, run_lbl, rank_nums, note,
            *[rows[m] for m in rows],
        )
        self.play(FadeOut(self.reorder_group, shift=UP * 0.2),
                  run_time=0.7)

    # ================================================================
    #  Act 2: means with 95% confidence intervals
    # ================================================================
    def play_intervals(self):
        header = Text("Collapse the runs: mean ± 95% CI",
                      font_size=28, color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.35)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.5)

        ax = Axes(
            x_range=[0.66, 0.84, 0.04], y_range=[0, 6, 1],
            x_length=8.5, y_length=4.0,
            axis_config={"color": AXIS_CLR, "font_size": 22},
            x_axis_config={"include_numbers": True},
            y_axis_config={"include_numbers": False},
            tips=False,
        )
        x_lab = ax.get_x_axis_label(
            Text("score", font_size=20, color=AXIS_CLR),
            edge=RIGHT, direction=DOWN,
        )
        ax_group = VGroup(ax, x_lab).move_to(DOWN * 0.5)
        self.play(Create(ax), FadeIn(x_lab), run_time=0.8)

        # Sorted-by-mean models with (mean, half-width) of a 95% CI.
        # Widths chosen so several intervals overlap with neighbours.
        rows = [
            ("Atlas-7B", PAL[0], 0.790, 0.022),
            ("Orca-13B", PAL[1], 0.775, 0.024),
            ("Nova-9B", PAL[3], 0.762, 0.020),
            ("Vega-8B", PAL[2], 0.755, 0.023),
            ("Lyra-7B", PAL[4], 0.730, 0.018),
            ("Comet-6B", "#7FBF9F", 0.715, 0.019),
        ]
        n = len(rows)

        marks = VGroup()
        for i, (name, color, mean, hw) in enumerate(rows):
            y_val = n - i  # rank 1 highest on the plot
            cy = ax.c2p(mean, y_val)[1]
            lo = ax.c2p(mean - hw, y_val)
            hi = ax.c2p(mean + hw, y_val)

            bar = Line(lo, hi, color=color, stroke_width=3)
            cap_lo = Line(lo + UP * 0.08, lo + DOWN * 0.08,
                          color=color, stroke_width=3)
            cap_hi = Line(hi + UP * 0.08, hi + DOWN * 0.08,
                          color=color, stroke_width=3)
            dot = Dot(ax.c2p(mean, y_val), radius=0.07, color=color)
            name_lbl = Text(name, font_size=18, color=color)
            name_lbl.next_to(lo, LEFT, buff=0.2)

            marks.add(VGroup(bar, cap_lo, cap_hi, dot, name_lbl))

        self.play(LaggedStart(*[FadeIn(m, shift=RIGHT * 0.12)
                                for m in marks], lag_ratio=0.12),
                  run_time=1.8)
        self.wait(1.5)

        # Highlight overlapping pairs: (Orca,Nova,Vega) cluster and
        # (Lyra,Comet) pair. Tint with ACCENT bands where they overlap.
        # Pair indices into `rows`.
        overlap_pairs = [(1, 2), (2, 3), (4, 5)]
        bands = VGroup()
        for a, b in overlap_pairs:
            ma, mb = rows[a], rows[b]
            ya = n - a
            yb = n - b
            # Overlap region in score space.
            left = max(ma[2] - ma[3], mb[2] - mb[3])
            right = min(ma[2] + ma[3], mb[2] + mb[3])
            if right <= left:
                continue
            x_l = ax.c2p(left, 0)[0]
            x_r = ax.c2p(right, 0)[0]
            top = ax.c2p(0, ya)[1] + 0.18
            bot = ax.c2p(0, yb)[1] - 0.18
            band = Rectangle(
                width=x_r - x_l, height=top - bot,
                stroke_width=0, fill_color=ACCENT, fill_opacity=0.16,
            )
            band.move_to(np.array([(x_l + x_r) / 2,
                                   (top + bot) / 2, 0]))
            bands.add(band)

        self.play(LaggedStart(*[FadeIn(b) for b in bands],
                              lag_ratio=0.25), run_time=1.2)

        note = Text("Overlapping intervals: rank gaps inside the noise",
                    font_size=22, color=ACCENT)
        note.to_edge(DOWN, buff=0.25)
        self.play(FadeIn(note, shift=UP * 0.1), run_time=0.5)
        self.wait(3.2)

        self.play(FadeOut(VGroup(header, ax_group, marks, bands, note),
                          shift=UP * 0.2), run_time=0.7)

    # ================================================================
    #  Takeaway: reliability vs validity
    # ================================================================
    def play_takeaway(self):
        heading = Text("Reliability before Validity", font_size=36,
                       color=WHITE, weight=BOLD)
        heading.to_edge(UP, buff=0.6)

        divider = Line(UP * 1.4, DOWN * 1.9, color=AXIS_CLR,
                       stroke_width=1.5)

        # Left column: reliability (the focus of this chapter).
        rel_title = Text("Reliability", font_size=34, color=ACCENT,
                         weight=BOLD)
        rel_sub = Text("same answer twice?", font_size=22, color=TEXT2)
        rel_sub.next_to(rel_title, DOWN, buff=0.3)
        rel_tag = Text("this chapter — comes first", font_size=20,
                       color=ACCENT)
        rel_tag.next_to(rel_sub, DOWN, buff=0.45)
        left = VGroup(rel_title, rel_sub, rel_tag)
        left.move_to(LEFT * 3.3 + DOWN * 0.3)

        rel_box = SurroundingRectangle(
            left, color=ACCENT, buff=0.4, corner_radius=0.12,
            stroke_width=2,
        )

        # Right column: validity (the next question).
        val_title = Text("Validity", font_size=34, color=TEXT2,
                         weight=BOLD)
        val_sub = Text("the right answer?", font_size=22, color=TEXT2)
        val_sub.next_to(val_title, DOWN, buff=0.3)
        right = VGroup(val_title, val_sub)
        right.move_to(RIGHT * 3.3 + DOWN * 0.45)

        self.play(FadeIn(heading, shift=DOWN * 0.15), run_time=0.7)
        self.play(Create(divider), run_time=0.5)
        self.play(FadeIn(left, shift=RIGHT * 0.15),
                  FadeIn(right, shift=LEFT * 0.15), run_time=0.9)
        self.wait(0.6)
        self.play(Create(rel_box), run_time=0.6)
        self.wait(3.0)

        source = Text("AIMS — Chapter 5: Reliability",
                      font_size=18, color=ManimColor("#444444"))
        source.to_edge(DOWN, buff=0.25)
        self.play(FadeIn(source), run_time=0.5)
        self.wait(2.0)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
