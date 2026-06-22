"""
AIMS Chapter 5 Animation: Judges and Their Biases
Cohen's kappa for LLM judges, and reliability's blind spot to systematic bias.

Run with:
    PATH="/lfs/local/0/sttruong/miniconda3/bin:$PATH" \
    manim -qh --disable_caching --media_dir media/ch5 animations/ch5/judge_kappa.py JudgeKappa
"""

from manim import *
import numpy as np

# ── design tokens ────────────────────────────────────────────────
ACCENT = "#FFD966"
BG = "#0f0f0f"
TEXT2 = "#aaaaaa"
PAL = ["#5B8DEE", "#45BF7C", "#F0A35C", "#E8637A", "#B07CD8"]
AXIS_CLR = "#888888"


class JudgeKappa(Scene):
    """Cohen's kappa for judges, and the reliability blind spot."""

    def construct(self):
        self.camera.background_color = BG
        self.play_title()
        self.play_agreement()       # Act 1
        self.play_chance_correct()  # Act 2
        self.play_blind_spot()      # Act 3
        self.play_takeaway()

    # ================================================================
    #  Title
    # ================================================================
    def play_title(self):
        title = Text("Judges and Their Biases", font_size=44,
                     color=WHITE, weight=BOLD)
        sub = Text("Cohen's kappa, and what reliability cannot see",
                   font_size=24, color=TEXT2)
        sub.next_to(title, DOWN, buff=0.35)
        line = Line(LEFT * 2.5, RIGHT * 2.5, color=ACCENT,
                    stroke_width=1.5)
        line.next_to(sub, DOWN, buff=0.3)
        g = VGroup(title, sub, line)

        self.play(FadeIn(title, shift=UP * 0.2), run_time=0.8)
        self.play(FadeIn(sub), Create(line), run_time=0.6)
        self.wait(1.8)
        self.play(FadeOut(g, shift=UP * 0.4), run_time=0.7)

    # ================================================================
    #  Act 1: Two judges label the same responses → 2x2 table
    # ================================================================
    def play_agreement(self):
        header = Text("Two Judges, Same Responses", font_size=28,
                      color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.35)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.5)

        # Verdicts for 8 items: 1 = Good, 0 = Bad
        a = [1, 1, 0, 1, 0, 1, 1, 0]
        b = [1, 1, 0, 1, 1, 1, 0, 0]   # disagree on items 5 and 7

        def chip(v, color):
            box = Square(side_length=0.6, color=color,
                         fill_color=color, fill_opacity=0.22,
                         stroke_width=2)
            lab = Text("G" if v == 1 else "B", font_size=20,
                       color=color, weight=BOLD)
            lab.move_to(box)
            return VGroup(box, lab)

        row_a = VGroup(*[chip(v, PAL[0]) for v in a]).arrange(RIGHT, buff=0.18)
        row_b = VGroup(*[chip(v, PAL[1]) for v in b]).arrange(RIGHT, buff=0.18)
        lab_a = Text("Judge A", font_size=20, color=PAL[0], weight=BOLD)
        lab_b = Text("Judge B", font_size=20, color=PAL[1], weight=BOLD)

        row_a.next_to(lab_a, RIGHT, buff=0.4)
        line_a = VGroup(lab_a, row_a)
        row_b.next_to(lab_b, RIGHT, buff=0.4)
        line_b = VGroup(lab_b, row_b)
        rows = VGroup(line_a, line_b).arrange(DOWN, buff=0.4,
                                              aligned_edge=LEFT)
        rows.move_to(UP * 1.4)
        # align row chips under each other
        row_b.align_to(row_a, LEFT)

        self.play(FadeIn(lab_a), LaggedStartMap(FadeIn, row_a, lag_ratio=0.1),
                  run_time=1.0)
        self.play(FadeIn(lab_b), LaggedStartMap(FadeIn, row_b, lag_ratio=0.1),
                  run_time=1.0)

        # Mark disagreements
        marks = VGroup()
        for i, (va, vb) in enumerate(zip(a, b)):
            if va != vb:
                box_a = row_a[i][0]
                box_b = row_b[i][0]
                rect = SurroundingRectangle(VGroup(box_a, box_b),
                                            color=PAL[3], buff=0.05,
                                            stroke_width=2.5)
                marks.add(rect)
        self.play(Create(marks), run_time=0.7)
        self.wait(1.2)

        # 2x2 confusion table between the two judges
        n11 = sum(1 for x, y in zip(a, b) if x == 1 and y == 1)  # 4
        n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)  # 1
        n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)  # 1
        n00 = sum(1 for x, y in zip(a, b) if x == 0 and y == 0)  # 2

        table = Table(
            [[str(n11), str(n10)], [str(n01), str(n00)]],
            row_labels=[Text("A: Good", font_size=18, color=PAL[0]),
                        Text("A: Bad", font_size=18, color=PAL[0])],
            col_labels=[Text("B: Good", font_size=18, color=PAL[1]),
                        Text("B: Bad", font_size=18, color=PAL[1])],
            include_outer_lines=True,
            line_config={"color": AXIS_CLR, "stroke_width": 1.5},
        )
        table.scale(0.55)
        table.move_to(DOWN * 1.3 + LEFT * 2.6)
        for entry in table.get_entries():
            entry.set_color(WHITE)
        # recolor header labels
        self.play(Create(table), run_time=1.2)
        self.wait(1.0)

        po = (n11 + n00) / 8
        po_txt = MathTex(
            r"p_o = \frac{4 + 2}{8} = 0.85",
            font_size=30, color=ACCENT,
        )
        po_cap = Text("raw agreement looks high", font_size=20, color=TEXT2)
        po_box = VGroup(po_txt, po_cap).arrange(DOWN, buff=0.3)
        po_box.move_to(DOWN * 1.3 + RIGHT * 3.2)

        self.play(Write(po_txt), run_time=0.9)
        self.play(FadeIn(po_cap, shift=UP * 0.1), run_time=0.5)
        self.wait(2.5)

        self.act1_group = VGroup(header, rows, marks, table, po_box)
        self.play(FadeOut(self.act1_group), run_time=0.7)

    # ================================================================
    #  Act 2: Chance correction → kappa
    # ================================================================
    def play_chance_correct(self):
        header = Text("Correcting for Chance", font_size=28,
                      color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.35)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.5)

        kappa = MathTex(
            r"\kappa = \frac{p_o - p_e}{1 - p_e}",
            font_size=46, color=ACCENT,
        )
        kappa.move_to(UP * 1.2)
        self.play(Write(kappa), run_time=1.0)
        self.wait(1.2)

        terms = VGroup(
            MathTex(r"p_o = 0.85", font_size=30, color=WHITE),
            MathTex(r"p_e = 0.625", font_size=30, color=TEXT2),
        ).arrange(RIGHT, buff=1.2)
        terms.next_to(kappa, DOWN, buff=0.6)
        self.play(FadeIn(terms[0], shift=UP * 0.1), run_time=0.5)
        self.play(FadeIn(terms[1], shift=UP * 0.1), run_time=0.5)
        self.wait(1.0)

        result = MathTex(
            r"\kappa = \frac{0.85 - 0.625}{1 - 0.625} = 0.60",
            font_size=34, color=ACCENT,
        )
        result.next_to(terms, DOWN, buff=0.55)
        self.play(Write(result), run_time=1.0)
        self.wait(1.5)

        cap = Text(
            "a generalizability coefficient for raters — "
            "more judges shrink rater variance",
            font_size=20, color=TEXT2,
        )
        cap.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(cap, shift=UP * 0.1), run_time=0.6)
        self.wait(3.0)

        self.act2_group = VGroup(header, kappa, terms, result, cap)
        self.play(FadeOut(self.act2_group), run_time=0.7)

    # ================================================================
    #  Act 3: THE BLIND SPOT — position bias
    # ================================================================
    def play_blind_spot(self):
        header = Text("The Blind Spot: Position Bias", font_size=28,
                      color=PAL[3], weight=BOLD)
        header.to_edge(UP, buff=0.35)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.6)

        rule = Text("A judge that always picks the response shown first",
                    font_size=22, color=TEXT2)
        rule.next_to(header, DOWN, buff=0.3)
        self.play(FadeIn(rule), run_time=0.5)

        # Three pairwise comparisons (A vs B); judge always picks the left
        def make_pair(left_color, right_color):
            left = VGroup(
                Square(side_length=0.7, color=left_color,
                       fill_color=left_color, fill_opacity=0.25,
                       stroke_width=2),
                Text("A", font_size=22, color=left_color, weight=BOLD),
            )
            left[1].move_to(left[0])
            vs = Text("vs", font_size=18, color=TEXT2)
            right = VGroup(
                Square(side_length=0.7, color=right_color,
                       fill_color=right_color, fill_opacity=0.25,
                       stroke_width=2),
                Text("B", font_size=22, color=right_color, weight=BOLD),
            )
            right[1].move_to(right[0])
            return VGroup(left, vs, right).arrange(RIGHT, buff=0.35)

        pairs = VGroup(*[make_pair(PAL[0], PAL[1]) for _ in range(3)])
        pairs.arrange(DOWN, buff=0.45)
        pairs.move_to(UP * 0.2 + LEFT * 2.3)
        self.play(LaggedStartMap(FadeIn, pairs, lag_ratio=0.2), run_time=0.9)

        # Always pick the first (left, "A")
        picks = VGroup()
        for p in pairs:
            ring = SurroundingRectangle(p[0], color=ACCENT, buff=0.08,
                                        stroke_width=3)
            picks.add(ring)
        arrow_lbl = Text("always picks first", font_size=20, color=ACCENT)
        arrow_lbl.next_to(pairs, RIGHT, buff=0.8)
        self.play(Create(picks), FadeIn(arrow_lbl, shift=LEFT * 0.1),
                  run_time=0.9)
        self.wait(1.5)

        # Run it twice → agrees with itself perfectly
        consist = VGroup(
            Text("Run twice on the same pairs:", font_size=22, color=WHITE),
            MathTex(r"\kappa \approx 1.0", font_size=34, color=PAL[1]),
            Text("perfectly consistent", font_size=20, color=PAL[1]),
        ).arrange(DOWN, buff=0.25)
        consist.move_to(DOWN * 1.6 + LEFT * 2.3)
        self.play(FadeIn(consist[0]), run_time=0.5)
        self.play(Write(consist[1]), FadeIn(consist[2]), run_time=0.8)
        self.wait(1.5)

        # But systematically wrong
        wrong = VGroup(
            Text("yet the verdict ignores quality", font_size=22,
                 color=PAL[3]),
            Text("SYSTEMATICALLY WRONG", font_size=24, color=PAL[3],
                 weight=BOLD),
        ).arrange(DOWN, buff=0.2)
        wrong.move_to(DOWN * 1.4 + RIGHT * 3.0)
        self.play(FadeIn(wrong[0]), run_time=0.5)
        self.play(Write(wrong[1]), run_time=0.7)
        self.wait(1.5)

        # Big caption
        big = Text("Reliable but systematically wrong.",
                   font_size=30, color=ACCENT, weight=BOLD)
        big.to_edge(DOWN, buff=0.35)
        self.play(
            FadeOut(VGroup(consist, wrong, rule, picks, arrow_lbl, pairs)),
            run_time=0.6,
        )
        self.play(FadeIn(big, shift=UP * 0.15), run_time=0.7)
        self.wait(2.8)

        self.act3_group = VGroup(header, big)
        self.play(FadeOut(self.act3_group), run_time=0.7)

    # ================================================================
    #  Takeaway
    # ================================================================
    def play_takeaway(self):
        heading = Text("Consistency is not Correctness", font_size=36,
                       color=WHITE, weight=BOLD)
        heading.shift(UP * 0.9)

        msg = Text(
            "Reliability cannot see bias — a judge can be "
            "perfectly consistent yet invalid.",
            font_size=24, color=TEXT2,
        )
        msg.next_to(heading, DOWN, buff=0.5)

        validity = MathTex(
            r"\mathrm{reliable} \;\neq\; \mathrm{valid}",
            font_size=34, color=ACCENT,
        )
        validity.next_to(msg, DOWN, buff=0.5)

        line = Line(LEFT * 2.2, RIGHT * 2.2, color=ACCENT, stroke_width=1.5)
        line.next_to(validity, DOWN, buff=0.5)

        source = Text("AIMS — Chapter 5: Reliability",
                      font_size=18, color=ManimColor("#444444"))
        source.next_to(line, DOWN, buff=0.35)

        self.play(FadeIn(heading, shift=DOWN * 0.15), run_time=0.7)
        self.play(FadeIn(msg, shift=UP * 0.1), run_time=0.6)
        self.play(Write(validity), run_time=0.8)
        self.play(Create(line), run_time=0.4)
        self.play(FadeIn(source), run_time=0.4)
        self.wait(3.5)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
