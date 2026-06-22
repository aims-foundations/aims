# Reliability — Video Script

**Chapter 5 of AI Measurement Science (AIMS)**
Target length: ~10-12 minutes
Format: Narrated animation (3Blue1Brown style)

---

## Production Notes

- **Animations** are in `animations/ch5/*.py` (Manim, 1080p60). Each scene listed
  below corresponds to a rendered `.mp4` in `media/ch5/videos/`.
- **Narration** generated via edge-tts, synced via `stitch_narrated.sh`.
- **Pacing markers:** `[pause]` = ~1 s beat. `[beat]` = ~0.5 s.

---

## PART 1 — WHEN LEADERBOARDS FLICKER (~1:45)

### 1.1 Opening Hook

**NARRATOR:**

In the last chapters, we learned to fit measurement models and to choose
informative items. But before we ask whether a benchmark measures the
*right* thing, we must ask something more basic.

[beat]

Does the evaluation give the same answer when applied to the same thing twice?

> [ANIMATION: leaderboard_flicker.py — LeaderboardFlicker]
> Cue: Leaderboard of six models, evaluated three times

Here are the same models, on the same benchmark, evaluated three times.
The rankings shuffle. Lines cross. Models trade places.

> Cue: Collapse runs into mean and 95% confidence interval

Collapse those runs into a mean and a confidence interval, and many of the
rank differences vanish into the noise — overlapping intervals we cannot
tell apart.

[pause]

This is the reliability problem. An evaluation that gives a different answer
every time cannot be measuring anything stable about the model.

> Cue: Split screen — Reliability vs Validity

Reliability asks: same answer twice? Validity asks: the right answer? This
chapter is about the first question. Before we can measure the right thing,
we must measure *something*.

[pause]

---

## PART 2 — DECOMPOSING THE VARIANCE (~2:00)

### 2.1 Variance Components

**NARRATOR:**

> [ANIMATION: variance_decomposition.py — VarianceDecomposition]
> Cue: Response matrix, pull out one cell's log-odds

Where does that variability come from? Start from the response model. The
log-odds of a correct response is an ability, minus a difficulty, plus a
person-by-item interaction.

> Cue: Variance splits into four stacked bars

When we ask how much a benchmark score varies, that variance splits into
four additive, non-overlapping pieces: a person component, an item
component, their interaction, and an irreducible Bernoulli residual.

> Cue: Highlight the person component

Only one of these is signal. The person component captures genuine
differences between models — the thing we actually want to generalize about.

> Cue: Reliability formula appears

Reliability is the share of total variance carried by that signal. We call
it the generalizability coefficient: person variance over total variance.

[pause]

And here is the catch. For a single binary trial, the signal bar is small
next to item difficulty and Bernoulli noise. Recovering it is the work of
the rest of the chapter.

---

## PART 3 — ESTIMATING THE COMPONENTS (~2:30)

### 3.1 Three Estimators

**NARRATOR:**

> [ANIMATION: three_estimators.py — ThreeEstimators]
> Cue: Plug-in — fitted abilities spread out

A finite benchmark only lets us *estimate* these components. Three methods
do it, in increasing distance from the response model.

The plug-in method fits each model's ability as a fixed effect, then asks how
spread out the fitted abilities are. But noisy estimates spread out more than
the truth, so the naive variance overshoots.

> Cue: Subtract average squared standard error

The fix is to subtract the average squared standard error — the estimation
noise masquerading as signal. That correction recovers the planted value,
and it shrinks as items accumulate, like one over the number of items.

> Cue: Bayesian — prior times likelihood

The Bayesian method models the population directly. Prior times likelihood
gives a posterior over the ability spread; we report its peak, or its mean
with a credible interval. The prior shrinks extreme estimates toward the
center.

> Cue: Method of moments — ANOVA mean squares

The method of moments fits nothing at all. It reads the components straight
off the analysis of variance — the mean squares for persons, items, and their
interaction. No likelihood, no prior: the cheapest of the three, and the
bluntest.

> Cue: One trial vs many — interaction separates from noise

One subtlety ties them together. With a single trial, the person-by-item
interaction and the Bernoulli residual are fused — the interaction absorbs
the noise and overshoots. Only replication — more than one trial per cell —
pulls them apart and converges to the truth.

[pause]

---

## PART 4 — RELIABILITY IS LOCAL (~1:30)

### 4.1 Conditional Reliability

**NARRATOR:**

> [ANIMATION: conditional_reliability.py — ConditionalReliability]
> Cue: Reliability curve over ability

Reliability is not a single number — it varies with ability. Here is the
conditional reliability for an item pool clustered at easy difficulties. On
average it looks fine.

> Cue: Slide marker to high ability — curve collapses

But slide to the high-ability end, where the frontier models live, and it
collapses. Easy items barely inform a strong model: every model gets them
right, so they carry no information about who is better.

> Cue: Overlay a difficulty-matched pool

Match the item difficulties to the ability range you care about, and
reliability stays high exactly where you need to tell models apart. This is
the design lesson from the chapter on item selection, seen from the
reliability side.

[pause]

---

## PART 5 — DESIGNING FOR GENERALIZABILITY (~2:00)

### 5.1 G-studies and D-studies

**NARRATOR:**

> [ANIMATION: g_d_study.py — GandDStudy]
> Cue: Model-by-item-by-rater design, seven components

Real evaluations have more than one facet. A model-by-item-by-rater design
has seven variance components. A generalizability study — a G-study —
estimates them from data.

> Cue: D-study heatmap — reliability over items and raters

The distinctive step is the D-study. Given the components, it asks: how many
items and how many raters does it take to reach a target reliability — say,
ninety percent? Here is reliability over a grid of items and raters, with the
ninety-percent contour drawn in.

> Cue: Trade-off — add raters vs add items

The key insight is that you can trade facets against each other. If rater
disagreement dominates, adding raters buys more than adding items. If item
sampling dominates, the reverse. The D-study makes that trade-off explicit
and quantitative — the cheapest path onto the contour.

[pause]

---

## PART 6 — JUDGES AND THEIR BIASES (~1:45)

### 6.1 Cohen's Kappa and Systematic Bias

**NARRATOR:**

> [ANIMATION: judge_kappa.py — JudgeKappa]
> Cue: Two judges labeling the same responses

When the rater is an LLM judge, its facet gets its own coefficient. Two
judges label the same responses. Their raw agreement looks high — but some
of that agreement is just chance.

> Cue: Chance-correction — Cohen's kappa

Cohen's kappa corrects for it: observed agreement minus chance agreement,
normalized. It plays the role of a generalizability coefficient for raters,
and more judges shrink the rater variance, just like more items.

> Cue: Position bias — judge agrees with itself but is wrong

But reliability has a blind spot. Consider a judge with position bias — it
always prefers the first option. Run it twice and it agrees with itself
perfectly. Reliability is high. Yet the measurement is *systematically*
wrong.

[beat]

Reliability cannot see bias. Consistency is not correctness — that is a
validity question, and the subject of the next chapter.

[pause]

---

## PART 7 — BACK TO CLASSICAL COEFFICIENTS (~1:15)

### 7.1 Spearman-Brown and SEM

**NARRATOR:**

> [ANIMATION: spearman_brown.py — SpearmanBrown]
> Cue: From the full matrix to total scores

When all you have is total scores, the same reliability comes back through
the classical coefficients — Cronbach's alpha, split-half — recovering the
person share directly.

> Cue: Spearman-Brown curve — lengthen the test

And the Spearman-Brown relation makes the design knob explicit: lengthen the
test and reliability rises, with diminishing returns. More items, more
signal — the same lesson the variance decomposition told us at the start.

> Cue: SEM band around a score

Finally, the standard error of measurement turns reliability into a band
around each score — how much it would wobble on re-testing.

[pause]

---

## PART 8 — CLOSING (~1:00)

### 8.1 Summary

**NARRATOR:**

> [On screen: key takeaways, clean text on dark background]

Let us step back.

Reliability is the prior question to validity: does the evaluation give the
same answer twice? We answered it by decomposing variance into signal and
noise.

[pause]

The person component is the signal; reliability is its share of the total —
the generalizability coefficient. We estimate it three ways: bias-corrected
plug-in, Bayesian shrinkage, and method of moments.

Reliability is local — it collapses for frontier models on an easy pool — and
it is a design target: G- and D-studies size items and raters to hit it.

For LLM judges, Cohen's kappa measures agreement, but reliability cannot see
systematic bias. Consistency is not correctness.

[pause]

Reliability is what makes a measurement worth trusting before we ask whether
it measures the right thing.

> [On screen: "AIMS — AI Measurement Science" / "aimslab.stanford.edu"]

---

## Animation-Scene Mapping

| Script section | Animation file | Scene name |
|----------------|----------------|------------|
| 1.1 Opening Hook | `leaderboard_flicker.py` | `LeaderboardFlicker` |
| 2.1 Variance Components | `variance_decomposition.py` | `VarianceDecomposition` |
| 3.1 Three Estimators | `three_estimators.py` | `ThreeEstimators` |
| 4.1 Conditional Reliability | `conditional_reliability.py` | `ConditionalReliability` |
| 5.1 G/D-studies | `g_d_study.py` | `GandDStudy` |
| 6.1 Judge Kappa | `judge_kappa.py` | `JudgeKappa` |
| 7.1 Spearman-Brown | `spearman_brown.py` | `SpearmanBrown` |

## Rendering All Animations

```bash
PATH="/lfs/local/0/sttruong/miniconda3/bin:$PATH"

for scene in \
  "leaderboard_flicker.py LeaderboardFlicker" \
  "variance_decomposition.py VarianceDecomposition" \
  "three_estimators.py ThreeEstimators" \
  "conditional_reliability.py ConditionalReliability" \
  "g_d_study.py GandDStudy" \
  "judge_kappa.py JudgeKappa" \
  "spearman_brown.py SpearmanBrown"; do
  set -- $scene
  manim -qh --disable_caching --media_dir media/ch5 animations/ch5/$1 $2
done
```
