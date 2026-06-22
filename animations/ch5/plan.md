# Reliability — Video Plan (Chapter 5)

**Chapter 5 of AI Measurement Science (AIMS)**
Target length: ~10–12 minutes
Format: Narrated animation (3Blue1Brown style), built the same way as Chapter 2.

This plan mirrors the Chapter 2 production: an opening hook, **7 content
animations** (one Manim `.py` per concept), branded title cards between parts,
and a narration script with `[ANIMATION]` / `Cue:` markers. See
`../HOWTO.md` for the full pipeline and `../ch2/` for a worked reference.

---

## The chapter's spine (what the video must convey)

Chapter 5 answers the *prior* question to validity: **does the evaluation give
the same answer when applied to the same thing twice?** The whole chapter is one
idea developed in layers:

1. **Leaderboards flicker** (`@sec-leaderboards-flicker`) — same models, repeated
   runs, rankings shuffle and CIs overlap. Reliability ≠ validity.
2. **Decompose the variance** (`@sec-response-to-variance`) — split a benchmark's
   response variance into **person + item + person×item + residual**, straight
   from the IRT response model. Reliability = the **person (signal) share** =
   the *generalizability coefficient* $\rho = V_{\text{person}}/\operatorname{Var}(Y)$.
3. **Estimate the components** three ways (`@sec-plug-in`, `@sec-bayesian`,
   `@sec-method-of-moments`) — plug-in (with the $\overline{\mathrm{SE}^2}$ bias
   correction), Bayesian (shrinkage / posterior), and method of moments (ANOVA
   mean squares). One trial fuses interaction with Bernoulli noise; replication
   separates them.
4. **Reliability varies with ability** (`@sec-bayesian`, conditional-reliability
   fig) — $\rho(\theta)$ collapses for frontier (high-$\theta$) models if the
   item pool is too easy.
5. **Design for it** (`@sec-gtheory`) — G-study estimates the components, D-study
   trades **items vs raters** to hit a target $G \ge 0.90$.
6. **Judges add a facet** (`@sec-ordinal`) — LLM-as-a-judge: chance-corrected
   agreement (Cohen's $\kappa$), and why *systematic* bias (position bias)
   escapes reliability entirely.
7. **Bridge to classical coefficients** (`@sec-ctt`) — Cronbach's $\alpha$,
   Spearman–Brown (longer test → higher reliability), SEM.

The arc is identical in shape to Chapter 2's "theory → estimation → design,"
which makes it a natural sequel visually.

---

## Storyboard — 7 content animations

Each tracks a figure that already exists in `src/chap5.qmd`, so the math is
already settled; the animation just makes it move.

### 1. `leaderboard_flicker.py — LeaderboardFlicker`  (~45s) · Part 1
**Tracks:** `nd-ranking`, `nd-intervals` figures; `@sec-leaderboards-flicker`.
- Act 1: A leaderboard of ~6 models. Run it three times — rows reorder, lines
  cross. "Same models. Same benchmark. Different answer."
- Act 2: Collapse the three runs into mean ± 95% CI bars; overlapping intervals
  highlighted in red. Rank differences that are within the noise.
- Act 3: Split-screen label — **Reliability** ("same answer twice?") vs
  **Validity** ("the *right* answer?", deferred to Ch. 6). Reliability comes first.
- Takeaway: "Before we ask if a benchmark measures the right thing, ask whether
  it measures *anything*."

### 2. `variance_decomposition.py — VarianceDecomposition`  (~60s) · Part 2
**Tracks:** `@eq-full-decomp`, the `va-decompose` figure.
- Act 1: A response matrix $Y_{ij}$ lights up; pull out one cell's log-odds
  $\eta_{ij} = \theta_i - \beta_j + \gamma_{ij}$.
- Act 2: Animate the variance splitting into four stacked bars —
  $V_{\text{person}}$, $V_{\text{item}}$, $V_{\text{interaction}}$,
  $V_{\text{residual}}$ (Bernoulli) — "additive, non-overlapping" (functional
  ANOVA / Hoeffding–Sobol).
- Act 3: Highlight only $V_{\text{person}}$ → the signal. Define
  $\rho = V_{\text{person}}/\operatorname{Var}(Y)$, the generalizability
  coefficient. For one binary trial the signal bar is *small* next to item +
  noise — motivating everything after.

### 3. `three_estimators.py — ThreeEstimators`  (~70s) · Part 3
**Tracks:** `plugin-estimator`, `bayes-estimator`, `va-identifiability` figures.
- Act 1 (Plug-in): Fit abilities as fixed effects; naive variance of fitted
  abilities **overshoots** the planted $\sigma_p^2=1$. Subtract
  $\overline{\mathrm{SE}^2}$ → recovers truth. Show the bias decaying like $1/M$
  as items accumulate.
- Act 2 (Bayesian): Prior × likelihood → posterior over $\sigma_p$; MAP vs full
  posterior mean with a shaded 95% credible band. Shrinkage in one frame.
- Act 3 (Method of moments): ANOVA mean squares $MS_p, MS_i, MS_{pi}$ → solve
  for components. No likelihood, no prior — the "bluntest but cheapest."
- Act 4 (Identifiability): With **one trial**, the interaction variance absorbs
  the Bernoulli residual and overshoots; sweep trials/cell → the two separate
  and converge to truth. "Replication is what splits signal's last two pieces."

> Note: this is the densest scene. If it runs long, split Act 4 into its own
> 30s `interaction_identifiability.py` clip (keeps each scene ≤60s per HOWTO).

### 4. `conditional_reliability.py — ConditionalReliability`  (~40s) · Part 4
**Tracks:** the `conditional-reliability` figure; `@sec-fisher-design` callback.
- Act 1: Plot $\rho(\theta)$ for an item pool clustered at easy difficulties —
  looks fine on average.
- Act 2: Slide a marker to high $\theta$ (frontier models): $\rho(\theta)$
  **collapses** — easy items barely inform a strong model.
- Act 3: Overlay a difficulty-matched pool; $\rho(\theta)$ stays high across the
  range. "Reliability is local; design items where you need to tell models apart."

### 5. `g_d_study.py — GandDStudy`  (~55s) · Part 5
**Tracks:** `@sec-gtheory`, the `gd-components` figure; multi-facet
$\sigma^2_p, \sigma^2_i, \sigma^2_r, \dots$.
- Act 1: A model×item×rater cube; G-study reads off the seven variance
  components as a bar chart.
- Act 2: D-study heatmap — reliability $G$ over a grid of (#items, #raters).
  Contour for $G = 0.90$.
- Act 3: Show the trade-off: when rater variance dominates, add raters; when item
  variance dominates, add items. Animate the cheapest path onto the 0.90 contour.

### 6. `judge_kappa.py — JudgeKappa`  (~50s) · Part 6
**Tracks:** `@sec-ordinal` — Cohen's $\kappa$ and position bias.
- Act 1: Two judges label the same responses; raw agreement looks high, but a lot
  is chance. Animate the chance-correction → $\kappa = (p_o - p_e)/(1 - p_e)$.
- Act 2: The rater facet as a *generalizability coefficient for judges*; more
  judges shrink rater variance (callback to the D-study).
- Act 3: **The catch.** A judge with position bias (always prefers option A).
  Show it agreeing with *itself* perfectly across runs → high reliability, but
  the measurement is *systematically wrong*. "Reliability cannot see bias; that's
  a validity question (Ch. 6)."

### 7. `spearman_brown.py — SpearmanBrown`  (~40s) · Part 7 / Closing bridge
**Tracks:** `@sec-ctt` — Cronbach's $\alpha$, Spearman–Brown, SEM.
- Act 1: When only total scores are available, the same reliability comes from
  classical coefficients: $\alpha$, split-half.
- Act 2: Spearman–Brown curve — lengthen the test, reliability rises with
  diminishing returns. Tie back to "more items → higher $\rho$" from Act 2.
- Act 3: SEM band around a score; close on the chapter's design principles.

---

## Part / title-card structure (mirrors `ch2/section_titles.py`)

| # | Clip | Scene |
|---|------|-------|
| 1 | `ch5_titles.py` | `ChapterOpening` |
| 2 | `ch5_titles.py` | `Part1Title` — When Leaderboards Flicker |
| 3 | `leaderboard_flicker.py` | `LeaderboardFlicker` |
| 4 | `ch5_titles.py` | `Part2Title` — Decomposing the Variance |
| 5 | `variance_decomposition.py` | `VarianceDecomposition` |
| 6 | `ch5_titles.py` | `Part3Title` — Estimating the Components |
| 7 | `three_estimators.py` | `ThreeEstimators` |
| 8 | `ch5_titles.py` | `Part4Title` — Reliability Is Local |
| 9 | `conditional_reliability.py` | `ConditionalReliability` |
| 10 | `ch5_titles.py` | `Part5Title` — Designing for Generalizability |
| 11 | `g_d_study.py` | `GandDStudy` |
| 12 | `ch5_titles.py` | `Part6Title` — Judges and Their Biases |
| 13 | `judge_kappa.py` | `JudgeKappa` |
| 14 | `ch5_titles.py` | `Part7Title` — Back to Classical Coefficients |
| 15 | `spearman_brown.py` | `SpearmanBrown` |
| 16 | `ch5_titles.py` | `ChapterClosing` |

~16 clips, comparable to Chapter 1 (14) and Chapter 2 (13). Estimated total
~10–11 min with narration.

---

## Design tokens & reuse

Copy the shared design tokens from `../HOWTO.md §2` (ACCENT `#FFD966`,
BG `#0f0f0f`, the 5-color PAL, 1080p60). Strong reuse opportunities from Chapter 2:
- `ch2/section_titles.py` → adapt to `ch5_titles.py` (just new part titles).
- `ch2/bayesian_inference.py` → the prior×likelihood→posterior machinery is
  directly reusable for `three_estimators.py` Act 2.
- `ch2/stitch_narrated.sh`, `ch2/generate_narration.py` → copy verbatim; only the
  `CLIPS` array and narration text change.
- The actual numbers/curves can be lifted from the chapter's executable cells
  (`plugin-estimator`, `bayes-estimator`, `va-decompose`, `gd-components`,
  `conditional-reliability`) so the video matches the figures exactly.

---

## Build order (recommended)

1. Write `script.md` (narration) following the `ch2/script.md` template.
2. Build + render the 7 content scenes individually at `-ql` to preview, then `-qh`.
3. Adapt `ch5_titles.py`; render the title cards.
4. Copy `stitch_narrated.sh`, set the `CLIPS` array per the table above.
5. Generate narration (edge-tts via `generate_narration.py`), stitch, review.
6. Embed in `src/chap5.qmd` right after the chapter heading, mirroring chap2:
   `{{< video ../animations/ch5/chapter5_narrated.mp4 >}}`

> **Numbering note.** The existing animation folders lag the displayed chapter by
> one (legacy of the chap0–9 → chap1–10 rename): `chap2.qmd` *Foundations* embeds
> `ch1/…`, `chap3.qmd` *Learning* embeds `ch2/…`. To match the new 1-indexed
> scheme, this plan uses `ch5/` for the Reliability chapter (`chap5.qmd`) — i.e.
> folder number = displayed chapter number, breaking the legacy offset on purpose.
> If you'd rather stay consistent with the offset, rename this folder to `ch4/`
> and embed `../animations/ch4/chapter4_narrated.mp4` instead.

Render command (per `HOWTO §4`):
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
