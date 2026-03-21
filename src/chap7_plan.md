# Chapter 7: Information and Mechanism Design --- Detailed Plan

## Overview

Chapter 7 bridges the measurement-theoretic foundations of Chapters 1--6 with the strategic and information-theoretic dimensions of AI evaluation. The central thesis: once a benchmark becomes influential, its design is no longer a purely statistical problem --- it is a mechanism design problem where evaluators and developers are strategic agents. The chapter draws on three primary sources:

1. **Guard paper** (Truong, Wang, et al.) --- "Incentive-Aligned Evaluation via Private Benchmark": Stackelberg evaluation game, information-variance tradeoff, distribution correction, holdout mechanisms.
2. **Wang 2024 thesis, Part II** --- Chapters 5--6: Counterfactual quality metrics with principal-agent models (Ch. 5), information elicitation in agency games with reveal/conceal/garble decisions (Ch. 6).
3. **Procaccia, Schiffer, Wang, Zhang 2025** --- "Metritocracy: Representative Metrics for Lite Benchmarks": Positional representation and positional proportionality for selecting representative metric subsets using social choice theory.

## Target Length

~1000--1100 lines (comparable to Ch. 5 at 1110 lines and Ch. 6 at 901 lines).

---

## Section Outline

### 0. YAML Header and ILOs (~50 lines)

Same structure as Ch. 5/6: pyodide setup, packages (numpy, matplotlib, scipy), ILOs, suggested lecture plan, notation table. Keep the 7 ILOs from the current placeholder, possibly refine wording to match the final section structure.

**Notation table additions:**
- $F$ = universe of tasks, $|F| = N$
- $F_E, F_M$ = evaluator's / builder's task sets
- $\pi_E, \pi_M$ = sampling distributions
- $\theta \in \Theta$ = model (parameters)
- $f(\theta)$ = task performance function
- $u_E(\theta) = \sum_{f \in F} f(\theta)$ = evaluator's utility (aggregate performance)
- $k$ = number of sampled evaluation tasks per round
- $\rho$ = distribution correction rate
- $\sigma^2$ = task score variance
- $\gamma$ = gaming penalty
- $\varepsilon, \eta$ = privacy / threshold parameters (holdout mechanism)
- $C$ = agent cost, $b$ = principal's value
- $g$ = group size (positional representation)
- $\epsilon$ = tolerance (positional proportionality)

---

### 1. When Measurement Becomes a Target (~120 lines) {#sec-goodhart}

**Purpose:** Motivate the chapter by showing that Goodhart's Law is not a metaphor but a formal game-theoretic phenomenon. Connect to validity concerns from Ch. 5 (contamination, CIV) and causal reasoning from Ch. 6.

**Content:**
- Open with Goodhart's Law quote and concrete AI examples: FMTI score inflation, Chatbot Arena strategic submission, benchmark contamination as rational behavior
- Four variants of Goodhart (Manheim & Garrabrant 2018): regressional, extremal, causal, adversarial
- Frame: the shift from "measurement error" (Chs. 3--4) and "validity threats" (Ch. 5) to "strategic manipulation" --- the DGP itself changes in response to the measurement
- Connection to performative prediction (Perdomo et al. 2020) and strategic classification (Hardt et al. 2016)
- Preview: the chapter builds three layers of strategic analysis: (i) benchmark disclosure and information design, (ii) metric selection and reporting granularity, (iii) mechanism design for repeated evaluation

**Cross-references:** @sec-contamination (Ch. 5), @sec-causal-contamination (Ch. 6), @sec-validity-threats (Ch. 5)

---

### 2. The Evaluation Game (~180 lines) {#sec-evaluation-game}

**Purpose:** Formalize the Stackelberg benchmark game from the Guard paper. Establish that deterministic benchmarks fail and randomization provides one-shot alignment.

#### 2.1 Setup: Evaluator and Builder (~50 lines) {#sec-game-setup}
- Task universe $F$, task performance function $f: \Theta \to [0,1]$
- Evaluator's utility $u_E(\theta) = \sum_{f \in F} f(\theta)$
- Builder's information set, $F_M$ vs $F_E$
- Stackelberg game timing: designer publishes mechanism $(M, r)$, builder responds with $\theta^*$, evaluation stage

**Key definition:** Stackelberg Benchmark Game (Definition, adapted from Guard paper Def. 3)

#### 2.2 Failure of Deterministic Mechanisms (~30 lines) {#sec-deterministic-failure}
- **Proposition** (Failure of Deterministic Mechanisms): If $S^*$ is deterministic and public, builder ignores tasks outside $S^*$
- Intuition: this is exactly Goodhart's adversarial variant
- Connection to Ch. 5 construct underrepresentation: a fixed benchmark systematically ignores capabilities outside $S^*$

#### 2.3 One-Shot Alignment via Randomization (~60 lines) {#sec-one-shot-alignment}
- **Theorem** (One-Shot Incentive Alignment): Under uniform prior and single-sample mechanism, builder's expected reward is proportional to $u_E(\theta)$
- Present both the omniscient version (Thm 1 of Guard) and the limited-information version (Thm 2 of Guard, where builder's prior matches their own sampling distribution)
- Proof sketch: symmetry argument, marginal probability of sampling any task is $1/|F|$
- ERM connection: incentive-aligned builder performs empirical risk minimization over $F_M$

**Code block idea:** Interactive simulation showing that under a random evaluation mechanism, a builder's optimal strategy converges to broad optimization, while under deterministic mechanism it converges to narrow specialization. Slider for $|F_E|/|F|$ ratio.

#### 2.4 Discussion: What Randomization Buys (~40 lines) {#sec-randomization-discussion}
- Connect to CAT and Fisher information from Ch. 3: randomized evaluation is not optimal for estimation precision but is optimal for incentive alignment
- Tension between statistical efficiency (Ch. 3 wants targeted items) and strategic robustness (this chapter wants unpredictable items)
- Connect to Bayesian persuasion: the evaluator is a sender choosing how much to reveal

---

### 3. The Information-Variance Tradeoff (~180 lines) {#sec-info-variance-tradeoff}

**Purpose:** Show that in repeated evaluation, randomization alone fails because information leaks. Formalize the fundamental tradeoff.

#### 3.1 Repeated Evaluation and Information Leakage (~60 lines) {#sec-repeated-game}
- **Definition** (Repeated Evaluation Game): rounds $t = 1, 2, \ldots$, fresh $F_E^{(t)}$ from $\pi_E$, builder observes $k$ tasks
- Information set $\mathcal{I}_t$, leakage $L_t = |\mathcal{I}_t|/|F|$
- Variance of $k$-task average: $\sigma^2/k$, required $k$ for distinguishing models grows as $\Delta \to 0$
- Market pressure: as models converge, $k$ must grow

#### 3.2 Posterior Concentration and Incentive Misalignment (~60 lines) {#sec-posterior-concentration}
- **Proposition** (Incentive Misalignment Under Distribution Learning): As $\hat{\pi}_{E,t}$ concentrates around $\pi_E$, builder specializes to high-density regions
- Bayesian posterior concentration rate: $O(d_{\text{eff}} \log m / m)$
- The asymptotic failure: $\theta_t^* \to \arg\max_\theta \mathbb{E}_{f \sim \pi_E}[f(\theta)]$, which differs from $u_E$ when $\pi_E \neq \text{Uniform}(F)$

#### 3.3 The Pareto Frontier (~60 lines) {#sec-pareto-frontier}
- Variance decreases in $k$, leakage increases in $k$: no strategy achieves both low variance and low leakage
- **Definition** (Residual Misalignment): $\Delta_t = \mathbb{E}_{f \sim \pi_E^{(t)}}[f(\theta_t^*)] - \frac{1}{|F|}u_E(\theta_t^*)$
- Visual: the Pareto frontier with $k$ on x-axis, variance and leakage on two y-axes

**Code block idea:** Simulation of the Pareto frontier. User chooses $k \in \{5, 10, 20, 50, 100\}$ and the simulation shows (a) evaluation variance over rounds, (b) builder's posterior KL divergence from true $\pi_E$ over rounds, (c) misalignment $\Delta_t$ over rounds. This directly replicates the Guard paper's simulation setup ($|F| = 200$, initially biased $\pi_E$).

**Cross-references:** @sec-fisher-information (Ch. 3), @sec-cat (Ch. 3)

---

### 4. Restoring Alignment (~180 lines) {#sec-restoring-alignment}

**Purpose:** Present the two complementary mechanisms that restore alignment: distribution correction and noise-gated holdout.

#### 4.1 Distribution Correction (~80 lines) {#sec-distribution-correction}
- **Assumption** (Distribution Correction): $\pi_E^{(t)} = (1-\rho)\pi_E^{(t-1)} + \rho \cdot \text{Uniform}(F)$
- **Proposition** (Alignment Recovery): as $\pi_E^{(t)} \to \text{Uniform}(F)$, builder's best response converges to maximizing $u_E(\theta)$
- **Proposition** (Misalignment Bound): $\Delta_t \leq \min\left(\frac{m_t}{m_t + |F|}, (1-\rho)^t\sqrt{D_0/2}\right)$
  - Two regimes: estimation-limited (small $t$) vs correction-limited (large $t$)
- Race between two learners: builder learning $\pi_E$ vs evaluator correcting $\pi_E$
- Practical grounding: coverage audits, incident reports, benchmark changelogs as the mechanism for correction

**Theorem presentation level:** State formally with proof sketch (Pinsker inequality for correction-limited term, Bayesian shrinkage for estimation-limited term). Full proofs deferred to exercises or appendix.

#### 4.2 Optimal Evaluation Size (~50 lines) {#sec-optimal-k}
- Evaluator's loss: $\mathcal{L}(k) = \sigma^2/k + \gamma k/\rho$
- **Proposition** (Optimal Sample Size): $k^* = \sigma\sqrt{\rho/\gamma}$
- Key insight: $k^*$ increases with $\sqrt{\rho}$, so faster correction allows larger evaluation sets
- Limiting behavior: $\rho \to 0$ gives $k^* \to 0$ (static regime), $\rho \to \infty$ gives $k^* \to \infty$ (unconstrained)
- Dynamic $k_t$ schedule: start cautious, open up as correction progresses

**Code block idea:** Interactive plot of $\mathcal{L}(k) = \sigma^2/k + \gamma k/\rho$ for different $\rho$ values. Sliders for $\sigma, \gamma, \rho$. Stars mark optimal $k^*$. Replicates Guard paper Figure 2 (right panel).

#### 4.3 Noise-Gated Holdout (~50 lines) {#sec-holdout}
- **Definition** (Holdout Evaluation Mechanism): public reference set $S_0$, threshold test, Laplace noise
- Key property: self-correcting --- reveals less when builder behaves well, more when gaming
- Effective leakage: $\lambda_{\text{out}}^{\text{holdout}} \approx e^{-\varepsilon\eta} \cdot k$
- Connection to differential privacy and Dwork et al. (2015) reusable holdout
- Why noise alone cannot replace distribution correction: DP slows learning but doesn't eliminate exploitable structure

**Cross-references:** @sec-three-estimators (Ch. 6, IPW/DR connections), @sec-covariate-shift (Ch. 6)

---

### 5. Metric Design as Principal-Agent Problem (~150 lines) {#sec-metric-design}

**Purpose:** Shift from "which tasks to show" to "which metrics to report." Draw on Wang Ch. 5 (counterfactual metrics) and Ch. 6 (information elicitation).

#### 5.1 When Metrics Create Perverse Incentives (~50 lines) {#sec-perverse-incentives}
- Motivating example: hospital mortality metrics (Dranove et al. 2003) --- publish average treated outcome, hospitals cream the easiest patients
- AI analogy: leaderboard that rewards average benchmark score incentivizes developers to specialize on easy benchmarks
- Principal-agent model (from Wang Ch. 5): principal chooses reward function $w$, agent best-responds with policy $\pi^w$
- **Proposition** (ATO Regret): Average treated outcome has unbounded regret
- **Proposition** (TT Zero Regret): Total treatment effect achieves zero regret when principal has unbiased counterfactual estimates
- AI translation: scoring the "total capability effect" across all tasks, not the average on a selected subset

#### 5.2 Information Asymmetry and Metric Elicitation (~50 lines) {#sec-information-elicitation}
- Setup from Wang Ch. 6: agent (developer) has private cost-correlated variable $X$; principal (evaluator) can design contracts conditioned on $X$ if revealed
- Key question: when does the developer prefer to reveal, conceal, or garble information about their model's capabilities?
- **Agent's revelation incentives:** reveal when conditioning on $X$ sufficiently differentiates high and low costs (Propositions 6.1--6.3 from Wang)
- **Principal always benefits from revelation** (Lemma 6.2)
- Connection to benchmark transparency: developers may prefer partial transparency when it allows the evaluator to design better-targeted contracts

#### 5.3 Garbling as Differential Privacy (~50 lines) {#sec-garbling-privacy}
- Garbling mechanism: agent reveals $Y = X$ w.p. $\varepsilon$, $Y = \xi$ w.p. $1-\varepsilon$ (randomized response, cf. differential privacy)
- **Key result:** Under fairly wide conditions, agent prefers garbled signal over both full concealment and full revelation (Propositions 6.4, 6.5 from Wang)
- Pareto improvement: garbling can increase total welfare over concealment
- AI application: model cards with noise, differential privacy in metric reporting
- Connection to holdout mechanism (Section 4.3): both use calibrated noise to improve incentives
- **Lemma** (Optimal garbling increases welfare over concealment, Lemma 6.9 from Wang)

**Code block idea:** Simulation of the agency game. Two exponential cost types with different means $\lambda_0, \lambda_1$. Heatmap showing $V_{\text{rev}} - V_{\text{con}}$ as in Wang Figure 6.2. Slider for garbling parameter $\varepsilon$ showing how agent utility varies.

**Cross-references:** @sec-dif-theory (Ch. 5, DIF as a form of metric asymmetry), @sec-validity-taxonomy (Ch. 5)

---

### 6. Representative Benchmark Selection (~120 lines) {#sec-representative-selection}

**Purpose:** Address the "which metrics to include in a lite benchmark" question using social choice theory (Procaccia et al. 2025).

#### 6.1 The Subset Selection Problem (~30 lines) {#sec-subset-selection}
- Motivation: BIG-bench (200+ metrics) $\to$ BIG-bench Lite (24); HELM $\to$ HELM Lite; Cal Hospital Compare (hundreds $\to$ 12)
- Formal setup: $n$ metrics, $m$ alternatives (models), each metric $i$ gives a ranking $\sigma_i$
- Goal: select $K \subset N$ that is "representative" --- but what does representative mean?

#### 6.2 Positional Representation (~40 lines) {#sec-positional-representation}
- **Definition** (Positional Representation): For group size $g$, any alternative ranked in top $r$ by $\geq \ell \cdot g$ metrics must be ranked in top $r$ by $\geq \ell$ metrics in $K$
- Prevents under-representation at every rank cutoff
- **Theorem** (Bounds): $\Omega(n/g \cdot \log(m) / \log(n/g \cdot \log m)) \leq |K| \leq O(n/g \cdot \log m)$
- Greedy algorithm (Algorithm 1 from Procaccia): polynomial time, within $\log m$ factor of optimal
- Connection to set cover

#### 6.3 Positional Proportionality (~40 lines) {#sec-positional-proportionality}
- **Definition** (Positional Proportionality): For every alternative and rank cutoff, $|C(N,r,a)/|N| - C(K,r,a)/|K|| \leq \varepsilon$
- Prevents both under- and over-representation
- **Theorem** (Bounds): $\Omega(1/\varepsilon^2 \cdot \log m) \leq |K| \leq O(1/\varepsilon^2 \cdot \log m)$ --- tight up to constants
- **Theorem** (Scoring Rule Approximation): If $K$ satisfies $\varepsilon$-PP, any scoring rule on $\sigma_K$ approximates the same rule on $\sigma_N$ within $\varepsilon$
- Generalizations: groups by language, difficulty, category

**Code block idea:** Interactive demonstration on synthetic benchmark data. Generate $n = 50$ metrics ranking $m = 20$ models. Run the greedy algorithm for positional representation at various $g$ values. Show how the selected subset compares to random selection in preserving top-$k$ rankings.

#### 6.4 Connection to Benchmark Granularity (~10 lines) {#sec-granularity-connection}
- Link positional proportionality to ILO 3 (sum scores vs. subscores): PP gives a formal criterion for when a "lite" benchmark preserves the same information as the full suite
- Connection to reliability (Ch. 4): the subscore reliability problem is a special case of positional proportionality when metrics are test items

**Cross-references:** @sec-dimensionality (Ch. 5), @sec-item-construction (Ch. 5), @sec-g-theory (Ch. 4)

---

### 7. Synthesis: Design Principles for Strategic Benchmarks (~60 lines) {#sec-design-principles}

**Purpose:** Unify the three threads into actionable design principles for AI evaluation.

- **Principle 1: Randomize and Refresh** --- Static benchmarks are Goodhart-vulnerable by construction. Use randomized evaluation with task renewal (Section 2--3).
- **Principle 2: Correct and Grow** --- Invest in distribution correction ($\rho$) as the primary lever; $k^*$ scales with $\sqrt{\rho}$ (Section 4.1--4.2).
- **Principle 3: Gate Information Release** --- Use holdout mechanisms with threshold tests to condition information flow on builder behavior (Section 4.3).
- **Principle 4: Align Metrics with Welfare** --- Use total treatment effect scoring, not averages; account for counterfactual baselines (Section 5.1).
- **Principle 5: Allow Partial Transparency** --- Garbling / differential privacy can create Pareto improvements when full transparency or full opacity are both suboptimal (Section 5.2--5.3).
- **Principle 6: Ensure Representative Subsets** --- When creating lite benchmarks, use formal representation criteria (positional representation or proportionality) rather than ad hoc selection (Section 6).

Table: Map each principle to the relevant formal result and a concrete recommendation for benchmark designers.

---

### 8. Discussion Questions (~30 lines) {#sec-design-discussion}

6--8 discussion questions, e.g.:
1. The guard paper assumes a benevolent evaluator. What changes if the evaluator also has strategic incentives (e.g., a company running its own benchmark)?
2. How does the information-variance tradeoff relate to the reliability-validity tradeoff from Chapters 4--5?
3. Can you design a mechanism where the builder's incentive is to improve on the *hardest* tasks rather than the average?
4. In what sense is the Chatbot Arena a randomized evaluation mechanism? Does it satisfy the conditions of Theorem 1?
5. How should a government regulator set $\gamma$ (the gaming penalty) for a safety benchmark?
6. If two "lite" benchmarks satisfy positional proportionality with the same $\varepsilon$ but select different subsets, which should be preferred?

---

### 9. Bibliographic Notes (~30 lines) {#sec-design-biblio}

- Bayesian persuasion: Kamenica & Gentzkow (2011), Bergemann & Morris (2019)
- Maxmin Expected Utility: Gilboa & Schmeidler (1989), Ellsberg (1961)
- Mechanism design: Myerson (1981), robust mechanism design (Bergemann & Morris 2005)
- Strategic classification: Hardt et al. (2016), Perdomo et al. (2020), Braverman & Garg (2020)
- Contract theory: Laffont & Tirole (1986), Holmstrom & Milgrom (1991)
- Differential privacy and adaptive data analysis: Dwork et al. (2015), Dwork & Roth (2014), Blum & Hardt (2015)
- Social choice and benchmarking: Zhang & Hardt (2024), Colombo et al. (2022), Rofin et al. (2022)
- Goodhart's Law: Goodhart (1984), Manheim & Garrabrant (2018), Gao et al. (2023)

---

### 10. Exercises (~40 lines) {#sec-design-exercises}

8--10 exercises at varying difficulty:

1. **(Easy)** Show that if $\pi_E = \text{Uniform}(F)$, the one-shot alignment theorem holds for any sample size $k$, not just $k = 1$.
2. **(Easy)** Verify that $k^* = \sigma\sqrt{\rho/\gamma}$ minimizes $\mathcal{L}(k) = \sigma^2/k + \gamma k/\rho$. What is $\mathcal{L}(k^*)$?
3. **(Medium)** Derive the misalignment bound's estimation-limited term using a Dirichlet$(1,\ldots,1)$ prior and $m$ effective observations.
4. **(Medium)** In the agency game with binary $X$, suppose $C|X=0 \sim \text{Exp}(1/\lambda_0)$ and $C|X=1$ is zero-cost. Derive the condition under which the agent prefers concealment (cf. Proposition 6.1).
5. **(Medium)** Show that any subset $K$ satisfying $\varepsilon$-positional proportionality approximates any scoring rule within $\varepsilon$ (prove Theorem 3.4 from Procaccia et al.).
6. **(Hard)** Extend the one-shot alignment theorem to the case where the builder has a non-uniform prior $p_M$ over $F$ (Theorem 2 from Guard paper).
7. **(Hard)** Prove that the holdout mechanism slows posterior concentration by a factor of $e^{\varepsilon\eta}$.
8. **(Hard, computational)** Implement the greedy algorithm for positional representation on the BIG-bench Lite data. Compare $|K|$ to the existing 24-metric subset.

---

## Key Theorems/Results to Present (with formality level)

| Result | Source | Formality |
|--------|--------|-----------|
| Failure of deterministic mechanisms | Guard Prop. 1 | Full statement + short proof |
| One-shot incentive alignment (omniscient) | Guard Thm. 1 | Full statement + proof sketch |
| One-shot alignment (limited info) | Guard Thm. 2 | Full statement, proof deferred |
| Posterior concentration $\Rightarrow$ misalignment | Guard Prop. 2 | Statement + intuition |
| Misalignment bound (two-regime) | Guard Prop. 4.3 | Full statement + proof sketch |
| Optimal $k^*$ under correction | Guard Prop. 4.4 | Full statement + derivation |
| ATO unbounded regret | Wang Prop. 5.1 | Statement + example |
| TT zero regret | Wang Prop. 5.3 | Statement + one-line proof |
| Agent prefers garbling | Wang Prop. 6.4/6.5 | Statement + intuition |
| Garbling increases welfare | Wang Lemma 6.9 | Statement only |
| Principal always prefers revelation | Wang Lemma 6.2 | Statement + short proof |
| Positional representation bounds | Procaccia Thm. 2.2/2.3 | Statement + algorithm |
| Positional proportionality bounds | Procaccia Thm. 3.2/3.3 | Statement + proof sketch |
| Scoring rule approximation | Procaccia Thm. 3.4 | Statement + proof sketch |

---

## Code Block Ideas (pyodide-python)

1. **Deterministic vs. randomized evaluation** (~40 lines): Simulate builder's optimal strategy under deterministic $S^*$ (specializes) vs. random draw (generalizes). Bar chart of task-level performance.

2. **Information-variance Pareto frontier** (~60 lines): Monte Carlo simulation over 200 rounds. Varies $k$. Plots variance, leakage, and misalignment. Replicates Guard paper Figure 1 concept.

3. **Distribution correction and alignment recovery** (~50 lines): Simulate the repeated game with correction rates $\rho \in \{0, 0.01, 0.05, 0.1, 0.2\}$. Two panels: misalignment $\Delta_t$ and KL divergence. Replicates Guard paper Figure 3.

4. **Optimal $k$ loss surface** (~30 lines): Plot $\mathcal{L}(k)$ for different $\rho$ values with sliders. Show optimal $k^*$.

5. **Agency game: reveal vs. conceal vs. garble** (~50 lines): Heatmap of $V_{\text{rev}} - V_{\text{con}}$ for exponential distributions. Slider for garbling $\varepsilon$.

6. **Greedy algorithm for positional representation** (~40 lines): Implement Algorithm 1 from Procaccia on synthetic data. Show how $|K|$ varies with $g$.

---

## Cross-References to Earlier Chapters

| This chapter section | Earlier chapter reference | Nature of connection |
|---------------------|-------------------------|---------------------|
| Sec. 1 (Goodhart) | Ch. 5 @sec-contamination, @sec-civ | Contamination as rational strategic behavior |
| Sec. 2 (Evaluation game) | Ch. 3 @sec-fisher-information | Tension: Fisher info wants targeted items, incentives want random items |
| Sec. 2.3 (ERM) | Ch. 2 @sec-mle | Builder's optimization is ERM from Ch. 2 |
| Sec. 3 (Info-variance) | Ch. 3 @sec-cat | CAT reveals info about ability *and* about the item bank |
| Sec. 4 (Correction) | Ch. 6 @sec-covariate-shift | Distribution correction is like reweighting under covariate shift |
| Sec. 4.3 (Holdout) | Ch. 6 @sec-three-estimators | Holdout uses IPW-like noise calibration |
| Sec. 5 (Metrics) | Ch. 5 @sec-validity-taxonomy | Metric design as consequential validity |
| Sec. 5.3 (Garbling) | Ch. 4 @sec-sem | Noise in metric reporting vs. measurement error in SEM |
| Sec. 6 (Subset selection) | Ch. 5 @sec-dimensionality | Subset selection parallels dimension reduction |
| Sec. 6 (Subset selection) | Ch. 4 @sec-g-theory | Subscore reliability $\leftrightarrow$ positional proportionality |

---

## Estimated Line Counts

| Section | Lines |
|---------|-------|
| 0. Header, ILOs, notation | 50 |
| 1. When Measurement Becomes a Target | 120 |
| 2. The Evaluation Game | 180 |
| 3. The Information-Variance Tradeoff | 180 |
| 4. Restoring Alignment | 180 |
| 5. Metric Design as Principal-Agent Problem | 150 |
| 6. Representative Benchmark Selection | 120 |
| 7. Design Principles Synthesis | 60 |
| 8. Discussion Questions | 30 |
| 9. Bibliographic Notes | 30 |
| 10. Exercises | 40 |
| **Total** | **~1140** |

---

## References to Add to `references.bib`

### From Guard paper
```
@article{goodhart1984problems, ...}  % Goodhart's Law
@article{manheim2018categorizing, ...}  % Four Goodhart variants
@article{gao2023scaling, ...}  % Scaling laws for reward overoptimization
@article{karwowski2023goodhart, ...}  % Goodhart in RL
@inproceedings{hardt2016strategic, ...}  % Strategic classification
@article{perdomo2020performative, ...}  % Performative prediction
@article{braverman2020role, ...}  % Role of randomness in strategic classification
@inproceedings{kleinberg2020classifiers, ...}  % Classifiers as decision-makers
@article{ederer2018gaming, ...}  % Strategic opacity / gaming
@inproceedings{dwork2015generalization, ...}  % Generalization in adaptive data analysis
@inproceedings{dwork2015reusable, ...}  % Reusable holdout
@book{dwork2014algorithmic, ...}  % Algorithmic foundations of DP
@inproceedings{blum2015ladder, ...}  % Ladder mechanism
@article{livebench2024, ...}  % LiveBench
@article{lmsys2024arenahard, ...}  % Arena-Hard
@article{vapnik1998statistical, ...}  % Statistical learning theory
```

### From Wang 2024 thesis
```
@phdthesis{wang2024metrics, ...}  % Wang thesis
@article{kamenica2011bayesian, ...}  % Bayesian persuasion
@article{bergemann2019information, ...}  % Information design survey
@book{laffont2009theory, ...}  % Contract theory
@article{laffont1986using, ...}  % Laffont-Tirole agency
@article{holmstrom1991multitask, ...}  % Holmstrom-Milgrom multitask
@article{holmstrom1979moral, ...}  % Sufficient statistic theorem
@article{milgrom1981good, ...}  % Good news and bad news
@article{bernheim1998incomplete, ...}  % Strategic ambiguity
@article{blackwell1951comparison, ...}  % Blackwell informativeness
@article{varian1985price, ...}  % Price discrimination and welfare
@article{bergemann2015limits, ...}  % Limits of price discrimination
@article{dranove2003paying, ...}  % Hospital mortality metrics
@article{aguirre2010monopoly, ...}  % Monopoly price discrimination welfare
```

### From Procaccia et al. 2025
```
@article{procaccia2025metritocracy, ...}  % Main paper
@article{zhang2024inherent, ...}  % Arrow's impossibility for benchmarks
@inproceedings{aziz2017justified, ...}  % Justified representation
@article{colombo2022what, ...}  % Borda count for benchmarks
@article{rofin2022votenrank, ...}  % Vote'N'Rank
@inproceedings{skowron2015proportional, ...}  % Proportional representation
@article{polo2024tinybenchmarks, ...}  % TinyBenchmarks
```

### Additional foundations
```
@article{gilboa1989maxmin, ...}  % Maxmin Expected Utility
@article{ellsberg1961risk, ...}  % Ellsberg paradox
@article{myerson1981optimal, ...}  % Optimal auction design
@article{bergemann2005robust, ...}  % Robust mechanism design
```

---

## Notes on Tone and Style

- Match the semi-formal style of Chapters 5--6: definitions in callout boxes, theorems stated precisely but proofs often sketched or deferred
- Use the "AI evaluation" framing throughout, with hospital/education analogies for intuition (following Wang's approach)
- Keep code blocks interactive and self-contained via pyodide (numpy, matplotlib, scipy only)
- Use callout boxes (`:::{.callout-tip}`, `:::{.callout-important}`) for key insights and caveats
- Each major section should open with a motivating question or example before formalizing

## Notes on What to Omit or Defer

- **Maxmin Expected Utility / Ellsberg paradox:** ILO 1 mentions this, but the Guard paper's framework is more directly applicable. Treat MEU as a brief motivating example in Section 1 (3--5 paragraphs) rather than a full development. The Stackelberg game subsumes the MEU framing.
- **Bayesian persuasion / concavification:** ILO 2 mentions this. Cover at the level of intuition in Section 2.4 (connection to information design) and Section 5.2 (Wang's persuasion game framing), but do not develop the full concavification machinery. The chapter already has enough technical depth.
- **Gap-targeted Gaussian correction:** The Guard paper's Assumption 2 (gap-targeted correction) is elegant but complex. Mention it as an alternative to linear mixing in Section 4.1; defer details to bibliographic notes.
- **Wang Ch. 5 ranking results:** The multi-agent ranking with distributional reweighting (Theorems 5.1--5.2) is useful but secondary. Mention in Section 5.1 as an extension; do not develop the full importance-weighting machinery.
- **Procaccia generalizations:** The generalized representation (Definition 4.1/4.2) and NP-hardness results are secondary. Mention in Section 6.3 as extensions; focus on the core positional representation and proportionality definitions.
