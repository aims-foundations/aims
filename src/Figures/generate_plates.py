"""Generate plate diagrams for Chapter 1 using standalone TikZ compilation.

Produces high-quality plate diagrams with proper LaTeX math fonts,
rounded plate corners, and consistent sizing across all diagrams.

Layout convention for IRT/factor models (crossed design):
  - Person plate (i) and item plate (j) are separate rectangles
  - The two plates overlap only at Y_{ij}
  - Person parameter (theta_i / U_i) is in the i-plate only
  - Item parameters (beta_j, a_j, ...) are in the j-plate only
  - Y_{ij} sits in the overlap of both plates

Layout convention for hierarchical models (nested design):
  - Outer plate for benchmarks (j), inner plate for items (i)
  - Genuine nesting (items within benchmarks)

Usage:
    python generate_plates.py
"""
import os
import subprocess
import tempfile
import shutil

OUT = os.path.dirname(os.path.abspath(__file__))
DPI = 300

# ─── Common TikZ preamble ───────────────────────────────────────────
PREAMBLE = r"""
\documentclass[tikz,border=8pt]{standalone}
\usepackage{amsmath,amssymb}
\usetikzlibrary{arrows.meta,positioning,fit,backgrounds,calc,decorations.pathreplacing}

% ── Node styles ──
\tikzset{
  latent/.style={
    circle, draw=black, thick, minimum size=28pt,
    inner sep=1pt, fill=white, font=\normalsize
  },
  observed/.style={
    circle, draw=black, thick, minimum size=28pt,
    inner sep=1pt, fill=black!15, font=\normalsize
  },
  plate/.style={
    draw=black!70, rounded corners=5pt, thick,
    inner sep=10pt
  },
  platebox/.style={
    draw=black!70, rounded corners=5pt, thick
  },
  platelabel/.style={
    font=\small
  },
  >={Stealth[length=5pt]}
}
"""


def compile_tikz(name: str, body: str) -> None:
    """Compile a TikZ body to PNG via pdflatex + pdftoppm."""
    tex = PREAMBLE + r"\begin{document}" + "\n" + body + "\n" + r"\end{document}"

    with tempfile.TemporaryDirectory() as tmp:
        tex_path = os.path.join(tmp, "diagram.tex")
        pdf_path = os.path.join(tmp, "diagram.pdf")
        with open(tex_path, "w") as f:
            f.write(tex)

        # Compile LaTeX
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmp, tex_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            print(f"  ERROR compiling {name}:")
            for line in result.stdout.split("\n")[-30:]:
                if line.strip():
                    print(f"    {line}")
            return

        # Convert PDF -> PNG
        out_prefix = os.path.join(tmp, "out")
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(DPI), pdf_path, out_prefix],
            capture_output=True, timeout=30,
        )
        src_png = out_prefix + "-1.png"
        dst_png = os.path.join(OUT, name)
        shutil.copy2(src_png, dst_png)
        print(f"  saved {name}")


# ═══════════════════════════════════════════════════════════════════════
# 1. Rasch (1PL) — crossed plates
#    i-plate contains θ_i and Y_{ij}
#    j-plate contains β_j and Y_{ij}
#    Plates overlap only at Y_{ij}
# ═══════════════════════════════════════════════════════════════════════
RASCH = r"""
\begin{tikzpicture}
  % Person parameter (i-plate only)
  \node[latent]   (theta) at (0, 2.2)   {$\theta_i$};
  % Item parameter (j-plate only)
  \node[latent]   (beta)  at (3.5, 2.2) {$\beta_j$};
  % Observed response (overlap of both plates)
  \node[observed] (Y)     at (1.75, 0)  {$Y_{ij}$};

  % Edges
  \draw[->] (theta) -- (Y);
  \draw[->] (beta)  -- (Y);

  % i-plate (persons) — overlap 1.6, clearance 0.3 around Y
  \draw[platebox] (-0.7, -1.4) rectangle (2.55, 3.0);
  \node[platelabel] at (-0.55, -1.25) [above right] {$i{=}1,\ldots,N$};

  % j-plate (items) — offset upward
  \draw[platebox] (0.95, -1.0) rectangle (4.2, 3.4);
  \node[platelabel] at (2.7, -0.85) [above right] {$j{=}1,\ldots,M$};
\end{tikzpicture}
"""

# ═══════════════════════════════════════════════════════════════════════
# 2. Two-Parameter Logistic (2PL) — crossed plates
# ═══════════════════════════════════════════════════════════════════════
TWO_PL = r"""
\begin{tikzpicture}
  % Person parameter (i-plate only)
  \node[latent]   (theta) at (0, 2.2)   {$\theta_i$};
  % Item parameters (j-plate only) — placed right of i-plate edge (3.3)
  \node[latent]   (beta)  at (4.0, 2.2) {$\beta_j$};
  \node[latent]   (a)     at (5.5, 2.2) {$a_j$};
  % Observed response (overlap)
  \node[observed] (Y)     at (2.5, 0)   {$Y_{ij}$};

  % Edges
  \draw[->] (theta) -- (Y);
  \draw[->] (beta)  -- (Y);
  \draw[->] (a)     -- (Y);

  % i-plate (persons) — overlap 1.6, clearance 0.3 around Y
  \draw[platebox] (-0.7, -1.4) rectangle (3.3, 3.0);
  \node[platelabel] at (-0.55, -1.25) [above right] {$i{=}1,\ldots,N$};

  % j-plate (items) — offset upward
  \draw[platebox] (1.7, -1.0) rectangle (6.2, 3.4);
  \node[platelabel] at (3.45, -0.85) [above right] {$j{=}1,\ldots,M$};
\end{tikzpicture}
"""

# ═══════════════════════════════════════════════════════════════════════
# 3. Three-Parameter Logistic (3PL) — crossed plates
# ═══════════════════════════════════════════════════════════════════════
THREE_PL = r"""
\begin{tikzpicture}
  % Person parameter (i-plate only)
  \node[latent]   (theta) at (0, 2.2)   {$\theta_i$};
  % Item parameters (j-plate only) — placed right of i-plate edge (4.3)
  \node[latent]   (a)     at (5.0, 2.2) {$a_j$};
  \node[latent]   (beta)  at (6.3, 2.2) {$\beta_j$};
  \node[latent]   (c)     at (7.6, 2.2) {$c_j$};
  % Observed response (overlap)
  \node[observed] (Y)     at (3.5, 0)   {$Y_{ij}$};

  % Edges
  \draw[->] (theta) -- (Y);
  \draw[->] (a)     -- (Y);
  \draw[->] (beta)  -- (Y);
  \draw[->] (c)     -- (Y);

  % i-plate (persons) — overlap 1.6, clearance 0.3 around Y
  \draw[platebox] (-0.7, -1.4) rectangle (4.3, 3.0);
  \node[platelabel] at (-0.55, -1.25) [above right] {$i{=}1,\ldots,N$};

  % j-plate (items) — offset upward
  \draw[platebox] (2.7, -1.0) rectangle (8.3, 3.4);
  \node[platelabel] at (4.45, -0.85) [above right] {$j{=}1,\ldots,M$};
\end{tikzpicture}
"""

# ═══════════════════════════════════════════════════════════════════════
# 4. Logistic Factor Model — crossed plates
# ═══════════════════════════════════════════════════════════════════════
FACTOR = r"""
\begin{tikzpicture}
  % Person parameter (i-plate only)
  \node[latent]   (U) at (0, 2.2)   {$U_i$};
  % Item parameters (j-plate only) — placed right of i-plate edge (3.3)
  \node[latent]   (V) at (4.0, 2.2) {$V_j$};
  \node[latent]   (Z) at (5.5, 2.2) {$Z_j$};
  % Observed response (overlap)
  \node[observed] (Y) at (2.5, 0)   {$Y_{ij}$};

  % Edges
  \draw[->] (U) -- (Y);
  \draw[->] (V) -- (Y);
  \draw[->] (Z) -- (Y);

  % i-plate (persons) — overlap 1.6, clearance 0.3 around Y
  \draw[platebox] (-0.7, -1.4) rectangle (3.3, 3.0);
  \node[platelabel] at (-0.55, -1.25) [above right] {$i{=}1,\ldots,N$};

  % j-plate (items) — offset upward
  \draw[platebox] (1.7, -1.0) rectangle (6.2, 3.4);
  \node[platelabel] at (3.45, -0.85) [above right] {$j{=}1,\ldots,M$};
\end{tikzpicture}
"""

# ═══════════════════════════════════════════════════════════════════════
# 5. Bradley-Terry
#    Single plate for comparisons, symmetric layout
# ═══════════════════════════════════════════════════════════════════════
BRADLEY_TERRY = r"""
\begin{tikzpicture}
  % Nodes
  \node[latent]   (ti) {$\theta_i$};
  \node[latent]   (tj) [right=24mm of ti] {$\theta_j$};
  \node[observed] (Y)  [below=18mm of $(ti)!0.5!(tj)$] {$Y_c$};

  % Edges
  \draw[->] (ti) -- (Y);
  \draw[->] (tj) -- (Y);

  % Single plate
  \node[plate, fit=(ti)(tj)(Y),
        label={[platelabel, anchor=south east]below right:$c{=}1,\ldots,C$}] {};
\end{tikzpicture}
"""

# ═══════════════════════════════════════════════════════════════════════
# 6. Ising / Network Model (undirected, no latent variables)
# ═══════════════════════════════════════════════════════════════════════
ISING = r"""
\begin{tikzpicture}[node distance=20mm]
  % Nodes (all observed)
  \node[observed] (y1) {$Y_1$};
  \node[observed] (y2) [right=of y1] {$Y_2$};
  \node[observed] (y3) [right=of y2] {$Y_3$};
  \node[observed] (y4) [below=of y1] {$Y_4$};
  \node[observed] (y5) [below=of y2] {$Y_5$};

  % Undirected edges
  \draw[-] (y1) -- node[above, font=\small] {$\omega_{12}$} (y2);
  \draw[-] (y2) -- (y3);
  \draw[-] (y1) -- (y4);
  \draw[-] (y2) -- (y4);
  \draw[-] (y2) -- (y5);
  \draw[-] (y4) -- (y5);
  \draw[-] (y3) -- (y5);
\end{tikzpicture}
"""

# ═══════════════════════════════════════════════════════════════════════
# 7. Hierarchical IRT (three-level nesting)
#    Domain hyperparams outside all plates;
#    outer plate for benchmarks (j), inner plate for items (i)
# ═══════════════════════════════════════════════════════════════════════
HIERARCHICAL = r"""
\begin{tikzpicture}
  % Domain-level hyperparameters (outside all plates)
  \node[latent] (mu0) {$\mu_0$};
  \node[latent] (tau) [right=14mm of mu0] {$\tau$};

  % Benchmark-level parameters
  \node[latent] (muj)    [below=16mm of mu0]  {$\mu_j$};
  \node[latent] (sigmaj) [right=14mm of muj]  {$\sigma_j$};

  % Item difficulty
  \node[latent] (b) [below=16mm of muj] {$b_{ij}$};

  % Observed response
  \node[observed] (Y) [below=20mm of b] {$Y_{ij}$};

  % Person ability (outside all plates)
  \node[latent] (theta) [left=22mm of Y] {$\theta$};

  % Edges
  \draw[->] (mu0)    -- (muj);
  \draw[->] (tau)    -- (muj);
  \draw[->] (muj)    -- (b);
  \draw[->] (sigmaj) -- (b);
  \draw[->] (b)      -- (Y);
  \draw[->] (theta)  -- (Y);

  % Inner plate (items within benchmark)
  \node[plate, fit=(b)(Y),
        label={[platelabel, anchor=south west]below left:$i{=}1,\ldots,n_j$}] (inner) {};
  % Outer plate (benchmarks)
  \node[plate, fit=(muj)(sigmaj)(inner),
        label={[platelabel, anchor=south east]below right:$j{=}1,\ldots,J$}] {};
\end{tikzpicture}
"""

# ═══════════════════════════════════════════════════════════════════════
# 8. Nomological network (Chapter 6, Validity) — measurement model
#    Two latent constructs (theta_1, theta_2) over four observed items:
#      item 1 loads on theta_1; item 2 cross-loads on both; items 3,4 on theta_2.
#    Each item also has an item-specific latent (unique factor) z_j.
#    An intervention I_k points into each construct (do on the causes of theta_k).
# ═══════════════════════════════════════════════════════════════════════
NOMOLOGICAL = r"""
\begin{tikzpicture}
  % Interventions on the causes of each construct
  \node[draw=black!70, thick, rounded corners=3pt,
        minimum height=24pt, inner sep=5pt, fill=black!5] (I1) at (-2.6, 2)  {$I_1$};
  \node[draw=black!70, thick, rounded corners=3pt,
        minimum height=24pt, inner sep=5pt, fill=black!5] (I2) at (-2.6, -1) {$I_2$};
  % Latent constructs
  \node[latent] (t1) at (0.4, 2)  {$\theta_1$};
  \node[latent] (t2) at (0.4, -1) {$\theta_2$};
  % Observed items
  \node[observed] (Y1) at (4, 3)  {$Y_1$};
  \node[observed] (Y2) at (4, 1)  {$Y_2$};
  \node[observed] (Y3) at (4, -1) {$Y_3$};
  \node[observed] (Y4) at (4, -3) {$Y_4$};
  % Item-specific latent variables (unique factors)
  \node[latent, minimum size=20pt, font=\small] (z1) at (6.4, 3)  {$z_1$};
  \node[latent, minimum size=20pt, font=\small] (z2) at (6.4, 1)  {$z_2$};
  \node[latent, minimum size=20pt, font=\small] (z3) at (6.4, -1) {$z_3$};
  \node[latent, minimum size=20pt, font=\small] (z4) at (6.4, -3) {$z_4$};

  % Interventions -> constructs
  \draw[->] (I1) -- (t1);
  \draw[->] (I2) -- (t2);
  % Construct 1 -> items 1, 2
  \draw[->] (t1) -- (Y1);
  \draw[->] (t1) -- (Y2);
  % Construct 2 -> items 2, 3, 4  (item 2 cross-loads on both constructs)
  \draw[->] (t2) -- (Y2);
  \draw[->] (t2) -- (Y3);
  \draw[->] (t2) -- (Y4);
  % Item-specific latents -> items
  \draw[->] (z1) -- (Y1);
  \draw[->] (z2) -- (Y2);
  \draw[->] (z3) -- (Y3);
  \draw[->] (z4) -- (Y4);
\end{tikzpicture}
"""

# ═══════════════════════════════════════════════════════════════════════
# 9. Non-identifiability of the nomological network (Chapter 6, Validity)
#    Two multi-construct latent models over the same four items. They differ in
#    the construct space:
#      (a) correlated-factors model: theta_1 -> Y1,Y2 ; theta_2 -> Y3,Y4 ; corr
#      (b) bifactor model: a general factor g on all items, plus orthogonal
#          specific factors s_1 -> Y1,Y2 and s_2 -> Y3,Y4
#    For suitable parameters both reproduce the same covariance (a constrained
#    bifactor is the correlated two-factor model), so the data alone cannot say
#    whether a general factor exists.
# ═══════════════════════════════════════════════════════════════════════
NOMOLOGICAL_EQUIV = r"""
\begin{tikzpicture}
  % ===== Panel (a): two correlated factors =====
  \node[font=\small] at (2.15, 3.7) {(a) correlated factors};
  \node[latent]   (at1) at (0.65, 2.3) {$\theta_1$};
  \node[latent]   (at2) at (3.65, 2.3) {$\theta_2$};
  \node[observed] (aY1) at (0.0, 0.7) {$Y_1$};
  \node[observed] (aY2) at (1.3, 0.7) {$Y_2$};
  \node[observed] (aY3) at (3.0, 0.7) {$Y_3$};
  \node[observed] (aY4) at (4.3, 0.7) {$Y_4$};
  \draw[->] (at1) -- (aY1);
  \draw[->] (at1) -- (aY2);
  \draw[->] (at2) -- (aY3);
  \draw[->] (at2) -- (aY4);
  \draw[<->] (at1) to[bend left=45] (at2);

  % ===== Panel (b): bifactor (general g + orthogonal specifics s1, s2) =====
  \node[font=\small] at (8.65, 3.7) {(b) bifactor};
  \node[latent]   (bs1) at (7.15, 2.3)  {$s_1$};
  \node[latent]   (bs2) at (10.15, 2.3) {$s_2$};
  \node[observed] (bY1) at (6.5, 0.7)  {$Y_1$};
  \node[observed] (bY2) at (7.8, 0.7)  {$Y_2$};
  \node[observed] (bY3) at (9.5, 0.7)  {$Y_3$};
  \node[observed] (bY4) at (10.8, 0.7) {$Y_4$};
  \node[latent]   (g)   at (8.65, -1.1) {$g$};
  \draw[->] (bs1) -- (bY1);
  \draw[->] (bs1) -- (bY2);
  \draw[->] (bs2) -- (bY3);
  \draw[->] (bs2) -- (bY4);
  \draw[->] (g) -- (bY1);
  \draw[->] (g) -- (bY2);
  \draw[->] (g) -- (bY3);
  \draw[->] (g) -- (bY4);

  % ===== Shared observable data =====
  \node[draw=black!70, thick, rounded corners=3pt, inner sep=6pt, fill=black!10]
        (data) at (4.0, -2.7) {same $\mathrm{Cov}(Y_1, Y_2, Y_3, Y_4)$};
  \draw[->, densely dashed] (2.15, -0.1) to[bend right=12] (data);
  \draw[->, densely dashed] (6.6, -1.6) to[bend right=8] (data);
\end{tikzpicture}
"""

# ═══════════════════════════════════════════════════════════════════════
# 10. Systematic construct-irrelevant variance (Chapter 6, Validity)
#     The target construct theta and a construct-irrelevant cause N are BOTH
#     parents of the shared items: Y_j = lambda_j theta + gamma_j N + z_j.
#     N is "systematic" because it loads on several items (not just one); if N
#     were item-specific it would collapse into the unique factor z_j. Drawn with
#     theta on the left (solid loadings) and N on the right (dashed loadings).
# ═══════════════════════════════════════════════════════════════════════
CIV_GRAPH = r"""
\begin{tikzpicture}
  % Observed items (vertical column)
  \node[observed] (Y1) at (0, 3)  {$Y_1$};
  \node[observed] (Y2) at (0, 1)  {$Y_2$};
  \node[observed] (Y3) at (0, -1) {$Y_3$};
  \node[observed] (Y4) at (0, -3) {$Y_4$};
  % Target construct (left)
  \node[latent] (th) at (-3.4, 0) {$\theta$};
  \node[font=\small] at (-3.4, -1.15) {construct};
  % Construct-irrelevant common cause (right)
  \node[latent] (N) at (3.4, 0) {$N$};
  \node[font=\small, align=center] at (3.4, -1.25) {construct-\\irrelevant};
  % Construct loadings (solid)
  \draw[->] (th) -- node[above, font=\scriptsize, pos=0.5]{$\lambda$} (Y1);
  \draw[->] (th) -- (Y2);
  \draw[->] (th) -- (Y3);
  \draw[->] (th) -- (Y4);
  % Nuisance loadings (dashed, to mark the unwanted source)
  \draw[->, dashed] (N) -- node[above, font=\scriptsize, pos=0.5]{$\gamma$} (Y1);
  \draw[->, dashed] (N) -- (Y2);
  \draw[->, dashed] (N) -- (Y3);
  \draw[->, dashed] (N) -- (Y4);
\end{tikzpicture}
"""


if __name__ == "__main__":
    print("Generating plate diagrams (standalone TikZ -> PNG)...")
    diagrams = [
        ("plate_rasch.png",        RASCH),
        ("plate_2pl.png",          TWO_PL),
        ("plate_3pl.png",          THREE_PL),
        ("plate_factor.png",       FACTOR),
        ("plate_bt.png",           BRADLEY_TERRY),
        ("plate_ising.png",        ISING),
        ("plate_hierarchical.png", HIERARCHICAL),
        ("nomological_network.png", NOMOLOGICAL),
        ("nomological_equiv.png",   NOMOLOGICAL_EQUIV),
        ("civ_graph.png",           CIV_GRAPH),
    ]
    for name, body in diagrams:
        compile_tikz(name, body)
    print("Done!")
