# Contributing to AI Measurement Science

Thanks for your interest in contributing! *AI Measurement Science* is a living textbook, and contributions from outside the author team are very welcome — from a one-line typo fix to a new chapter. This document explains how we work together on the book. The short version: we care more about a **coherent, comprehensive narrative** than about volume of text. The best contributions start with a conversation with the current authors about what is wrong or missing *before* anyone writes the replacement.

* [Project Overview](#project-overview)
* [Ways to Contribute](#ways-to-contribute)
* [How a Contribution Happens](#how-a-contribution-happens)
* [From Contributor to Author](#from-contributor-to-author)
* [Development Environment Setup](#development-environment-setup)
* [Submitting a Pull Request](#submitting-a-pull-request)

## Project Overview

See the [README](README.md) for what the book is and how to build it. For contributors, the key facts:

* The book is a [Quarto](https://quarto.org) project. Chapters are authored in `.qmd` (Quarto Markdown) — prose, LaTeX math, executable Python, and Manim-rendered videos in one file. It builds to both HTML ([aimslab.stanford.edu/textbook/](https://aimslab.stanford.edu/textbook/)) and PDF.
* `_quarto.yml` is the single source of truth for the book's structure (chapters, parts, appendices). The book currently has eleven chapters organized into three parts, plus a notation appendix.
* Content lives in `src/chap1.qmd … src/chap11.qmd`, with `index.qmd` as the preface. Shared assets are in `src/data/`, `src/Figures/`, and `references.bib`.

Because this is a textbook and not a software library, the unit of quality is the **reader's understanding**. A change is good if it makes the book clearer, more correct, or more complete *without* breaking the flow of the surrounding chapters.

## Ways to Contribute

Useful contributions come in many sizes:

**Corrections and clarity**
* Typos, broken cross-references (`@sec-…`, `@fig-…`, `@eq-…`), broken links
* Math errors, mislabeled figures, incorrect citations
* Rewording a confusing passage, adding an intuition or worked example

**Content**
* New examples, exercises, figures, or animations for existing chapters
* Filling a gap in an existing chapter (a missing method, derivation, or caveat)
* New sections or chapters that extend the book's scope

**Infrastructure**
* Build, rendering, and deployment fixes
* Improvements to the interactive (Pyodide) code blocks, plots, or videos
* Accessibility, navigation, and styling improvements

## How a Contribution Happens

We use a discussion-first, four-step process. The goal is to keep the book coherent and to make sure effort is spent on changes we have agreed are worth making.

1. **Propose an issue.** Anyone can [open a GitHub issue](https://github.com/aims-foundations/aims/issues) describing the change they would like to make. For anything beyond a trivial fix, say what is *wrong or missing* in the current text, and what you propose to do about it. A typo fix can be a one-line issue (or just a direct PR); a new section should explain how it fits the existing narrative.

2. **Discuss.** The authors discuss the proposal with you on the issue and decide whether it makes sense, and how it should be scoped. This is the most important step — agreeing on *what* and *why* before *how* is what keeps the book coherent. Please wait for this conversation before writing a large amount of content.

3. **Write the PR.** A contributor — this could be an author, the issue proposer, or someone else — opens a pull request implementing the agreed change. See [Submitting a Pull Request](#submitting-a-pull-request) for the mechanics.

4. **Review and merge.** Authors review the PR, request any adjustments, and merge once it is ready.

A complete rewrite that arrives without a prior discussion about what was wrong with the existing content is hard to review and hard to merge — not because the writing is unwelcome, but because we cannot tell what problem it solves or whether it preserves the surrounding narrative. Please open the conversation first.

## From Contributor to Author

Sustained, substantial contributors can become authors of the book. We keep this deliberately open: rather than committing up front to aspiring authors, we let authorship follow demonstrated engagement. As a rough guide, a contribution on the order of 20% of the book worth of substantial content is a reasonable bar for authorship — for a contributor who is not already an author and who wants the role. The threshold is about engagement. We weigh contributions that keep the narrative coherent and comprehensive: participating in the discussion (step 2), responding to review, and caring about how new material connects to the rest of the book. Generating a large volume of text without that engagement does not count toward authorship. The current authors reserve the right to make the final decision on authorship.

Becoming an author means opting into the book's ongoing life — future revisions, reviews, and maintenance — not just a credit on past work. It is fine to make substantial contributions without taking on that commitment; we are glad to acknowledge contributors either way.

If you are working toward this or want to understand how a particular contribution would be weighed, we are happy to discuss.

## Development Environment Setup

Full instructions are in the [README](README.md). The short version:

```bash
git clone https://github.com/aims-foundations/aims
cd aims

# Build environment (enough to render the book)
pip install -r requirements-build.txt

# One-time Quarto extensions
quarto add coatless/quarto-pyodide --no-prompt
quarto add leovan/quarto-pseudocode --no-prompt
```

Then preview with live reload:

```bash
quarto preview        # http://localhost:4200/
```

For a one-shot HTML build (the fastest check), use `quarto render --to html`.
Install the full `requirements.txt` only if you also need the data-processing,
embedding, or narration scripts under `scripts/` and `src/data/`. R dependencies
(`renv::restore()`) are only needed for chapters with R cells.

## Submitting a Pull Request

1. Fork the repo and create a feature branch off `main`.
2. Make your changes. Before opening the PR, verify:
   - [ ] `quarto render --to html` builds without errors
   - [ ] New or moved content has correct cross-references (`@sec-…`, `@fig-…`,
     `@eq-…`, `@tbl-…`) and citations resolve against `references.bib`
   - [ ] New figures, videos, and data artifacts are exported as resources in
     `_quarto.yml` if chapters depend on them
   - [ ] Prose matches the voice and notation of the surrounding chapters (see
     `src/notation.qmd`)
3. Open a pull request against `main` and link the issue it implements.
4. CI runs `quarto render` on the PR. An author will review; most PRs need one
   approval before merge.

If your change touches the freeze cache or executed cells, note it in the PR — freeze artifacts under `_freeze/` are committed so CI does not re-execute.