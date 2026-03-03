# Documentation Style Guide

This guide defines the writing standards for all MacroEconometricModels.jl documentation pages. Every page must follow these rules. The target quality is Stata/EViews manual level — professional econometrics software documentation that reads like a textbook.

---

## Page Anatomy

Every documentation page follows this skeleton:

```
# Page Title

[1-3 sentence introduction stating what this page covers and why it matters]

[Bulleted feature list if the page covers multiple capabilities]

## Quick Start

[3-6 progressive recipes: simple → intermediate → advanced]

---

## Section Title

[Section content: intro → math → theory note → code → interpretation → return values]

---

## Next Section

...

---

## Complete Example

[Full end-to-end workflow combining everything on this page]

---

## Common Pitfalls

[Numbered troubleshooting items]

---

## References

[Full bibliography with DOIs]
```

**Rules:**
- H1 for page title only (one per page)
- H2 for major sections (separated by `---` horizontal rules)
- H3 for subsections within an H2
- **Never use H4 or deeper** — flatten the hierarchy instead
- Quick Start appears immediately after the introduction, before the first `---`
- Complete Example appears near the end, before Common Pitfalls and References
- Every page ends with a References section listing all cited works

---

## Writing Voice

**Register:** Professional econometrics textbook. The reader is a PhD student or applied researcher who knows economics but may not know this specific Julia package.

**Tense:** Present tense, active voice.
- YES: "The solver computes the QZ decomposition"
- NO: "The solver will compute the QZ decomposition"
- NO: "The QZ decomposition is computed by the solver"

**Precision:** Every claim is backed by an equation, a citation, or a code example. No hedging.
- YES: "Pruning prevents explosive sample paths"
- NO: "Pruning attempts to prevent explosive sample paths"
- NO: "Pruning can help prevent explosive sample paths"

**Brevity:** Say it once, say it precisely, move on.
- YES: "The Blanchard-Kahn condition requires n_unstable = n_forward."
- NO: "It is important to note that the Blanchard-Kahn condition, which is a key requirement for the existence and uniqueness of a rational expectations solution, requires that the number of unstable eigenvalues equals the number of forward-looking variables."

**Econometric terminology:** Use standard terms from Lutkepohl (2005), Hamilton (1994), Woodford (2003). Define terms on first use with **bold**.

---

## Section Pattern

Every H2 section follows this internal structure:

1. **Introduction** (2-4 sentences): What this section covers and why it matters economically
2. **Model specification** (display math): The key equation(s) defining the method
3. **"where" bullet list**: Define every symbol in the equation
4. **Theory note** (optional `!!! note "Technical Note"`): Implementation details, algorithm steps, or mathematical subtleties
5. **Code example**: Complete, runnable Julia code showing the method in action
6. **Output display**: Use `report()`, `print_table()`, or show the result object — **never raw `println` for results**
7. **Interpretation paragraph** (3-4 sentences): What the output means economically. Use concrete numbers from the example.
8. **Keyword table** (if applicable): Keyword | Type | Default | Description
9. **Return value table**: Field | Type | Description

Not every section needs all 9 elements. Scale to complexity. A simple utility function may need only steps 1, 5, 6, 8, 9. A major estimation method needs all of them.

---

## Quick Start Recipes

Each recipe is a self-contained code block showing one capability:

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# [1-line comment explaining what this recipe demonstrates]
result = function_call(args; key_kwargs...)
report(result)
```

**Rules:**
- Always start with `using MacroEconometricModels`
- Always set `Random.seed!(42)` when randomness is involved
- Inline comments explain the "what", not the "how"
- Use `report(result)` or `plot_result(result)` to display results — **not `println`**
- Each recipe is 5-15 lines
- Progressive complexity: Recipe 1 is the simplest possible usage, Recipe 6 is an advanced configuration
- Recipe titles use **bold**: `**Recipe 1: Solve and plot IRFs**`

---

## Code Examples

**Completeness:** Every code block must be runnable as-is. No fragments, no `...` ellipsis, no "add your data here" placeholders.

**Output display:** Use the package's built-in display infrastructure:
- `report(result)` — for estimation results, model summaries
- `print_table(stdout, result)` — for tabular output
- `plot_result(result)` — for visualization
- Direct field access (`result.theta`, `result.J_stat`) only when showing specific values for interpretation
- **Avoid `println` for displaying results.** Use `println` only for simple scalar diagnostics like convergence flags.

**Comments:** Inline comments explain intent, not mechanics:
- YES: `# Large negative demand shock pushes economy to ZLB`
- NO: `# Set shocks[1,1] to -3.0`

**Data examples:** Use `load_example(:fred_md)` or synthetic data from model simulation. Never reference external files.

**Progressive complexity:**
1. Minimal example with defaults
2. Example with key keyword arguments
3. Advanced example combining multiple features

---

## Mathematics

**Display equations:** Use fenced math blocks:

````
```math
y_t = G_1 \, y_{t-1} + C + \text{impact} \cdot \varepsilon_t
```
````

**Inline math:** Use double backtick notation: ``` ``\theta`` ```, ``` ``n \times n`` ```

**"where" bullet lists:** Every display equation is followed immediately by a bullet list defining each symbol:

```
where:
- ``y_t`` is the ``n \times 1`` vector of endogenous variables
- ``G_1`` is the ``n \times n`` state transition matrix
- ``\varepsilon_t \sim N(0, I)`` are structural shocks
```

**Consistent notation dictionary:**

| Symbol | Meaning |
|--------|---------|
| ``y_t`` | Endogenous variables at time t |
| ``x_t`` | State variables at time t |
| ``\varepsilon_t`` | Structural shocks |
| ``\theta`` | Deep structural parameters |
| ``\Sigma`` | Covariance matrix |
| ``T`` | Sample size / number of periods |
| ``n`` | Number of endogenous variables |
| ``n_x`` | Number of state variables |
| ``n_y`` | Number of control variables |
| ``H`` | IRF/forecast horizon |
| ``p`` | Lag order |
| ``\phi`` | Tempering parameter (SMC) |
| ``\sigma`` | Perturbation scaling parameter |

**Operators:** Use `\cdot` for matrix-vector products, `\otimes` for Kronecker products, `\text{tr}()` for trace.

---

## Tables

**Keyword argument tables:**

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `degree` | `Int` | `5` | Chebyshev polynomial degree |
| `tol` | `Real` | ``10^{-8}`` | Newton convergence tolerance |

- Type column uses backtick code formatting
- Default column uses backtick code formatting or math notation for powers of 10
- Description is a sentence fragment (no period), starts with capital letter

**Return value tables:**

| Field | Type | Description |
|-------|------|-------------|
| `G1` | `Matrix{T}` | ``n \times n`` state transition matrix |
| `impact` | `Matrix{T}` | ``n \times n_{shocks}`` impact matrix |

- Same formatting rules as keyword tables

**Comparison tables** (for choosing between methods):

| Feature needed | Recommended | Why |
|----------------|-------------|-----|
| Standard IRFs | `:gensys` | Robust, handles singularity |
| Risk premia | `:perturbation` (order=2) | Captures precautionary effects |

- "Why" column is 3-8 words, not a full sentence

---

## Admonitions

**Technical notes** explain implementation details, algorithm subtleties, or mathematical background that an advanced user would want:

```
!!! note "Technical Note"
    The matrices are computed via central differences with step size
    ``h = \max(10^{-7}, 10^{-7} |y_j|)``. No analytical derivatives are required.
```

**Warnings** alert users to common mistakes, gotchas, or critical requirements:

```
!!! warning "Common cause of indeterminacy"
    The Taylor principle requires ``\phi_\pi > 1``. With ``\phi_\pi < 1``,
    the New Keynesian model is typically indeterminate.
```

**Rules:**
- Admonitions appear **before** the code they relate to (not after)
- Title is descriptive: "Technical Note", "Common cause of indeterminacy", "Treatment Timing Encoding"
- Body is 2-5 lines
- Never more than 2 admonitions per H2 section

---

## References

**Inline citations:** Author (Year) format in running text:
- "The Gensys algorithm (Sims 2002) solves the linearized system..."
- "Following Schmitt-Grohe & Uribe (2004), the second-order decision rule..."

**Reference section:** Full bibliographic entries at the bottom of each page:

```
## References

- Author, A. B., & Author, C. D. (Year). Title of Paper.
  *Journal Name*, Volume(Issue), Pages. [DOI](https://doi.org/...)
```

**Rules:**
- Every method, algorithm, or technique gets a citation on first mention
- Full reference list at the bottom of every page (not just a master list)
- DOI links for all journal articles
- Book references include publisher and ISBN

---

## Formatting

- **Bold** for key terms on first use and for emphasis in lists
- *Italics* for book titles only (never for emphasis)
- `backticks` for code: function names, variable names, file names, keyword arguments
- Double backticks for inline math: ``` ``\theta`` ```
- `---` horizontal rules separate H2 sections
- Cross-references: `[Section Name](@ref target)` with page anchors `(@id target)` on H1 titles
- Visualization: `plot_result(obj)` followed by `@raw html` iframe embedding (not wrapped in triple backticks)

---

## Anti-Patterns

**Never do these:**

1. **`println` for results** — Use `report()`, `print_table()`, or field access with interpretation
2. **Code fragments** — Every code block must be complete and runnable
3. **Undefined symbols** — Every variable in a math block must be defined in the "where" list
4. **Orphan equations** — Every display equation needs context (intro sentence + "where" list + interpretation)
5. **H4 or deeper** — Flatten hierarchy; use bold text for sub-subsections if needed
6. **Hedging language** — "attempts to", "can help", "may be useful" — state what it does
7. **Passive voice for actions** — "The model is solved by" → "The solver computes"
8. **Missing references** — Every algorithm needs Author (Year) on first mention
9. **Raw output without interpretation** — After every code output, explain what the numbers mean
10. **Duplicated content** — Cross-reference other pages instead of repeating material
