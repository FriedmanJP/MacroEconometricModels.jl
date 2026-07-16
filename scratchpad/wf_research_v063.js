export const meta = {
  name: 'v063-research',
  description: 'Verify + spec all 39 Stage 5-6 issues (#151-189) against current code; build conflict graph + batch plan',
  phases: [
    { title: 'Research-Core',  detail: '#151-162 core/HAC/numerics — verify vs current src + produce specs' },
    { title: 'Research-Micro', detail: '#163-189 DiD/micro/panel/teststat — verify vs current src + produce specs' },
    { title: 'Synthesize',     detail: 'conflict graph + ascending-order batch plan' },
  ],
}

const REPORT = 'docs/plans/2026-07-02-reliability-overhaul-report.md'

// tier: 'high' = subtle econometrics needing deep reasoning; 'med' = routine; 'batch' = list of sub-items
const ISSUES = [
  // Stage 5 — core/HAC/numerics (#151-162)
  { num: 151, tier: 'high',  grp: 'Research-Core' },
  { num: 152, tier: 'med',   grp: 'Research-Core' },
  { num: 153, tier: 'med',   grp: 'Research-Core' },
  { num: 154, tier: 'med',   grp: 'Research-Core' },
  { num: 155, tier: 'med',   grp: 'Research-Core' },
  { num: 156, tier: 'med',   grp: 'Research-Core' },
  { num: 157, tier: 'med',   grp: 'Research-Core' },
  { num: 158, tier: 'med',   grp: 'Research-Core' },
  { num: 159, tier: 'high',  grp: 'Research-Core' },
  { num: 160, tier: 'med',   grp: 'Research-Core' },
  { num: 161, tier: 'batch', grp: 'Research-Core' },
  { num: 162, tier: 'batch', grp: 'Research-Core' },
  // Stage 6 — DiD/micro/panel/teststat (#163-189)
  { num: 163, tier: 'high',  grp: 'Research-Micro' },
  { num: 164, tier: 'high',  grp: 'Research-Micro' },
  { num: 165, tier: 'high',  grp: 'Research-Micro' },
  { num: 166, tier: 'high',  grp: 'Research-Micro' },
  { num: 167, tier: 'high',  grp: 'Research-Micro' },
  { num: 168, tier: 'med',   grp: 'Research-Micro' },
  { num: 169, tier: 'med',   grp: 'Research-Micro' },
  { num: 170, tier: 'high',  grp: 'Research-Micro' },
  { num: 171, tier: 'high',  grp: 'Research-Micro' },
  { num: 172, tier: 'high',  grp: 'Research-Micro' },
  { num: 173, tier: 'med',   grp: 'Research-Micro' },
  { num: 174, tier: 'med',   grp: 'Research-Micro' },
  { num: 175, tier: 'med',   grp: 'Research-Micro' },
  { num: 176, tier: 'med',   grp: 'Research-Micro' },
  { num: 177, tier: 'high',  grp: 'Research-Micro' },
  { num: 178, tier: 'med',   grp: 'Research-Micro' },
  { num: 179, tier: 'med',   grp: 'Research-Micro' },
  { num: 180, tier: 'med',   grp: 'Research-Micro' },
  { num: 181, tier: 'med',   grp: 'Research-Micro' },
  { num: 182, tier: 'med',   grp: 'Research-Micro' },
  { num: 183, tier: 'med',   grp: 'Research-Micro' },
  { num: 184, tier: 'med',   grp: 'Research-Micro' },
  { num: 185, tier: 'med',   grp: 'Research-Micro' },
  { num: 186, tier: 'high',  grp: 'Research-Micro' },
  { num: 187, tier: 'med',   grp: 'Research-Micro' },
  { num: 188, tier: 'batch', grp: 'Research-Micro' },
  { num: 189, tier: 'batch', grp: 'Research-Micro' },
]

const SPEC_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['issue','task','title','severity','status','root_cause','files','fix_plan','test_plan','references','depends_on','touches_files','grouping','effort','risk','notes'],
  properties: {
    issue: { type: 'integer' },
    task: { type: 'string' },
    title: { type: 'string' },
    severity: { type: 'string' },
    status: { type: 'string', enum: ['confirmed','already-fixed','false-positive','partial'],
              description: 'confirmed = bug present in current code; already-fixed = a prior stage fixed it; false-positive = audit was wrong; partial = partly fixed / partly valid' },
    root_cause: { type: 'string', description: 'What is actually wrong in CURRENT code, with concrete file:line citations you verified by reading the source' },
    files: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['path','what_changes'],
      properties: { path: {type:'string'}, symbol: {type:'string'}, line_range: {type:'string'}, what_changes: {type:'string'} } } },
    fix_plan: { type: 'string', description: 'Precise, ordered implementation steps a coder can follow without re-deriving; include exact formulas/constants' },
    test_plan: { type: 'object', additionalProperties: false, required: ['file','new_testsets','oracle'],
      properties: { file: {type:'string', description:'target test file path, e.g. test/core/test_covariance.jl'},
                    new_testsets: {type:'string', description:'the @testset(s) to add and what they assert'},
                    oracle: {type:'string', description:'concrete expected numbers / reference values / analytic checks the test pins (the hard part — be specific)'} } },
    references: { type: 'string', description: 'papers, formulas, equation numbers' },
    depends_on: { type: 'array', items: { type: 'integer' }, description: 'other issue numbers this must land after (logical dependency)' },
    touches_files: { type: 'array', items: { type: 'string' }, description: 'EVERY src/ file this fix edits — used to build the conflict graph' },
    grouping: { type: 'string', description: 'suggested commit grouping: "solo" or "group with #NNN,#NNN because ..."' },
    effort: { type: 'string', enum: ['S','M','L'] },
    risk: { type: 'string', enum: ['low','med','high'] },
    notes: { type: 'string', description: 'gotchas, false-positive risk, ground-truth caveats, anything the implementer must know' },
  },
}

function researchPrompt(num, tier) {
  const task = 'T0' + (num - 99)
  const batchNote = tier === 'batch'
    ? '\n\nThis is a BATCH issue covering MULTIPLE sub-items (each with its own anchor code in the report). Enumerate EVERY sub-item separately inside fix_plan and test_plan (one paragraph per sub-item, prefixed by its anchor code). Do not collapse them.'
    : ''
  return `You are a meticulous econometrics + numerical-computing reviewer working on the Julia package MacroEconometricModels.jl. You are producing an IMPLEMENTATION SPEC for GitHub issue #${num} (task ${task}), part of release v0.6.3 (Stage 5-6 reliability remediation).

Working directory is the repo root. Do this in order:

1. Read the issue body precisely:  \`gh issue view ${num} --json title,body,labels\`  (the body is self-contained: anchors, formulas, acceptance criteria).
2. The issue derives from an audit finding anchored by a code like C-NN or M-NN or S-NN. Find that anchor in the local report ${REPORT} and read the surrounding section for full context (grep for the anchor code, then read ~40 lines around it). This report is LOCAL-ONLY ground truth.
3. CRITICAL — verify against CURRENT code, do not trust the audit's line numbers (the audit is from 2026-07-02; stages 1-4 have since edited many files). Open the actual src/ file(s), locate the exact function, and CONFIRM the described defect still exists verbatim. If a prior stage already fixed it, or the audit misread the code (audit findings have a known false-positive rate — classify diffs against econometric THEORY, not against any single reference implementation), say so and set status accordingly.
4. Produce the spec. Requirements:
   - root_cause: cite the CURRENT file:line you actually read (Grep/Read), quote the offending expression.
   - fix_plan: exact, ordered, implementable steps. Include the precise formula/constant/algorithm (e.g. Andrews 1991 plug-in constants per kernel; MacKinnon response-surface coefficients; Rambachan-Roth 2023 Δ^{SD}(M) / FLCI construction; influence-function form). A competent coder must be able to implement WITHOUT re-deriving. If exact reference constants are needed (critical-value tables, response-surface coefficients), state where they come from and the actual values if you can determine them.
   - test_plan.oracle: the concrete numbers/relations the failing test will pin. This is the hardest and most valuable part — an analytic identity, a published reference value, an equivalence (e.g. "robust==OPG when scores from observed info", "Joseph form == standard when gain exact"), or a monotonicity/sign property. Be specific and correct.
   - touches_files: list EVERY src/ file the fix will edit (used to compute the cross-issue conflict graph). Be exhaustive and exact (real paths).
   - depends_on / grouping: note logical ordering and whether it should share a commit with an adjacent issue (grouped commits are allowed when issues touch the same code and "land coherently").
   - Match the codebase's conventions (from CLAUDE.md): qualify Optim./NLopt./ForwardDiff.; robust_inv/safe_cholesky; never name a var \`eps\`; T<:AbstractFloat; public API accepts AbstractVector/Matrix.${batchNote}

Return ONLY the structured spec object. Ground every claim in code you actually read — no speculation.`
}

// ---- Phase 1+2: research all 39 (two progress groups, run concurrently) ----
log(`Researching ${ISSUES.length} issues (#151-189) against current code...`)
const specs = await parallel(ISSUES.map((it) => () =>
  agent(researchPrompt(it.num, it.tier), {
    label: `#${it.num} T0${it.num - 99}`,
    phase: it.grp,
    schema: SPEC_SCHEMA,
    effort: it.tier === 'high' ? 'high' : 'medium',
  })
))

const ok = specs.filter(Boolean)
log(`Got ${ok.length}/${ISSUES.length} specs. Building conflict graph...`)

// ---- Deterministic conflict graph (ground truth, no agent) ----
const fileMap = {}   // src file -> [issue numbers]
for (const s of ok) {
  for (const f of (s.touches_files || [])) {
    if (!fileMap[f]) fileMap[f] = []
    if (!fileMap[f].includes(s.issue)) fileMap[f].push(s.issue)
  }
}
const conflicts = Object.entries(fileMap)
  .filter(([, issues]) => issues.length > 1)
  .map(([file, issues]) => ({ file, issues: issues.sort((a, b) => a - b) }))
  .sort((a, b) => b.issues.length - a.issues.length)

const statusRoll = {}
for (const s of ok) statusRoll[s.status] = (statusRoll[s.status] || 0) + 1

// ---- Phase 3: synthesis agent — human-readable ascending-order batch plan ----
phase('Synthesize')
const compact = ok.map((s) => ({
  issue: s.issue, task: s.task, title: s.title, severity: s.severity, status: s.status,
  effort: s.effort, risk: s.risk, touches_files: s.touches_files, depends_on: s.depends_on,
  grouping: s.grouping, test_file: s.test_plan && s.test_plan.file,
}))
const planMd = await agent(
  `You are the tech lead sequencing implementation of release v0.6.3 (Stage 5-6, issues #151-189) of MacroEconometricModels.jl.

Rules of the release workflow (must respect):
- Solve in ASCENDING issue order TDD-style (failing test -> fix -> run ONLY the relevant test file(s) with MACRO_MULTIPROCESS_TESTS=1). One commit per issue, EXCEPT adjacent issues that touch the same code may be grouped into one coherent commit.
- Files edited by multiple issues MUST be done sequentially (no parallel edits to a shared file).

Here is the deterministic file->issues conflict graph (files touched by >1 issue):
${JSON.stringify(conflicts, null, 1)}

Here are the 39 issue specs (compact):
${JSON.stringify(compact, null, 1)}

Status roll-up (vs current code): ${JSON.stringify(statusRoll)}.

Produce a MARKDOWN implementation plan with these sections:
1. **Executive summary** — total issues, any that are already-fixed/false-positive (list them explicitly with the issue #, so they can be closed WITHOUT a code change), overall risk.
2. **Conflict clusters** — group issues by the source files they share; within each cluster give the mandatory sequential order and note which to group into a single commit.
3. **Ascending-order execution list** — a numbered checklist of implementation UNITS (a unit = one issue, or a small grouped set), each line: \`[ ] #NNN[,#NNN] — <one-line what> — files: <...> — test: <test file> — effort S/M/L — risk\`. Preserve global ascending order across units.
4. **High-risk deep-dives** — for each risk:high or severity:critical/high econometrics issue, 2-3 sentences on the crux and the oracle the test must pin.
5. **Cross-cutting notes** — shared helpers to introduce once (e.g. a common HC-meat / kernel-weight helper), and any dependency edges.

Be concrete and terse. This plan is the durable execution guide for the whole stage.`,
  { label: 'synthesize-plan', phase: 'Synthesize', effort: 'high' }
)

return { specs: ok, conflicts, statusRoll, planMd, n_requested: ISSUES.length, n_specs: ok.length }
