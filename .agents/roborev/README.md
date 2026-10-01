# Mobgap feature review panel

`feature_ready` has five required read-only reviewers: `full_stack`,
`plan_conformance`, `conventions`, `simplicity`, and `scientific_correctness`.
Their rubrics and shared guidance live here; `.roborev.toml` wires them together.
The scientific reviewer covers numerical correctness, literature/reference
implementation fidelity, scientific terminology, evaluation, and supported claims.

## Run a review

RoboRev v0.69.0 is the verified configuration baseline. Custom review types need
an authenticated supported agent with native JSON Schema output: Codex must support
`exec --output-schema`; Claude Code must support `--json-schema`.
Panels require the daemon. Its normal review adapters and these rubrics keep
members read-only. Reviewers inspect tests and recorded results; they never run
tests, builds, benchmarks, examples, or implementation workflows.

From the feature checkout, use its actual base:

```bash
roborev config validate --local
roborev review --branch --base <feature-base> --panel feature_ready --wait
```

For a finalized stack, `roborev review --since <stack-base> --panel feature_ready
--wait` is another option. `--local` runs a single review and does not fan out.
Keep the accepted plan, acceptance criteria, approved deviations, and relevant
verification evidence available in the reviewed checkout or a configured include.
Reviewers do not inherit the implementation conversation.

Installation does not start reviews or change automatic-review selectors.
`review.default_panel` and `review.hook_review_panel` are not set in the repository.
Personal selectors still apply if configured. `fix_reasoning = "high"` supplies
synthesis reasoning in v0.69.0 and also affects ordinary fix jobs.

## Personal agent choices

The tracked panel deliberately omits agent and model pins. Custom-type members
inherit custom-type/workflow settings and then generic defaults; they do not
necessarily inherit ordinary `review_agent`/`review_model`. Synthesis inherits the
fix workflow. For example, a developer can merge these top-level settings into
`~/.roborev/config.toml`, before any table headers:

```toml
default_agent = "claude-code"
default_model = "opus"
fix_agent = "claude-code"
fix_model = "opus"
```

This example changes personal defaults across repositories. Preserve existing
settings and remove or adjust conflicting personal pins when intentionally
changing a provider. No personal configuration is modified by this repo setup.

For a provider choice specific to this panel, merge the tables from
[profiles/claude.toml](profiles/claude.toml) or
[profiles/codex.toml](profiles/codex.toml) into your personal config instead.
They define distinct personal panel/member names, reuse the repo's rubric files
through uniquely named custom types, and choose both member agents and synthesis. Select one explicitly:

```bash
roborev review --branch --base <feature-base> --panel mobgap_claude --wait
# Or --panel mobgap_codex
```

Codex uses `gpt-6.1-sol`, with `gpt-6-luna` for conventions. Claude uses its `opus`
and `sonnet` aliases; adjust them for your account and installed CLI. All five
members are required and use high reasoning. These are examples to merge, not
replacement global config files, and RoboRev does not automatically load them.

In v0.69.0, repo definitions replace complete global entries with the same name.
A personal `[review.subagents.scientific_correctness]` cannot override the tracked
entry. Distinct profile names avoid that conflict. RoboRev v0.69.0 also validates global
custom types before merging repo types, so each profile includes its own unique
type definitions pointing to the shared repo-relative rubric paths. This duplicates
configuration wiring, not instructions. Select these profiles only in mobgap
checkouts that contain the rubric files. After merging a profile, run
`roborev config validate` from this checkout. The `[projects.<remote>]`
settings support model/reasoning overrides, but no agent override. The source
loads `.roborev.toml`; there is no separate `.roborev.local.toml` overlay. Panel
member and synthesis selection is resolved independently of the ordinary
single-review `--agent`/`--model` choices, so use the personal panel profile for a
reliable provider switch rather than those flags.

Source references for the verified release:

- [Panel definitions, inheritance, and merging](https://github.com/kenn-io/roborev/blob/v0.69.0/internal/config/panels.go)
- [Member and synthesis execution selection](https://github.com/kenn-io/roborev/blob/v0.69.0/internal/daemon/panel_enqueue.go)
- [Custom review types and schema support](https://github.com/kenn-io/roborev/blob/v0.69.0/docs/advanced/custom-review-types.md)
- [Personal project defaults](https://github.com/kenn-io/roborev/blob/v0.69.0/internal/config/projects.go)

## Scientific source access

Use available read-only online tools to consult original papers, supplements,
reference implementations, and relevant literature. Cite the relevant equation,
section, or code version and distinguish an error from a documented adaptation.
A faithful historical reference implementation need not adopt a newer algorithm.
Names of variables and parameters are in scope when they misstate scientific meaning.

Tool availability differs by agent. RoboRev v0.69.0's
[Claude review adapter](https://github.com/kenn-io/roborev/blob/v0.69.0/internal/agent/claude.go)
auto-allows only `Read`, `Glob`, and `Grep`; it does not auto-allow web tools.
Codex uses a read-only sandbox, but actual network/research access depends on the
installed runtime. Do not enable unsafe agents or bypass restrictions to browse.
When sources are inaccessible, supply local papers/reference excerpts as review
artifacts or configured includes and report the research limitation. Findings
must not imply that inaccessible sources were checked.

## Repository-local tpcp skills

The upstream bundle is installed in `.agents/skills/`:

- `tpcp-builder`: entry point and common guardrails.
- `tpcp-basics`: algorithms, pipelines, parameters, cloning, and action methods.
- `tpcp-datasets`: dataset indexing and access.
- `tpcp-optimization`: training, optimization, and validation.
- `tpcp-multiprocessing`: parallelism and worker state.

The five files were installed unchanged from
[mad-lab-fau/tpcp](https://github.com/mad-lab-fau/tpcp/tree/df01fe2f5505d39dd8e436cfe878d63a36d9bdd0/skills/tpcp),
commit `df01fe2f5505d39dd8e436cfe878d63a36d9bdd0`. The whole bundle preserves its
sibling skill links. Local skills take precedence over global copies. Reviewers
use them as inspection criteria only and check applicability to the supported tpcp
version. Agent skill discovery will include the new local skills on the next turn.
