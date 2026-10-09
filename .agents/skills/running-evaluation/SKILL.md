---
name: running-evaluation
description: Run MobGap evaluations with local data and companion result branches. Use when running or regenerating evaluations, publishing their results, or merging code PRs with companion evaluation-result PRs.
---

# Running evaluation

## Set up local paths

- Read the relevant generation and analysis scripts under `revalidation/` to identify their environment variables, dataset requirements, and output folders.
- Paths to local datasets and the companion `mobilise-d/mobgap_validation` repository belong in this checkout's ignored `.env`. Use `git worktree list` to find the main checkout. If the current worktree lacks `.env`, copy the main checkout's file, then adjust paths for this worktree. Preserve existing local settings and never print secrets or commit `.env`.
- Check that required paths exist. If paths or required configuration are missing, ask the user to supply them and create or complete `.env` before running.
- Common controls are `MOBGAP_VALIDATION_DATA_PATH`, dataset-specific variables such as `MOBGAP_TVS_DATASET_PATH` or `MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH`, `MOBGAP_CACHE_DIR_PATH`, and `MOBGAP_N_JOBS`. Inspect the script rather than assuming every evaluation uses all of them.

## Isolate results for feature branches

- When the code branch is not `main`, create a companion worktree of the evaluation-results repository on a branch with the same name. Reuse an existing matching worktree when available; preserve its unrelated changes.
- Point this code worktree's `MOBGAP_VALIDATION_DATA_PATH` at that companion worktree. Generate results there, not in the companion repository's main checkout.
- While the code PR is open, update the affected analysis scripts to load the companion result branch. Scripts using `ValidationResultLoader` typically set `__RESULT_VERSION` or `version` to the branch name. Local-only scripts use the companion worktree path. Local loading ignores `version`, so verify both local and remote targets where applicable.
- Record the code commit and relevant run configuration in the companion commit or PR description so later changes can be assessed. Keep raw datasets out of result commits.

## Scope and configure the run

- Evaluations can take a very long time. Do not start them as routine tests, lint checks, or incidental verification. Run only evaluations needed for the user's task, limiting algorithms and datasets to those affected. Ask the user to confirm when scope or the need for an expensive run is uncertain.
- If the user requests only some algorithms, temporarily comment out other algorithm configurations in the generation and affected analysis scripts. Preserve the original configuration and restore it when work on those algorithms is finished. Never commit the temporary commented-out configuration.
- Check CPU count, available RAM, nested model/BLAS threads, GPU availability when relevant, and cache/output disk space. Assess whether `n_jobs` is sensible for this machine; avoid multiplying outer workers by unrestricted inner threads.
- Prefer an explicit shared cache directory outside individual worktrees to reuse dataset loading. Check that it is writable and has enough space. Inspect `MOBGAP_CACHE_DIR_PATH` and any separate model cache settings.
- If concurrency or cache settings should change, suggest concrete values with a short reason and ask the user before the run. Do not silently change those settings.
- Run the selected generation script with the project's environment, then its analysis script. For example, `uv run python revalidation/gait_sequences/_98_gsd_result_generation_no_exc.py` generates results; the matching `_01_gsd_analysis.py` analyzes them. This example is not permission to launch that evaluation.
- Check progress early, then use infrequent checks appropriate to the run duration and the user's requested cadence. Avoid duplicate runs when a process is already active.

## Keep or publish results

- After the run, ask whether the user wants to keep the results and whether they should be pushed. Do not discard results or publish them without the corresponding decision.
- If keeping them, restore temporary algorithm comments and inspect the result diff for expected algorithms, datasets, and accidental files. Update `results_file_registry.txt` with `uv run poe update_validation_results` when the affected results use that registry; `.env` must still point at the companion worktree.
- If publishing is approved, commit and push the dedicated companion branch. If the code work has a PR, create a companion result PR and link each PR in the other's description. Register both with the thread's PR-linking tool when available.
- Keep results in the companion repository. Never copy evaluation results or numeric result summaries into tracked local documentation. Executable analysis pages should load external results. A brief high-level comparison may go in PR descriptions as evidence for an algorithm change.

## Merge code and companion results together

When the user requests merging a code PR with a companion result PR:

1. Compare the current code with the commit used to generate results. Inspect algorithm, preprocessing, data selection, reference labels, splitters, tuning, scoring, and configuration changes. Explain which changes could affect the results and ask whether the user wants regeneration. If no changes appear relevant, state that assessment.
2. Complete any requested scoped regeneration and update the companion PR and registry before proceeding.
3. Restore temporary algorithm comments. Change affected analysis/result-loading scripts to target evaluation-results `main` again, commit and push this change to the code PR. For local verification, use the appropriate companion checkout; never commit machine paths.
4. Merge the evaluation-result PR first, then the code PR, under the user's merge authorization. Keep the linked pair coordinated; do not merge code first while it relies on unavailable results.
5. Verify the documentation build on the merged code `main`, using results from companion `main` rather than a local feature-branch override. The repository task is `uv run poe docs`; generation scripts ending in `_no_exc.py` appear in the gallery but are not executed during the docs build. Confirm the build passes and address failures without launching unrelated expensive evaluations.
