---
name: python-release-workflow
description: Use when releasing a uv+poe+GitHub Actions+Read the Docs Python project; supports patch/minor/major and enforces CI+RTD gates before GitHub release.
---

# Python Release Workflow

## Scope
- For repos like this: `uv`, `poe`, `gh`, RTD, changelog + tag releases.
- Any release type: patch/minor/major.
- Any change type: code, docs, tooling, compat.

## Inputs
- `bump`: `patch` | `minor` | `major`
- `base_ref`: usually `main`
- release notes source: latest changelog section
- validation-data repo release: if docs or release assets load versioned validation results from a separate repo, publish/tag `vX.Y.Z` there first

## Semver Quick Rule
- `patch`: fixes/docs/tooling/compat, no breaking API
- `minor`: backward-compatible feature(s)
- `major`: breaking behavior/API

## Flow
1. **Preflight**
   - `git status -sb` (must be clean or intentionally scoped)
   - `git log -5 --oneline` (message style)
2. **External dependency gate (when applicable)**
   - If the project docs/build pipeline loads validation artifacts from a separate repo by version, publish the matching validation-data release first.
   - Example: publish `mobgap_validation` tag/release `vX.Y.Z` before pushing the `mobgap` release commit.
   - If this is skipped, doc/revalidation workflows can fail even when the package repo itself is correct.
3. **Local verification (fresh)**
   - `uv run poe ci_check`
   - `uv run poe test`
   - `uv build`
   - docs gate (always for this setup): `uv run poe docs_clean`
4. **Release prep**
   - Add changelog section for next version.
   - Bump version: `uv run poe version --bump <patch|minor|major>`
   - Verify touched files (typically `pyproject.toml`, `src/<pkg>/__init__.py`, `uv.lock`, `CHANGELOG.md`).
5. **Commit strategy**
   - Commit fixes first.
   - Commit release prep separately (version + changelog).
6. **Push + CI gate**
   - `git push`
   - Wait green for required workflow(s): `gh run watch <run-id>`
   - Must be `conclusion=success` for release commit SHA.
7. **RTD gate (hard)**
   - Wait latest build for release SHA to finish with `success=true`.
   - Poll API if needed:
     - `curl -s "https://readthedocs.org/api/v3/projects/<project>/builds/?limit=5"`
8. **Create GitHub release**
   - Tag format: `vX.Y.Z`
   - `gh release create vX.Y.Z --target <base_ref> --title "vX.Y.Z" --notes "...from changelog..."`
9. **Post-release checks**
   - Watch publish workflow triggered by release.
   - Confirm publish workflow success.
   - Confirm RTD `stable`/version builds finish green (if configured).

## Hard Gates
- Never skip a required validation-data release when docs/builds depend on versioned external results.
- Never create GH release before CI green.
- Never create GH release before RTD green for release SHA.
- Never claim done without command evidence.

## Fast Command Set
```bash
gh release create vX.Y.Z --repo mobilise-d/mobgap_validation --target main --title "vX.Y.Z" --notes "Validation results for mobgap vX.Y.Z"
uv run poe ci_check && uv run poe test && uv build && uv run poe docs_clean
uv run poe version --bump patch
git push
gh run list --limit 5
gh run watch <run-id>
curl -s "https://readthedocs.org/api/v3/projects/<project>/builds/?limit=5"
gh release create vX.Y.Z --target main --title "vX.Y.Z" --notes "<changelog section>"
```
