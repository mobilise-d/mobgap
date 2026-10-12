## Repository-local skills and refactoring

- When the same skill name is available from multiple locations, always use the repository-local skill under `$REPO_ROOT/.agents/skills/`. Repository-local skills take precedence over user-global skills. Do not load or apply the corresponding skill from `$HOME/.agents/skills/` when a repository-local version exists.
- Consider the repository's current publication state and backwards-compatibility requirements recorded in `.agents/refactor-policy.md` in all interactions.

## Documentation and agent artifacts

- Follow [docs/AGENTS.md](docs/AGENTS.md) when adding or changing documentation, including docstrings and examples outside `docs/`.
- Never include agent-generated plan or specification files in changes merged to `main`, anywhere in this repository. Keep temporary agent plans, task specifications, and review coordination files outside the tracked repository.

## Changelog

- Changelog entries describe the final changes relative to the last published release.
- When stacked changes revise an unreleased feature, update or replace its existing unreleased entry to describe the resulting behavior. Do not add entries narrating intermediate versions, draft API changes, or development discussions.
