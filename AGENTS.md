## Agent skills

### Issue tracker

Issues and PRDs are tracked in this repository's GitHub Issues using the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Triage uses the five canonical labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, and `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

This repository uses the single-context domain-doc layout. See `docs/agents/domain.md`.

## Training checkpoints

This is a learning project. Training runs may overwrite the shared checkpoint for the
relevant stage. Do not create or retain separate `.pth` files for each model parameter,
architecture, or random seed unless the user explicitly requests it. Do not commit
training-generated `.pth` files; record experiment settings and conclusions in
`README.md` instead.
