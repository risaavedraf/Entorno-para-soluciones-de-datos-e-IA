# Skill Registry: entorno-mlops

**Generated:** 2026-04-18
**Project:** entorno-mlops
**Stack:** Python 3.12, FastAPI, scikit-learn, PostgreSQL, Docker

---

## User Skills (Global)

| Skill | Trigger | Source |
|-------|---------|--------|
| sdd-init | Initialize SDD context | `~/.config/opencode/skills/sdd-init` |
| sdd-explore | Explore and investigate ideas | `~/.config/opencode/skills/sdd-explore` |
| sdd-propose | Create change proposals | `~/.config/opencode/skills/sdd-propose` |
| sdd-spec | Write specifications | `~/.config/opencode/skills/sdd-spec` |
| sdd-design | Create technical design | `~/.config/opencode/skills/sdd-design` |
| sdd-tasks | Break down into tasks | `~/.config/opencode/skills/sdd-tasks` |
| sdd-apply | Implement tasks | `~/.config/opencode/skills/sdd-apply` |
| sdd-verify | Verify implementation | `~/.config/opencode/skills/sdd-verify` |
| sdd-archive | Archive completed changes | `~/.config/opencode/skills/sdd-archive` |
| sdd-onboard | SDD workflow walkthrough | `~/.config/opencode/skills/sdd-onboard` |
| go-testing | Go testing patterns | `~/.config/opencode/skills/go-testing` |
| skill-creator | Create new AI skills | `~/.config/opencode/skills/skill-creator` |
| skill-registry | Update skill registry | `~/.config/opencode/skills/skill-registry` |
| issue-creation | Create GitHub issues | `~/.config/opencode/skills/issue-creation` |
| branch-pr | Create pull requests | `~/.config/opencode/skills/branch-pr` |
| judgment-day | Adversarial review | `~/.config/opencode/skills/judgment-day` |
| engram | Engram memory capture | `~/.config/opencode/skills/engram` |
| memoria | User memory protocol | `~/.config/opencode/skills/memoria` |
| analista | Analysis and structuring | `~/.config/opencode/skills/analista` |
| novadb-mantenimiento | Memory maintenance | `~/.config/opencode/skills/novadb-mantenimiento` |

---

## Project Conventions

No project-level agent instruction files found (no `agents.md`, `CLAUDE.md`, `.cursorrules`, etc.)

---

## Stack-Specific Guidance

### Python / FastAPI / ML

**Code Style:**
- Line length: 100 characters (`ruff.toml`)
- Target Python: 3.10+ (`ruff.toml`)
- Import order: isort rules enforced by ruff
- Naming: PEP8 naming conventions (with ML exceptions for X, y, X_train, X_test)

**Testing:**
- Framework: pytest with FastAPI TestClient
- Use mocks to isolate tests from model dependencies
- Run: `python -m pytest tests/ -v`
- Coverage: `python -m pytest tests/ -v --cov=app`

**Linting:**
- Linter: ruff (replaces flake8 + black + isort)
- Check: `ruff check app/ scripts/ tests/`
- Format: `ruff format app/ scripts/ tests/`

**ML Pipeline:**
- Feature engineering in both training and inference
- One-Hot Encoding for categorical variables
- Model serialization with joblib
- Metadata tracking (R², MAE, RMSE)

---

## CI/CD Integration

**GitHub Actions Workflow:**
- Test matrix: Python 3.10, 3.11, 3.12
- Linting with ruff (check + format check)
- Docker build verification
- Sequential jobs: test → lint → build

---

## Quick Commands

```bash
# Run tests
python -m pytest tests/ -v

# Lint code
ruff check app/ scripts/ tests/
ruff format app/ scripts/ tests/

# Run API locally
python -m uvicorn app.main:app --reload

# Docker
docker compose up -d  # PostgreSQL + pgAdmin
docker build -t housing-api .

# ML Pipeline
python scripts/ingesta.py        # Load CSV → PostgreSQL
python scripts/entrenamiento.py  # Train and save model
```
