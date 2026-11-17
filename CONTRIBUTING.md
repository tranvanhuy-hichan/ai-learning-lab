# Contributing

Thanks for your interest in AI Vision Toolkit. Bug reports, ideas and pull requests are welcome.

## Development setup

```bash
git clone https://github.com/tranvanhuy-hichan/ai-vision-toolkit.git
cd ai-vision-toolkit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

Common tasks (also available through `make`):

| Task | Command |
|---|---|
| Run the app | `python app.py` |
| Run the tests | `pytest` |
| Lint | `ruff check .` |
| Format | `ruff format .` |

CI runs lint, a format check and the tests on Python 3.10, 3.11 and 3.12. Please make sure all of them pass locally first.

## Pull requests

1. Open an issue first for anything larger than a small fix, so we can agree on the approach.
2. Create a branch from `main` and keep the change focused.
3. Add or update tests for behaviour you change.
4. Use [Conventional Commits](https://www.conventionalcommits.org/) for messages (`feat:`, `fix:`, `docs:`, `test:`, `ci:`, `chore:`).
5. If you change a model or the pre-processing, update `models/MODEL_CARD.md` with the new numbers and say how they were measured.

## Models and data

- Do not commit datasets. They have their own licenses (see the README).
- Weight files must load with `torch.load(..., weights_only=True)`; do not add pickled objects.
- Report metrics on a held-out test set, and state the seed and image size.
