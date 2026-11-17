# Security Policy

## Supported versions

Only the latest release on `main` receives fixes.

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Use GitHub's private reporting instead:
**Security tab > Report a vulnerability** on this repository.

You can expect an acknowledgement within a few days. This is a personal project maintained in spare time, so there is no formal SLA.

## Notes for users

- Model weights are loaded with `torch.load(..., weights_only=True)`, which refuses to unpickle arbitrary code. Only load weight files from sources you trust.
- The Gradio app is intended for local use. If you expose it publicly, run it behind authentication and set upload limits.
