# Security Policy

Arxie is currently a single-user, self-hosted research application.

## Supported Versions

Security fixes target the latest public release branch and latest tagged release.

## Reporting A Vulnerability

Do not open public issues for suspected vulnerabilities or leaked secrets.

Report security issues by contacting the repository owner through GitHub.
Include:

- affected version or commit
- deployment mode
- reproduction steps
- impact summary
- any logs with secrets removed

## Local Data And Secrets

- Keep `.env` local and untracked.
- Do not upload private papers, drafts, code, or results to public issue threads.
- Arxie only uses explicit study sources that the user provides; it should not auto-scan arbitrary filesystem paths.
