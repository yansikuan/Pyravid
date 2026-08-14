# Security

Do not report vulnerabilities or exposed credentials through a public issue.
Contact the repository maintainers privately through the security contact
configured on the GitHub repository.

CAM can connect to hosted APIs and local model servers. Credentials must be
provided through environment variables and must never appear in configuration
files, command output, logs, fixtures, or commits. If a credential is committed
or otherwise exposed, revoke it immediately; deleting the file in a later
commit is not sufficient.

Before publishing changes, run:

```bash
python scripts/scan_sensitive.py
```
