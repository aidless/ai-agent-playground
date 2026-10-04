# Security Policy

## Supported Versions

Security fixes are applied to the current default branch. Archived versions are
not patched.

## Reporting a Vulnerability

**Please do not open a public issue for security problems.**

Use GitHub's private reporting channel:
**Security → Report a vulnerability** (https://github.com/ai-agent-playground/security/advisories/new),
or contact the maintainer directly.

Please include:

- what the issue is and which component is affected
- steps to reproduce, or a proof-of-concept
- the impact you believe it has

## Response

A human-readable acknowledgement within a few business days, best effort.
Triage and a fix timeline depend on severity; no SLA is promised. Confidentiality
is maintained for reporters throughout.

## Scope

This repository is an application that runs locally or on a hosted demo. Reports about dependencies with a published CVE
are welcome and will be tracked; issues in upstream services are out of scope.

If a deployed demo instance is affected, say which host so the maintainer can rotate
credentials that were baked into that deployment.
