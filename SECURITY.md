# Security policy

SplatWork's apps talk to each other over your local network: the Mac app, SplatWork Node and SplatWork Capture
run small HTTP servers, discover each other with Bonjour, and exchange datasets and results. Reports about anything
that lets someone on the network (or a crafted file) read, write or run things they shouldn't are very welcome.

## Reporting a vulnerability

Please **don't open a public issue**. Report it privately through GitHub:
[**Report a vulnerability**](https://github.com/Benevision/SplatWork/security/advisories/new) (Security tab →
*Report a vulnerability*).

Include what you found, how to reproduce it, and which app and version are affected. You'll get an answer as soon
as possible, and credit in the release notes if you'd like it.

## Supported versions

Fixes go into the latest release of each app and the developer build. Please check that the issue still exists
there before reporting.

## Hardening tips

- Turn on access tokens for workers that are reachable by other people on your network, and keep SplatWork Node's
  ports (TCP 7878, UDP 5353) closed to the internet.
- Only open captures, datasets and splat files from sources you trust.
