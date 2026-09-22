<!--
The title of this pull request becomes the commit message on main, because merges are squashed.
release-please reads those messages to write the changelog and pick the next version, so the title
has to follow Conventional Commits:

    <type>[(scope)][!]: <description>

    feat(agenda): return the status of every agenda item
    fix(auth): tell clients when the access token has expired
    feat(api)!: drop POST /api/summarize-download          <- ! marks a breaking change

Types: build chore ci docs feat fix perf refactor revert style test

Only feat, fix, perf, refactor and revert appear in the changelog. A CI check validates the title;
if it fails, edit the title in the browser and it re-runs on its own.

The body below becomes the commit body on main. Do not edit CHANGELOG.md, it is generated.
-->

## What changed

## Why

## How it was verified
