This directory is the canonical home for vendored upstream model repositories.

Layout:

- `vendor/openpi`: QuantyCat fork of Physical-Intelligence OpenPI.
- `vendor/rynnvla-002`: QuantyCat fork of RynnVLA-002.

Rules:

- Keep upstream or forked source changes inside `vendor/*`.
- Keep Quantycat wrappers, launchers, configs, and evaluation glue under `models/*`.
- Prefer repo-root-relative paths from `quantycat-positronic` instead of absolute home-directory paths.

During the current migration, compatibility symlinks may remain at older locations on disk, but new tooling should resolve vendored repos from this directory first.
