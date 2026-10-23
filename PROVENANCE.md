# Source provenance

This repository continues the general CSL validation work previously hosted in
[Men-cotton/cerebras-native16-repro](https://github.com/Men-cotton/cerebras-native16-repro).
The migration starts from commit
[`6467a527e07417fb5cd2c7f742eb92cbce266951`](https://github.com/Men-cotton/cerebras-native16-repro/commit/6467a527e07417fb5cd2c7f742eb92cbce266951),
with its Git history retained. That snapshot contains 263 tracked files with
29,101,085 bytes of file content. New operator code and entry documentation are
added on top of the inherited tree.

The inherited Python scripts, CSL sources, deterministic case packages, tests and
guides retain their paths and bytes at migration. The root `.gitignore` gains
allowlist entries for new files. Changes to subsequent implementations belong to
later commits; migration itself does not rewrite completed result records or the
old repository.

## Preserved packages

| Package | Recorded origin and scope |
| --- | --- |
| `source/`, `diagnostics/` | Native16 transfer reproducer and instrumented variants; see `USAGE.md` |
| `clock_source/` and `clock_*.py` | Host-command/device-clock experiments and result interpretation; see `CLOCKS.md` and `CLOCK_CAMPAIGN.md` |
| `mesh_cases/` | Frozen MeshGraphNet cases and oracles; manifest source revision `231ff3c8aefedddfdc4d8e6d3ac5309c1fc78b08` |
| `mesh_followup_cases/` | Separate H4/S2 and H8/S2 continuation cohorts; manifest source revision `c198e401b546df65f371dae452bcb4460f1f4fad` |
| `tank_cases/` | Top-K capacity and halo fixtures, independent oracles and retained original sources; manifest source revision `473f86a01aa018ae38b3bb3e5e6dc8d6f3b2ed8c` |
| `mesh_local_evidence/` | A retained local SDK 2.10.2 capture, explicitly identified as local simulation in its README |

The manifest source revisions identify the source research repository at export
time, not the current head of this repository. Package manifests preserve SHA-256
identities for inputs, expected arrays, CSL source and independent oracles.
`tank_cases/provenance/` also retains the exporter and original vendor/reference
sources. The audit reconstructs declared compatibility transformations and compares
them with the saved official sources. Keep these paths and records intact.

## Attribution and licensing

The inherited snapshot has no repository-wide `LICENSE` or `NOTICE` file.
This migration does not assign a new blanket license to inherited material.
Vendor/reference sources retain their recorded origins and any existing notices;
their presence does not imply that the new repository owns them. In particular,
the retained stencil sources record both the historical SDK example package and
the SDK 2.10.2 official references used for compatibility checks.

The new repository name describes the broader validation purpose. It does not
change the ownership of imported sources, certify a kernel for hardware, or turn
local simulation evidence into a hardware measurement.
