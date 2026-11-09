# Execution requests

Store request manifests under `requests/<family>/`. The directory groups related
experiments; the manifest ID identifies one intended execution. Discover the
available manifests from the directory tree or the repository CLI, rather than
maintaining a second job list in documentation. Files committed without request
trailers are templates and do not submit jobs.

Each schema-1 manifest contains `id`, `description`, and foreground `steps`.
Each step records `name`, `script`, `args`, `timeout_seconds`, and `kind`.
Script paths are relative to the repository root, for example
`validations/native16/appliance.py`. Appliance steps select an entrypoint
listed in the corresponding family's `validation.json`; each such step submits
one compile or run operation. Host steps perform local analysis and must not
submit remote work. Do not use campaign or retry helpers as appliance steps:
they can submit multiple jobs outside the controller's reconciliation.

The controller chooses the interpreter through uv and substitutes `{output}`
with an absolute, request-specific output directory. It freezes a separate
clone at the request commit and reconciles each step's job IDs before advancing.
A run step consumes the exact artifact record produced by its compile step.
Scripts must remain in the foreground and report their appliance job IDs.

Select a manifest in the final paragraph of its request commit message:

```text
ALCF-Request: my-case-20261023-01
ALCF-Manifest: requests/my-family/my-case-20261023-01.json
```

The trailer ID must match the manifest ID. Use a new ID for each intended
execution, including reruns; changing a template alone does not request one.
See the [repository workflow](../README.md) for branch setup and submission.

Set the namespace and time bounds for the target installation. The inherited
templates use `job-operator`, matching the supplied ALCF job record, with a
1200-second client budget and a separate 300-second run-worker bound. For host
steps, `timeout_seconds` is a wall-clock limit. For appliance steps, it excludes
intervals between two scheduler observations confirming that at least one of this step's
identified jobs is unfinished and every unfinished job is waiting in CREATED, QUEUED, QUEUEING, PENDING or SCHEDULED.
There is no separate deadline for confirmed queue waiting. Discovery, phase
transitions, running jobs, finalization and result collection consume the client
budget. These are initial budgets, not measured service times. The worker bound
starts inside the allocated job and remains unchanged. Client timeout does not
prove that the remote job terminated. See the [operator timeout policy](../docs/workflows/operator.md#queue-waiting-and-timeouts).

Preserve each adapter's numerical checks and retain its capture. Archive
integrity, remote terminal status, and numerical correctness are separate
claims. Document a request's scientific scope with its validation family;
keep result artifacts outside the source checkout.
