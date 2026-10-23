# Hardware request examples

These manifests select existing single compile/run entrypoints. Merely adding
the files to the repository does not submit a job. A controller request must
explicitly select one manifest through the documented commit trailers.

| Manifest | Work | Retained output below the controller's request output directory |
| --- | --- | --- |
| `native16-baseline.json` | Original two-PE transfer probe, offset 3 / length 7 / mode 0 | `build/artifact.json`, `run-control/collection.json`, and the downloaded capture |
| `topk-small.json` | Frozen uneven 3x2 Top-K case with A/B/A checks | `build/build.json`, `run-aba/collection.json`, `run-aba/audit.json`, and full capture |

The controller chooses the Python interpreter and substitutes `{output}` with
an absolute, request-specific directory. The scripts run from the selected
commit's clone. Each appliance step submits one compile or run operation;
the controller must reconcile its job IDs before advancing. The run step
uses the exact artifact record produced by the preceding compile step.

Both examples use namespace `job-operator`, matching the supplied ALCF job
record. Change the manifest if the site assigns another namespace. A 1200-second
client bound includes queue and service time; the run worker has a separate
300-second host bound. These are starting bounds inherited from the existing
campaigns, not measured turnaround estimates for a new server execution.
Client timeout does not prove remote termination.

Each request ID identifies one execution. To request another execution,
copy the manifest and give it a new ID before making a new request commit.
Do not use the legacy `*_campaign.py` entrypoints as appliance steps: they
can submit multiple jobs and continue after client errors, which would bypass
the controller's per-job admission and reconciliation.

Both run entrypoints perform their existing result checks before returning
success. Archive integrity and remote job success remain separate from the
numerical result. The native16 control tests the chosen condition only; it
does not establish correctness for the known length-8 boundary. The Top-K
request establishes its small-case behavior, not hardware scaling.
