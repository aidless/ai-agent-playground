# Failure modes

How a green check in this repository can be wrong, and what was changed so the common cases fail closed. This is not a claim that the system is safe.

<!-- bandit-baseline-results: 26 -->

## Safety gate

The `security` job runs `uvx bandit ... -ll -b .bandit-baseline.json` and `uvx pip-audit`.

- **Known findings do not fail the job.** The baseline file holds 26 results (4 of them HIGH: `B102` exec, `B602` shell, and others listed in the file). A green Bandit step means "nothing new relative to that file", not "no findings". Do not regenerate the baseline from a dirty tree. Shrink it only when a finding is actually fixed.
- **A swallowed command used to be green.** The step was once `uv run bandit ... || echo "bandit done"`. Bandit was not installed in the lock, so the scan never started and the `echo` made the step pass. `.github/workflows/mutation-control.yml` injects a HIGH `B602` and requires a non-zero exit. It also re-runs the old `|| echo` form and requires the verdict rule to reject that form. The control covers Bandit only. It does not cover `pip-audit`, the package build, or `scripts/pentest.py`.
- **`uv build` is not on the push path.** The build job is `workflow_dispatch` because `uv build` fails in this tree. A green CI run does not mean the package builds.
- **`pip-audit` is not mutation-tested.** It fails the job on a non-zero exit today. A future `|| true` would not be caught by `mutation-control.yml`. `tests/test_claim_drift.py` rejects `|| true`, `|| echo`, and `continue-on-error` in `ci.yml`.
- **`scripts/pentest.py` is not a pentest and not a CI job.** It is 14 in-process checks (injection helper, path denial, token length, and similar). A previous shield said "14/14 pentest" with no run attached. Run the script if you want its exit code. Do not cite a shield.
- **`scripts/b3_security_bench.py` used to mark attacks blocked because the expected label contained the letters "block".** The label is not a vote anymore: `defense_held` is true only when the prompt-injection helper or the path check fires. The script exits 0 only when every case tripped a detector. It does not call a model. An earlier draft constructed an OpenAI client and never used it. A 10/10 figure from the old scorer is not evidence.

## Eval harness

`eval_harness/` is not in the CI pytest selection. A green test job says nothing about these scorers.

Fixed in this tree, with tests in `tests/test_eval_honesty.py`:

- **No scores used to total 1.0.** An empty rubric passed. The total is now 0.0 when nothing was scored.
- **Empty keyword lists used to score 1.0.** They are omitted. `score_contains([])` and `score_forbidden([])` return 0.0 if called directly.
- **The LLM judge used to return 0.5 on an exception or an unparseable reply, and 1.0 when the prompt was empty.** Those paths return 0.0. A 0.5 blended with a keyword score of 1.0 cleared the 0.6 threshold while the judge was down.
- **A runner exception used to still pass** if the error string happened to score above the threshold. `is_passing` returns false when `error` is set.
- **A missing `tool_success_rate` used to default to 1.0.** It is now 0.0.
- **`scripts/eval_gate.py --ci` used to exit 0 when no baseline existed**, after writing one. That is not a comparison. `--ci` now exits 2. Without `--ci`, the same situation still exits 0 so a local first run can save the file. A wrapper that treats that 0 as "gate passed" is still wrong.
- **`scripts/e2e_test.py::test_streaming_agent` used to `return` when the key was missing or the call failed.** Pytest recorded a pass. The CI placeholder key now skips. A live key that errors fails the test.

Still true:

- **Length alone can pass a case.** `hello_003` (refuse a harmful request) has a minimum length and a judge prompt, and no forbidden keywords. `--quick` drops the judge, so a long answer scores on length only. Keyword and length checks are not a safety evaluation.
- **`score_round_efficiency` returns 1.0 when the meta has no tool calls.** That is "nothing to penalize", and it can still dilute a bad score upward if other process scores are mixed in.
- **The pass threshold is 0.6 on an unweighted average.** One strong keyword score can carry a weak judge.

## Counts

CI runs `python -m pytest scripts/e2e_test.py tests`. The README comment `ci-collect-only` must match `pytest --collect-only` for that selection. Passed/skipped is not frozen in the README: it depends on locust and on whether the API key is a placeholder.

A repo-wide `pytest` also imports modules named `test_*.py` and `*_test.py` outside that selection (`test_agent.py`, `scripts/system_e2e_test.py`). `scripts/connectivity_check.py` used to be named `test_connectivity.py` and performed network calls at import during collection. The CI command does not collect it.

## Stress and repair scripts

`scripts/stress_test.py` issues 500 calls, then 500 more, against `127.0.0.1:8000`. It does not assert, and it does not write a report. A previous shield said 1000/1000.

`scripts/code_bench.py` calls a live model on 10 tasks. No report is checked in. A previous shield said 90%.
