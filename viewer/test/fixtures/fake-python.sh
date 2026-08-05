#!/usr/bin/env bash
# Fake "python" for jobs.test.js/clinic.test.js: writes its own argv to
# stdout (so a test can assert the exact CLI invocation jobs.js/server.js
# built), optionally sleeps (FAKE_PYTHON_DELAY_MS, milliseconds) to simulate
# a still-running training job, then exits with FAKE_PYTHON_EXIT_CODE
# (default 0). Never touches torch, never trains anything -- these tests
# must run fast and offline.
#
# "factory meta" and any "--dry-run" invocation are special-cased to print a
# single canned JSON document to stdout and nothing else, mirroring the real
# CLI's own stdout/stderr split (trace/argv noise on stderr, the payload
# alone on stdout) -- server.js's /api/factory-meta and /api/preview/* both
# do `JSON.parse(result.stdout)` directly (not the combined stdout+stderr
# job log every other invocation writes to), so stdout must be pure JSON for
# their happy path to be exercised at all.
case " $* " in
  *" meta "*)
    echo "argv: $*" >&2
    echo '{"modes": ["fresh", "clone", "resume", "fine_tune"], "objectives": ["windowed_rollout", "autoregressive"], "genome_schemas": ["generic_action_effects_v1"], "backbones": ["gru", "dilated_conv", "transformer"]}'
    exit "${FAKE_PYTHON_EXIT_CODE:-0}"
    ;;
  *" --dry-run "*)
    echo "argv: $*" >&2
    echo '{"dry_run": true, "marker": "fake-dry-run"}'
    exit "${FAKE_PYTHON_EXIT_CODE:-0}"
    ;;
esac

# The delay only applies to a training-style invocation (baseline/search/
# clone/resume/breed) -- never to "factory cancel"/"factory reconcile"/
# "factory reconcile-kill", exactly like the real CLI: a cancellation
# request or a state-file reconciliation is a fast local write regardless
# of how long the run it targets has been training. cancelJob's own
# cancel/reconcile-kill sub-calls go through this same fake python, so
# without this distinction they would block for the launched job's entire
# fake delay -- "reconcile-kill" is its own case (not covered by
# "reconcile"'s own pattern below) since it is one hyphenated argv token,
# not "reconcile" followed by a separate "kill" word.
echo "argv: $*"
case " $* " in
  *" cancel "*|*" reconcile "*|*" reconcile-kill "*) ;;
  *) if [ -n "$FAKE_PYTHON_DELAY_MS" ]; then
       sleep "$(awk "BEGIN { print $FAKE_PYTHON_DELAY_MS / 1000 }")"
     fi ;;
esac
exit "${FAKE_PYTHON_EXIT_CODE:-0}"
