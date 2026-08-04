#!/usr/bin/env bash
# Fake "python" for jobs.test.js: writes its own argv to stdout (so a test
# can assert the exact CLI invocation jobs.js built), optionally sleeps
# (FAKE_PYTHON_DELAY_MS, milliseconds) to simulate a still-running training
# job, then exits with FAKE_PYTHON_EXIT_CODE (default 0). Never touches
# torch, never trains anything -- jobs.test.js must run fast and offline.
#
# The delay only applies to a training-style invocation (baseline/search/
# clone/resume/breed) -- never to "factory cancel"/"factory reconcile",
# exactly like the real CLI: a cancellation request is a fast flag-file
# write regardless of how long the run it targets has been training.
# cancelJob's own cancel sub-call goes through this same fake python, so
# without this distinction it would block for the launched job's entire
# fake delay before ever reaching its SIGTERM fallback.
echo "argv: $*"
case " $* " in
  *" cancel "*|*" reconcile "*) ;;
  *) if [ -n "$FAKE_PYTHON_DELAY_MS" ]; then
       sleep "$(awk "BEGIN { print $FAKE_PYTHON_DELAY_MS / 1000 }")"
     fi ;;
esac
exit "${FAKE_PYTHON_EXIT_CODE:-0}"
