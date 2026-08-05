"use strict";

const args = process.argv.slice(2);
const joined = args.join(" ");

function finish() {
  process.exit(Number(process.env.FAKE_PYTHON_EXIT_CODE || 0));
}

if (args.includes("meta")) {
  process.stderr.write(`argv: ${joined}\n`);
  process.stdout.write(JSON.stringify({
    modes: ["fresh", "clone", "resume", "fine_tune"],
    objectives: ["windowed_rollout", "autoregressive"],
    genome_schemas: ["generic_action_effects_v1"],
    architecture_genome_schemas: ["architecture_search_v1"],
    backbone_presets: {
      gru: { backbone: "gru", backbone_kwargs: {} },
      dilated_conv: { backbone: "dilated_conv", backbone_kwargs: {} },
      transformer_h2_l1: { backbone: "transformer", backbone_kwargs: { n_heads: 2, n_layers: 1 } },
      transformer_h4_l2: { backbone: "transformer", backbone_kwargs: { n_heads: 4, n_layers: 2 } },
    },
    backbones: ["gru", "dilated_conv", "transformer"],
  }));
  finish();
}

if (args.includes("--dry-run")) {
  process.stderr.write(`argv: ${joined}\n`);
  process.stdout.write(JSON.stringify({ dry_run: true, marker: "fake-dry-run" }));
  finish();
}

process.stdout.write(`argv: ${joined}\n`);
const fast = args.includes("cancel") || args.includes("reconcile") || args.includes("reconcile-kill");
const delay = fast ? 0 : Number(process.env.FAKE_PYTHON_DELAY_MS || 0);
if (delay > 0) setTimeout(finish, delay);
else finish();
