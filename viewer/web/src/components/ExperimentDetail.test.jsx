import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { ExperimentDetail } from "./ExperimentDetail.jsx";

afterEach(cleanup);

const artifacts = {
  experiment: { experiment_id: "run-1" },
  lineage: { mode: "clone", parent: { run_id: "run-0" }, parent_checkpoint_sha: "abcdef123456789", sibling_group: "sib-1", configuration_parents: [], weight_donor: null },
  contracts: { architecture_hash: "arch-hash-0123456789", data_contract_hash: "data-hash-0123456789", training_contract_hash: "train-hash-0123456789" },
  data_manifest: { corpus_id: "crafter-generic-v1", sessions: { train: ["a", "b"], validation: ["c"], test: [] } },
  execution: { git_commit: "0123456789abcdef", dirty_tree: false, device: "cuda:0", precision: "fp32", torch_version: "2.4.0", cuda_version: "12.1" },
};

describe("ExperimentDetail", () => {
  it("renders lineage, contract hashes, data, and execution provenance", () => {
    render(<ExperimentDetail artifacts={artifacts} summary={{ promotion: { promoted: true, reasons: ["beats champion by margin"] } }} />);
    expect(screen.getByText("clone")).toBeInTheDocument();
    expect(screen.getByText("run-0")).toBeInTheDocument();
    expect(screen.getByText("train 2 · validation 1 · test 0")).toBeInTheDocument();
    expect(screen.getByText("clean")).toBeInTheDocument();
    expect(screen.getByText("Promoted")).toBeInTheDocument();
    expect(screen.getByText("beats champion by margin")).toBeInTheDocument();
  });

  it("shows a no-data state with no manifests", () => {
    render(<ExperimentDetail artifacts={null} summary={null} />);
    expect(screen.getByText(/no Model Factory manifests recorded/)).toBeInTheDocument();
  });
});
