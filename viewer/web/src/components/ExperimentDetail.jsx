import { shortHash } from "../lib/format.js";

function Field({ label, value, title }) {
  if (value === null || value === undefined || value === "") return null;
  return <div className="run-summary__stat"><span className="run-summary__label">{label}</span><strong className="run-summary__value" title={title}>{value}</strong></div>;
}

function sessionCounts(sessions) {
  if (!sessions || typeof sessions !== "object") return null;
  return Object.entries(sessions).map(([split, list]) => `${split} ${Array.isArray(list) ? list.length : "?"}`).join(" · ");
}

/**
 * A run's Model Factory manifests (epic #212 §7-9): what lineage mode
 * produced it, its three compatibility-identity hashes, the execution
 * environment that ran it, the corpus it trained on, and the promotion
 * verdict against its declared gates. Every value here comes straight from
 * the factory's own JSON contracts -- this panel names them, it doesn't
 * re-derive them.
 */
export function ExperimentDetail({ artifacts, summary }) {
  if (!artifacts) {
    return (
      <section className="diagnostic experiment" aria-labelledby="experiment-title">
        <h3 id="experiment-title">Experiment</h3>
        <p className="no-data">no Model Factory manifests recorded for this run</p>
      </section>
    );
  }

  const { lineage, contracts, execution, data_manifest: dataManifest, experiment } = artifacts;
  const verdict = summary?.promotion;

  return (
    <section className="diagnostic experiment" aria-labelledby="experiment-title">
      <h3 id="experiment-title">Experiment</h3>

      <h4>Lineage</h4>
      <div className="run-summary__grid">
        <Field label="Run" value={experiment?.experiment_id} />
        <Field label="Mode" value={lineage?.mode} />
        <Field label="Parent run" value={lineage?.parent?.run_id} />
        <Field label="Parent checkpoint" value={lineage?.parent_checkpoint_sha ? shortHash(lineage.parent_checkpoint_sha) : null} title={lineage?.parent_checkpoint_sha} />
        <Field label="Sibling group" value={lineage?.sibling_group} />
        <Field label="Configuration parents" value={lineage?.configuration_parents?.length ? lineage.configuration_parents.join(", ") : null} />
        <Field label="Weight donor" value={lineage?.weight_donor} />
        {!lineage && <p className="no-data">no lineage manifest recorded</p>}
      </div>

      <h4>Contract identity</h4>
      <div className="run-summary__grid">
        <Field label="Architecture hash" value={contracts?.architecture_hash ? shortHash(contracts.architecture_hash) : null} title={contracts?.architecture_hash} />
        <Field label="Data contract hash" value={contracts?.data_contract_hash ? shortHash(contracts.data_contract_hash) : null} title={contracts?.data_contract_hash} />
        <Field label="Training contract hash" value={contracts?.training_contract_hash ? shortHash(contracts.training_contract_hash) : null} title={contracts?.training_contract_hash} />
        {!contracts && <p className="no-data">no contracts manifest recorded</p>}
      </div>

      <h4>Data</h4>
      <div className="run-summary__grid">
        <Field label="Corpus" value={dataManifest?.corpus_id} />
        <Field label="Sessions" value={sessionCounts(dataManifest?.sessions)} />
        {!dataManifest && <p className="no-data">no data manifest recorded</p>}
      </div>

      <h4>Execution</h4>
      <div className="run-summary__grid">
        <Field label="Commit" value={execution?.git_commit ? shortHash(execution.git_commit, 10) : null} title={execution?.git_commit} />
        <Field label="Working tree" value={execution ? (execution.dirty_tree ? "dirty" : "clean") : null} />
        <Field label="Device" value={execution?.device} />
        <Field label="Precision" value={execution?.precision} />
        <Field label="PyTorch" value={execution?.torch_version} />
        <Field label="CUDA" value={execution?.cuda_version} />
        {!execution && <p className="no-data">no execution provenance recorded</p>}
      </div>

      <h4>Promotion</h4>
      {verdict ? (
        <>
          <p><span className={`run-summary__status${verdict.promoted ? " run-summary__status--good" : " run-summary__status--bad"}`}>{verdict.promoted ? "Promoted" : "Not promoted"}</span></p>
          {verdict.reasons?.length > 0 && <ul>{verdict.reasons.map((reason, i) => <li key={i}>{reason}</li>)}</ul>}
        </>
      ) : <p className="no-data">no promotion verdict recorded</p>}
    </section>
  );
}
