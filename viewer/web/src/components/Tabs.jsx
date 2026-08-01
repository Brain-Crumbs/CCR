export function Tabs({ tabs, active, onChange }) {
  return (
    <div className="tabs" role="tablist">
      {tabs.map(([id, label]) => (
        <button key={id} type="button" role="tab" aria-selected={active === id}
          className={`tab${active === id ? " is-active" : ""}`} onClick={() => onChange(id)}>
          {label}
        </button>
      ))}
    </div>
  );
}
