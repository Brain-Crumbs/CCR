export function Picker({ label, ariaLabel, value, options, onChange }) {
  return (
    <label className="picker">{label}
      <select aria-label={ariaLabel} value={value ?? ""} onChange={(e) => onChange(e.target.value)}>
        {options.map((opt) => <option key={opt} value={opt}>{opt}</option>)}
      </select>
    </label>
  );
}
