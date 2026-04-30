import React from "react";
import { toFriendlyLabel } from "../../utils/labels";

const specs = [
  ["carbon_price_level", 0, 150, 1],
  ["policy_stringency_index", 0, 100, 1],
  ["renewable_target", 0, 100, 1],
  ["industrial_output_index", 50, 150, 1],
  ["gdp_growth", -5, 10, 0.1],
  ["phaseout_rate", 0, 10, 0.1],
  ["efficiency_improvement", 0, 5, 0.1],
  ["cbam", 0, 1, 1],
];

export default function PolicySliders({ values, onChange }) {
  return (
    <div className="space-y-3">
      {specs.map(([key, min, max, step]) => (
        <div key={key}>
          <div className="text-sm flex justify-between">
            <span>{toFriendlyLabel(key)}</span>
            <span>{values[key]}</span>
          </div>
          <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={values[key]}
            onChange={(e) => onChange(key, Number(e.target.value))}
            className="w-full"
          />
        </div>
      ))}
    </div>
  );
}
