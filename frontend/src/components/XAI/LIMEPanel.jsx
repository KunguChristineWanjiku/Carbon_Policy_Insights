import React from "react";
import { toFriendlyLabel } from "../../utils/labels";

export default function LIMEPanel({ limeWeights = [], agreement = "N/A" }) {
  return (
    <div className="card space-y-3">
      <h3 className="font-semibold">Panel B - Local Factors</h3>
      {limeWeights.map((w) => (
        <div key={w.feature} className="flex items-center gap-2">
          <div className="w-36 text-sm truncate">{toFriendlyLabel(w.feature)}</div>
          <div className="flex-1 h-2 rounded bg-slate-200">
            <div className={`h-2 rounded ${w.weight >= 0 ? "bg-accent" : "bg-danger"}`} style={{ width: `${Math.min(100, Math.abs(w.weight) * 100)}%` }} />
          </div>
        </div>
      ))}
      <div className="text-sm">
        Alignment score: <span className="font-semibold">{agreement}</span>
      </div>
    </div>
  );
}
