import React from "react";
import { toFriendlyLabel } from "../../utils/labels";

export default function SHAPChart({ items = [] }) {
  return (
    <div className="card">
      <h4 className="font-semibold mb-3">Key Factor Impact</h4>
      <div className="space-y-2">
        {items.map((it) => (
          <div key={it.feature} className="flex items-center gap-2">
            <div className="w-40 text-sm truncate">{toFriendlyLabel(it.feature)}</div>
            <div className="h-2 bg-accent/20 rounded flex-1">
              <div className="h-2 bg-accent rounded" style={{ width: `${Math.min(100, Math.abs(it.shap_value) * 100)}%` }} />
            </div>
            <div className="text-xs">{it.shap_value?.toFixed?.(3) ?? it.shap_value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
