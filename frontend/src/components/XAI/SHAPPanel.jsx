import React, { useState } from "react";
import SHAPChart from "../Charts/SHAPChart";

export default function SHAPPanel({ shapTop5 = [] }) {
  const [mode, setMode] = useState("Overall");

  return (
    <div className="card space-y-3">
      <div className="flex justify-between items-center">
        <h3 className="font-semibold">Panel A - Key Drivers</h3>
        <select value={mode} onChange={(e) => setMode(e.target.value)} className="border rounded px-2 py-1">
          <option>Overall</option>
          <option>This prediction</option>
        </select>
      </div>
      <SHAPChart items={shapTop5} />
      <div className="text-sm text-muted">View: {mode} | Factor impact analysis</div>
    </div>
  );
}
