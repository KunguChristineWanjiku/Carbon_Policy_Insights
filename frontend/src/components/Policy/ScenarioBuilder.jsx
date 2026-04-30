import React, { useMemo, useState } from "react";
import toast from "react-hot-toast";
import api from "../../api/client";
import PolicySliders from "./PolicySliders";
import { toFriendlyLabel } from "../../utils/labels";

const init = {
  carbon_price_level: 40,
  policy_stringency_index: 55,
  renewable_target: 35,
  industrial_output_index: 100,
  gdp_growth: 2,
  phaseout_rate: 3,
  efficiency_improvement: 1.5,
  cbam: 0,
};

export default function ScenarioBuilder({ onSaved, latestResult }) {
  const [vals, setVals] = useState(init);
  const [name, setName] = useState("Scenario A");

  const save = async () => {
    const { data } = await api.post("/simulate", { policy_params: vals, scenario_name: name });
    toast.success("Scenario saved");
    onSaved?.(data.scenario);
  };

  const latestCards = useMemo(() => {
    if (!latestResult?.prediction) return [];
    return Object.entries(latestResult.prediction).map(([key, value]) => ({
      key,
      label: toFriendlyLabel(key),
      value: Number(value).toFixed(2),
    }));
  }, [latestResult]);

  return (
    <div className="card space-y-4">
      <h3 className="font-semibold">Scenario Builder</h3>
      <input className="border rounded px-3 py-2 w-full" value={name} onChange={(e) => setName(e.target.value)} placeholder="Scenario name" />
      <PolicySliders values={vals} onChange={(k, v) => setVals((p) => ({ ...p, [k]: v }))} />
      <button className="px-3 py-2 bg-accent text-white rounded" onClick={save}>
        Save Scenario
      </button>

      {latestCards.length > 0 ? (
        <div className="bg-slate-50 p-3 rounded border border-slate-200">
          <div className="text-sm font-semibold mb-2">Latest scenario estimate</div>
          <div className="grid grid-cols-1 gap-2 text-sm">
            {latestCards.map((item) => (
              <div key={item.key} className="flex justify-between gap-3">
                <span className="text-muted">{item.label}</span>
                <span className="font-medium">{item.value}</span>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}
