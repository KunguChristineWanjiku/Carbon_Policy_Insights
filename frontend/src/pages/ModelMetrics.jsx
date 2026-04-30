import React, { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import api from "../api/client";
import BarChart from "../components/Charts/BarChart";
import TimeSeriesChart from "../components/Charts/TimeSeriesChart";
import { toFriendlyLabel, toFriendlyModelName } from "../utils/labels";

const MODEL_ORDER = ["random_forest", "xgboost", "lightgbm", "catboost", "ffnn", "lstm", "ensemble"];
const TARGET_ORDER = ["eu_ets_carbon_price_eur", "ghg_reduction_pct", "policy_effectiveness_score"];

function asNumber(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function cellClass(value) {
  if (value === null) return "bg-slate-100 text-muted";
  if (value <= 5) return "bg-accent2/20 text-textc";
  if (value <= 10) return "bg-warn/20 text-textc";
  return "bg-danger/20 text-textc";
}

function prettifyKeys(input) {
  if (Array.isArray(input)) {
    return input.map((item) => prettifyKeys(item));
  }
  if (input && typeof input === "object") {
    return Object.fromEntries(Object.entries(input).map(([k, v]) => [toFriendlyLabel(k), prettifyKeys(v)]));
  }
  return input;
}

export default function ModelMetrics() {
  const { data } = useQuery({
    queryKey: ["metrics"],
    queryFn: async () => (await api.get("/model-metrics")).data,
  });

  const rows = data?.metrics || [];
  const hasMetrics = rows.length > 0;
  const comparisonView = useMemo(() => prettifyKeys(data?.comparison || {}), [data]);

  const matrixRows = useMemo(
    () =>
      MODEL_ORDER.map((model) => {
        const row = { model };
        TARGET_ORDER.forEach((target) => {
          const hit = rows.find((r) => r.model === model && r.target === target);
          row[target] = asNumber(hit?.MAE);
        });
        return row;
      }),
    [rows]
  );

  const barData = useMemo(
    () =>
      rows.slice(0, 21).map((r) => ({
        name: `${toFriendlyModelName(r.model)} - ${toFriendlyLabel(r.target)}`.slice(0, 38),
        MAE: Number(r.MAE || 0),
        RMSE: Number(r.RMSE || 0),
      })),
    [rows]
  );

  const avgCurve = useMemo(() => {
    if (!hasMetrics) return [];
    return TARGET_ORDER.map((target, idx) => {
      const targetRows = rows.filter((r) => r.target === target);
      const avgMae =
        targetRows.length > 0 ? targetRows.reduce((acc, r) => acc + Number(r.MAE || 0), 0) / targetRows.length : 0;
      const avgRmse =
        targetRows.length > 0 ? targetRows.reduce((acc, r) => acc + Number(r.RMSE || 0), 0) / targetRows.length : 0;
      return {
        stage: idx + 1,
        target,
        targetLabel: toFriendlyLabel(target),
        avg_mae: Number(avgMae.toFixed(4)),
        avg_rmse: Number(avgRmse.toFixed(4)),
      };
    });
  }, [hasMetrics, rows]);

  return (
    <div className="space-y-4">
      {!hasMetrics ? (
        <div className="card border border-warn/40 bg-warn/10 text-sm">
          Model metrics artefacts are not available yet. Run <code>python pipeline.py</code> to populate full model matrix values.
        </div>
      ) : null}

      <div className="card overflow-auto">
        <h4 className="font-semibold mb-3">Model Error Matrix</h4>
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr>
              <th className="text-left p-2 border border-slate-200 bg-slate-50">Model</th>
              {TARGET_ORDER.map((t) => (
                <th key={t} className="text-left p-2 border border-slate-200 bg-slate-50">
                  {toFriendlyLabel(t)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrixRows.map((r) => (
              <tr key={r.model}>
                <td className="p-2 border border-slate-200 font-medium">{toFriendlyModelName(r.model)}</td>
                {TARGET_ORDER.map((t) => (
                  <td key={`${r.model}-${t}`} className={`p-2 border border-slate-200 ${cellClass(r[t])}`}>
                    {r[t] === null ? "N/A" : r[t].toFixed(4)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <BarChart title="Error by Model and Outcome" data={barData} x="name" bars={["MAE", "RMSE"]} />

      <TimeSeriesChart
        title="Average Error by Outcome"
        data={avgCurve}
        x="targetLabel"
        series={[
          { key: "avg_mae", color: "#1A56DB", name: "Average MAE" },
          { key: "avg_rmse", color: "#E02424", name: "Average RMSE" },
        ]}
      />

      <div className="card">
        <h4 className="font-semibold mb-2">Statistical Tests and Bias Summary</h4>
        <pre className="text-xs overflow-auto">{JSON.stringify(comparisonView, null, 2)}</pre>
      </div>
    </div>
  );
}
