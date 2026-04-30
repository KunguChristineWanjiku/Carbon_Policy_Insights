import React, { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import api from "../api/client";
import useAppStore from "../store/useAppStore";
import { toFriendlyLabel, toFriendlyModelName } from "../utils/labels";

const models = ["random_forest", "xgboost", "lightgbm", "catboost", "ffnn", "lstm", "ensemble"];

const fieldSpecs = [
  ["year", "Year"],
  ["carbon_price_eur", "Carbon Price (EUR/tCO2)"],
  ["ghg_target_pct", "Emissions Reduction Target (%)"],
  ["energy_intensity", "Energy Use Intensity"],
  ["renewable_share", "Renewable Energy Share (%)"],
  ["gdp_per_capita", "GDP per Person"],
  ["policy_stringency_index", "Policy Strength Index"],
  ["fossil_fuel_share", "Fossil Fuel Share (%)"],
  ["industrial_output_index", "Industrial Activity Index"],
  ["lag_1yr_ghg", "Last Year's Emissions Index"],
  ["lag_3yr_ghg", "Three-Year Emissions Index"],
];

export default function Predictions() {
  const nav = useNavigate();
  const setPrediction = useAppStore((s) => s.setPrediction);

  const { data: countries } = useQuery({
    queryKey: ["countries"],
    queryFn: async () => (await api.get("/data/countries")).data,
  });

  const [model, setModel] = useState("ensemble");
  const [form, setForm] = useState({
    country: "",
    year: 2024,
    carbon_price_eur: 60,
    ghg_target_pct: 5,
    energy_intensity: 0.3,
    renewable_share: 30,
    gdp_per_capita: 15000,
    policy_stringency_index: 55,
    fossil_fuel_share: 60,
    industrial_output_index: 100,
    lag_1yr_ghg: 100,
    lag_3yr_ghg: 102,
  });

  const m = useMutation({
    mutationFn: async () => {
      const features = {
        country_name: form.country,
        year: form.year,
        eu_ets_carbon_price_eur: form.carbon_price_eur,
        ghg_reduction_pct: form.ghg_target_pct,
        energy_intensity: form.energy_intensity,
        renewable_share: form.renewable_share,
        gdp_per_capita: form.gdp_per_capita,
        policy_stringency_index: form.policy_stringency_index,
        fossil_fuel_share: form.fossil_fuel_share,
        industrial_output_index: form.industrial_output_index,
        lag_1yr_ghg: form.lag_1yr_ghg,
        lag_3yr_ghg: form.lag_3yr_ghg,
      };
      return (await api.post("/predict", { features, model })).data;
    },
    onSuccess: (d) => setPrediction(d),
  });

  const res = m.data;
  const band = useMemo(() => {
    const s = res?.prediction?.policy_effectiveness_score;
    if (s == null) return "N/A";
    if (s >= 70) return "GREEN";
    if (s >= 40) return "AMBER";
    return "RED";
  }, [res]);

  return (
    <div className="space-y-4">
      <div className="card grid grid-cols-3 gap-3">
        <div className="space-y-1">
          <label htmlFor="pred-country" className="text-xs font-medium text-slate-600">
            Country
          </label>
          <select
            id="pred-country"
            className="border rounded px-2 py-2 w-full"
            value={form.country}
            onChange={(e) => setForm({ ...form, country: e.target.value })}
          >
            <option value="">Select country</option>
            {(countries?.countries || []).map((c) => (
              <option key={c.country_iso3} value={c.country_name}>
                {c.country_name}
              </option>
            ))}
          </select>
        </div>

        {fieldSpecs.map(([key, label]) => (
          <div key={key} className="space-y-1">
            <label htmlFor={`pred-${key}`} className="text-xs font-medium text-slate-600">
              {label}
            </label>
            <input
              id={`pred-${key}`}
              type="number"
              className="border rounded px-2 py-2 w-full"
              value={form[key]}
              onChange={(e) => setForm({ ...form, [key]: Number(e.target.value) })}
            />
          </div>
        ))}
      </div>

      <div className="flex gap-2 items-end">
        <div className="space-y-1">
          <label htmlFor="pred-model" className="text-xs font-medium text-slate-600">
            Model
          </label>
          <select id="pred-model" className="border rounded px-2 py-2" value={model} onChange={(e) => setModel(e.target.value)}>
            {models.map((x) => (
              <option key={x} value={x}>
                {toFriendlyModelName(x)}
              </option>
            ))}
          </select>
        </div>
        <button className="px-3 py-2 bg-accent text-white rounded" onClick={() => m.mutate()}>
          Predict
        </button>
      </div>

      {res && (
        <div className="grid grid-cols-3 gap-3">
          {Object.entries(res.prediction).map(([k, v]) => (
            <div key={k} className="card">
              <div className="text-sm text-muted">{toFriendlyLabel(k)}</div>
              <div className="text-xl font-bold">{Number(v).toFixed(3)}</div>
            </div>
          ))}
          <div className="card">
            <div className="text-sm">Risk band</div>
            <div className="text-lg font-semibold">{band}</div>
          </div>
        </div>
      )}

      {res && (
        <button className="px-3 py-2 border rounded" onClick={() => nav("/explainability")}>
          Explain this prediction
        </button>
      )}
    </div>
  );
}
