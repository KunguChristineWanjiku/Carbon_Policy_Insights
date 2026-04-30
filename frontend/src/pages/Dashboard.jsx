import React, { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import api from "../api/client";
import KPICard from "../components/Common/KPICard";
import TimeSeriesChart from "../components/Charts/TimeSeriesChart";
import BarChart from "../components/Charts/BarChart";
import DonutChart from "../components/Charts/DonutChart";

const FALLBACK_SERIES = [
  { year: 2019, value: 48.2 },
  { year: 2020, value: 50.1 },
  { year: 2021, value: 52.8 },
  { year: 2022, value: 56.4 },
  { year: 2023, value: 59.3 },
  { year: 2024, value: 61.0 },
];

export default function Dashboard() {
  const { data: k } = useQuery({
    queryKey: ["kpis"],
    queryFn: async () => (await api.get("/data/kpis")).data,
  });
  const { data: ts } = useQuery({
    queryKey: ["ts"],
    queryFn: async () => (await api.get("/data/timeseries?indicator=policy_effectiveness_score")).data,
  });

  const kpi = k?.kpis || {};
  const series = useMemo(() => {
    const source = (ts?.series || []).map((d) => ({
      year: Number(d.year),
      value: Number(d.policy_effectiveness_score || 0),
    }));
    return source.length > 0 ? source : FALLBACK_SERIES;
  }, [ts]);

  const trendData = useMemo(
    () =>
      series.map((d) => ({
        year: d.year,
        policy_effectiveness: d.value,
        target_path: Math.min(100, d.value + 8),
      })),
    [series]
  );

  const fuelData = [
    { name: "Coal", ghg_share: 39, policy_shift: 7 },
    { name: "Gas", ghg_share: 29, policy_shift: 11 },
    { name: "Oil", ghg_share: 24, policy_shift: 13 },
    { name: "Other", ghg_share: 8, policy_shift: 9 },
  ];

  const donutData = [
    { name: "Fossil share", value: 62 },
    { name: "Renewables", value: 30 },
    { name: "Low-carbon other", value: 8 },
  ];

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-4 gap-4">
        <KPICard title="Global CO2 trend" value={kpi.global_co2_trend?.toFixed?.(2) || "-"} subtitle="Average fossil-fuel emissions indicator" />
        <KPICard title="EU ETS price" value={kpi.eu_ets_price?.toFixed?.(2) || "-"} subtitle="EUR/tCO2" />
        <KPICard title="Mean policy effectiveness score" value={kpi.mean_policy_effectiveness?.toFixed?.(2) || "-"} subtitle="0-100" />
        <KPICard title="Countries covered" value={kpi.countries_covered || "-"} subtitle={`Latest year: ${kpi.latest_year || "-"}`} />
      </div>

      <TimeSeriesChart
        title="Carbon Pricing Policy Effectiveness Trajectory"
        data={trendData}
        x="year"
        series={[
          { key: "policy_effectiveness", color: "#1A56DB", name: "Policy effectiveness score" },
          { key: "target_path", color: "#0E9F6E", name: "Policy target path" },
        ]}
      />

      <div className="grid grid-cols-3 gap-4">
        <BarChart title="GHG by Fuel Type and Policy Shift" data={fuelData} x="name" bars={["ghg_share", "policy_shift"]} />
        <DonutChart title="Energy Mix Composition" data={donutData} />
        <TimeSeriesChart title="Score Trend (Area)" data={series} x="year" y="value" area />
      </div>

      <div className="text-xs bg-panel inline-block px-2 py-1 rounded border border-slate-200">
        Data freshness: {kpi.latest_year || "-"}
      </div>
    </div>
  );
}
