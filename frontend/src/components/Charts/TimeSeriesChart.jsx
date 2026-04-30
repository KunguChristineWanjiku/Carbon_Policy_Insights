import React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Area,
  AreaChart,
  Legend,
} from "recharts";

const DEFAULT_SERIES = [
  { key: "value", color: "#1A56DB", name: "Value" },
];

export default function TimeSeriesChart({
  data = [],
  x = "year",
  y = "value",
  area = false,
  title,
  series = DEFAULT_SERIES,
}) {
  const hasData = Array.isArray(data) && data.length > 0;
  const chartSeries = Array.isArray(series) && series.length > 0 ? series : DEFAULT_SERIES;
  const ChartComponent = area ? AreaChart : LineChart;

  return (
    <div className="card h-80">
      {title ? <h4 className="font-semibold mb-2">{title}</h4> : null}
      {!hasData ? (
        <div className={`${title ? "h-[90%]" : "h-full"} flex items-center justify-center text-sm text-muted bg-slate-50 rounded`}>
          No time-series data available for this selection.
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={title ? "92%" : "100%"}>
          <ChartComponent data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={x} />
            <YAxis />
            <Tooltip />
            {chartSeries.length > 1 ? <Legend /> : null}
            {area ? (
              <Area type="monotone" dataKey={y} stroke="#1A56DB" fill="#1A56DB33" />
            ) : (
              chartSeries.map((s) => (
                <Line key={s.key} type="monotone" dataKey={s.key} name={s.name} stroke={s.color} strokeWidth={2} dot={false} />
              ))
            )}
          </ChartComponent>
        </ResponsiveContainer>
      )}
    </div>
  );
}
