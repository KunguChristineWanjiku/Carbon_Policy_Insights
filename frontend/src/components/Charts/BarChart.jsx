import React from "react";
import { ResponsiveContainer, BarChart as RBC, Bar, XAxis, YAxis, Tooltip, Legend, CartesianGrid } from "recharts";
import { toFriendlyLabel } from "../../utils/labels";

const COLORS = ["#1A56DB", "#0E9F6E", "#E02424", "#7E3AF2", "#FF8A4C", "#06B6D4"];

export default function BarChart({ data = [], x = "name", bars = ["a", "b"], title }) {
  const hasData = Array.isArray(data) && data.length > 0;

  return (
    <div className="card h-80">
      {title ? <h4 className="font-semibold mb-2">{title}</h4> : null}
      {!hasData ? (
        <div className={`${title ? "h-[90%]" : "h-full"} flex items-center justify-center text-sm text-muted bg-slate-50 rounded`}>
          No bar-chart data available.
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={title ? "92%" : "100%"}>
          <RBC data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={x} />
            <YAxis />
            <Tooltip formatter={(value, name) => [value, toFriendlyLabel(name)]} />
            <Legend />
            {bars.map((b, i) => (
              <Bar key={b} dataKey={b} name={toFriendlyLabel(b)} fill={COLORS[i % COLORS.length]} />
            ))}
          </RBC>
        </ResponsiveContainer>
      )}
    </div>
  );
}
