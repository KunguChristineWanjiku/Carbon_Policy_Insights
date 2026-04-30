import React from "react";
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend } from "recharts";

const COLORS = ["#1A56DB", "#0E9F6E", "#FF8A4C", "#E02424", "#7E3AF2", "#06B6D4"];

export default function DonutChart({ data = [], title, valueKey = "value", nameKey = "name" }) {
  const hasData = Array.isArray(data) && data.length > 0;
  const total = hasData ? data.reduce((acc, item) => acc + (Number(item[valueKey]) || 0), 0) : 0;

  return (
    <div className="card h-80">
      {title ? <h4 className="font-semibold mb-2">{title}</h4> : null}
      {!hasData ? (
        <div className={`${title ? "h-[90%]" : "h-full"} flex items-center justify-center text-sm text-muted bg-slate-50 rounded`}>
          No composition data available.
        </div>
      ) : (
        <div className={`relative ${title ? "h-[90%]" : "h-full"}`}>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie data={data} dataKey={valueKey} nameKey={nameKey} innerRadius={60} outerRadius={95} paddingAngle={2}>
                {data.map((entry, index) => (
                  <Cell key={`${entry[nameKey]}-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="text-center">
              <div className="text-xs text-muted">Total</div>
              <div className="font-bold text-lg">{total.toFixed(1)}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
