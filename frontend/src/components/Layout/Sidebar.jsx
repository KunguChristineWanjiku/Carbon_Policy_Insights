import React from "react";
import { NavLink } from "react-router-dom";

const links = [
  ["/", "Dashboard"],
  ["/predictions", "Quick Estimate"],
  ["/explainability", "Why This Result"],
  ["/policy-simulator", "Scenario Planner"],
  ["/model-metrics", "Model Performance"],
  ["/data-explorer", "Data Explorer"],
];

export default function Sidebar() {
  return (
    <aside className="w-64 min-h-screen bg-panel border-r border-slate-200 p-4 sticky top-0">
      <h1 className="text-lg font-bold mb-6">Carbon Policy Insights</h1>
      <nav className="space-y-2">
        {links.map(([to, label]) => (
          <NavLink key={to} to={to} className={({ isActive }) => `block px-3 py-2 rounded ${isActive ? "bg-accent text-white" : "hover:bg-slate-100"}`}>
            {label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
