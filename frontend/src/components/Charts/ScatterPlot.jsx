import React from "react";
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
export default function ScatterPlot({data,x="x",y="y"}){return <div className="card h-80"><ResponsiveContainer width="100%" height="100%"><ScatterChart><CartesianGrid/><XAxis dataKey={x}/><YAxis dataKey={y}/><Tooltip/><Scatter data={data} fill="#1A56DB"/></ScatterChart></ResponsiveContainer></div>}
