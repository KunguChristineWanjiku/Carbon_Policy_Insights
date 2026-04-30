import React from "react";
export default function KPICard({title,value,subtitle}){return <div className="card"><div className="text-sm text-muted">{title}</div><div className="text-2xl font-bold mt-1">{value}</div><div className="text-xs text-muted mt-1">{subtitle}</div></div>}
