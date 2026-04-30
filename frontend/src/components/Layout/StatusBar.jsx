import React from "react";
import { useQuery } from "@tanstack/react-query";
import api from "../../api/client";
export default function StatusBar(){const {data}=useQuery({queryKey:["health"],queryFn:async()=> (await api.get("/health")).data});return <div className="px-6 pb-4 text-sm text-muted">API Status: <span className="font-semibold text-accent2">{data?.status||"loading"}</span></div>}
