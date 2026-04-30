import React, { useState } from "react";
import ScenarioBuilder from "../components/Policy/ScenarioBuilder";
import ScenarioCompare from "../components/Policy/ScenarioCompare";
export default function PolicySimulator(){const [scenarios,setScenarios]=useState([]);return <div className="grid grid-cols-2 gap-4"><ScenarioBuilder onSaved={(s)=>setScenarios((p)=>[...p,s].slice(-4))} latestResult={scenarios[scenarios.length-1]}/><ScenarioCompare scenarios={scenarios}/></div>}
