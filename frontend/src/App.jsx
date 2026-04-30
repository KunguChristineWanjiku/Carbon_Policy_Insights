import React from "react";
import { Routes, Route } from "react-router-dom";
import Sidebar from "./components/Layout/Sidebar";
import Header from "./components/Layout/Header";
import StatusBar from "./components/Layout/StatusBar";
import Dashboard from "./pages/Dashboard";
import Predictions from "./pages/Predictions";
import Explainability from "./pages/Explainability";
import PolicySimulator from "./pages/PolicySimulator";
import ModelMetrics from "./pages/ModelMetrics";
import DataExplorer from "./pages/DataExplorer";

export default function App() {
  return (
    <div className="min-h-screen bg-bg text-textc font-[Inter]">
      <div className="flex">
        <Sidebar />
        <main className="flex-1">
          <Header />
          <StatusBar />
          <div className="p-6">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/predictions" element={<Predictions />} />
              <Route path="/explainability" element={<Explainability />} />
              <Route path="/policy-simulator" element={<PolicySimulator />} />
              <Route path="/model-metrics" element={<ModelMetrics />} />
              <Route path="/data-explorer" element={<DataExplorer />} />
            </Routes>
          </div>
        </main>
      </div>
    </div>
  );
}
