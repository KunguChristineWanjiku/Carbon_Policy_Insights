import React from "react";
import { useMutation } from "@tanstack/react-query";
import toast from "react-hot-toast";
import api from "../api/client";
import useAppStore from "../store/useAppStore";
import SHAPPanel from "../components/XAI/SHAPPanel";
import LIMEPanel from "../components/XAI/LIMEPanel";
import NarrativePanel from "../components/XAI/NarrativePanel";

export default function Explainability() {
  const current = useAppStore((s) => s.currentPrediction);
  const features =
    current?.shap_top5?.reduce((a, x) => ({ ...a, [x.feature]: x.shap_value }), {}) || {
      policy_stringency_index: 50,
    };

  const shap = useMutation({
    mutationFn: async () =>
      (await api.post("/explain/shap", { features, sample_id: 1, model: current?.model_used || "ensemble" })).data,
  });

  const lime = useMutation({
    mutationFn: async () =>
      (await api.post("/explain/lime", { features, sample_id: 1, model: current?.model_used || "ensemble" })).data,
  });

  const narrative = useMutation({
    mutationFn: async () => (await api.post("/explain/narrative", { features, prediction: current?.prediction || {} })).data,
  });

  const load = async () => {
    await shap.mutateAsync();
    await lime.mutateAsync();
    await narrative.mutateAsync();
  };

  return (
    <div className="space-y-4">
      <button className="px-3 py-2 bg-accent text-white rounded" onClick={load}>
        Load full explanation
      </button>
      <div className="grid grid-cols-3 gap-4">
        <SHAPPanel shapTop5={shap.data?.shap_top5 || current?.shap_top5 || []} />
        <LIMEPanel limeWeights={lime.data?.lime_weights || []} agreement="0.78 +/- 0.10" />
        <NarrativePanel
          narrative={narrative.data?.narrative}
          loading={narrative.isPending}
          onRegenerate={() => narrative.mutate()}
          onCopy={(cleanedText) => {
            navigator.clipboard.writeText(cleanedText || narrative.data?.narrative || "");
            toast.success("Narrative copied");
          }}
        />
      </div>
    </div>
  );
}
