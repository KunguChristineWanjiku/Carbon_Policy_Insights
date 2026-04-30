import { create } from "zustand";
const useAppStore = create((set) => ({
  currentPrediction: null,
  selectedScenarioIds: [],
  filters: { country: "", region: "", income: "", yearRange: [1971, 2024] },
  setPrediction: (currentPrediction) => set({ currentPrediction }),
  setScenarioIds: (selectedScenarioIds) => set({ selectedScenarioIds }),
  setFilters: (filters) => set({ filters }),
}));
export default useAppStore;
