const LABEL_OVERRIDES = {
  year: "Year",
  country_name: "Country",
  country_iso3: "Country Code",
  model: "Model",

  carbon_price_eur: "Carbon Price (EUR/tCO2)",
  eu_ets_carbon_price_eur: "Carbon Price (EUR/tCO2)",
  ghg_target_pct: "Emissions Reduction Target (%)",
  ghg_reduction_pct: "Emissions Reduction (%)",
  ghg_reduction: "Emissions Reduction (%)",
  policy_effectiveness_score: "Policy Effectiveness Score",
  policy_effectiveness: "Policy Effectiveness Score",
  effectiveness: "Policy Effectiveness Score",
  policy_shift: "Policy Shift",
  target_path: "Target Path",
  carbon_price: "Carbon Price (EUR/tCO2)",

  energy_intensity: "Energy Use Intensity",
  renewable_share: "Renewable Energy Share (%)",
  renewable_target: "Renewable Energy Target (%)",
  fossil_fuel_share: "Fossil Fuel Share (%)",
  carbon_price_level: "Carbon Price (EUR/tCO2)",
  gdp_per_capita: "GDP per Person",
  gdp_growth: "GDP Growth (%)",
  policy_stringency_index: "Policy Strength Index",
  industrial_output_index: "Industrial Activity Index",
  lag_1yr_ghg: "Last Year's Emissions Index",
  lag_3yr_ghg: "Three-Year Emissions Index",
  phaseout_rate: "Fossil Fuel Phase-Out Rate (%/yr)",
  efficiency_improvement: "Energy Efficiency Improvement (%/yr)",
  cbam: "Cross-Border Carbon Adjustment",
  ghg_share: "Emissions Share (%)",
  global_co2_trend: "Global CO2 Trend",
  mean_policy_effectiveness: "Average Policy Effectiveness",
  countries_covered: "Countries Covered",
  latest_year: "Latest Year",
  avg_mae: "Average MAE",
  avg_rmse: "Average RMSE",
};

const MODEL_OVERRIDES = {
  random_forest: "Random Forest",
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
  catboost: "CatBoost",
  ffnn: "Neural Network (FFNN)",
  lstm: "LSTM Network",
  ensemble: "Ensemble Model",
};

const WORD_OVERRIDES = {
  eu: "EU",
  ets: "ETS",
  ghg: "GHG",
  gdp: "GDP",
  co2: "CO2",
  mae: "MAE",
  rmse: "RMSE",
  xai: "XAI",
};

function titleCaseWord(word) {
  const lower = String(word || "").toLowerCase();
  if (WORD_OVERRIDES[lower]) return WORD_OVERRIDES[lower];
  if (!lower) return "";
  return lower.charAt(0).toUpperCase() + lower.slice(1);
}

export function toFriendlyLabel(key) {
  if (key === null || key === undefined) return "";
  const input = String(key).trim();
  if (!input) return "";
  if (LABEL_OVERRIDES[input]) return LABEL_OVERRIDES[input];

  return input
    .replace(/[_-]+/g, " ")
    .split(/\s+/)
    .map((word) => titleCaseWord(word))
    .join(" ");
}

export function toFriendlyModelName(model) {
  if (model === null || model === undefined) return "";
  const input = String(model).trim();
  if (!input) return "";
  return MODEL_OVERRIDES[input] || toFriendlyLabel(input);
}
