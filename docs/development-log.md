# Development Log

## Overview

This log summarizes the major implementation phases for the Carbon Policy Insights project.
Dates below are approximate phase windows captured from project materials and code state.

## Phase 1: Core architecture and data setup

- Defined project structure for backend, frontend, model artefacts, and XAI outputs.
- Prepared engineered dataset and feature/target schema files.
- Set up baseline runtime and deployment config files.

## Phase 2: Modeling pipeline and API

- Implemented training/evaluation pipeline with metrics and benchmarking outputs.
- Added backend API endpoints for prediction, explainability, and simulation.
- Integrated model loading, fallback behavior, and response formatting.

## Phase 3: Frontend application

- Built React app shell, routing, and shared state management.
- Added pages for dashboard, predictions, explainability, model metrics, policy simulator, and data explorer.
- Added chart and panel components for SHAP/LIME/narrative views.

## Phase 4: Explainability and output artefacts

- Added model residual and comparison artefacts under `models/`.
- Added SHAP summary/dependence/waterfall outputs and narrative payloads under `xai/`.
- Wired frontend to API explainability flows.

## Phase 5: UX and repository hygiene

- Improved user-facing labels and reduced technical jargon in UI.
- Removed non-documentation comments and obsolete scaffolding files.
- Reorganized history into smaller commits for clearer project evolution.
