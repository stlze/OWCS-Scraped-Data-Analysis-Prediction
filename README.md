# Overwatch Champions Series — FaceIT Analytics & Predictions

A complete analytics and machine learning project built using data from the **Overwatch Champions Series (FaceIT)**.  
This repository covers **NA** and **EMEA** regions across **Stages 1–2** and partial **Stage 3**, providing full end-to-end processing — from data scraping and cleaning to model training and visualization in Power BI.

---

## Overview

This project combines data engineering, exploratory analysis, and predictive modeling for competitive Overwatch.  
It enables both analytical insights and machine learning training.

### Key Components
- Data scraping from FaceIT's public API  
- Cleaning and aggregation of player, team, and match data  
- Exploratory data analysis (EDA) and feature engineering  
- Match outcome prediction using machine learning  
- Power BI dashboard for data visualization

---

## Usage

1. Open **FaceIT Developer Portal** and create a new **API key (Server-Side)**.  
2. Replace the `API_KEY` variable in **Scraping data from Faceit API.py** with your key.  
3. Go to a match (room) on FaceIT and copy the **room ID**.  
4. Paste the ID into the `MATCH_IDS` list inside the script.  
5. Run **Cleaning the dataset.py** to rename players from FaceIT handles to in-game names.  
6. Run other scripts as needed — they are structured and documented for sequential use.

---

## Known Data Limitations

Some matches on FaceIT do not have full scoreboard data available.  
These missing records may slightly reduce analytical accuracy.

Examples:
- [Team Peps vs Vision Esports](https://www.faceit.com/en/ow2/room/1-c360b538-f56a-4ed3-b63d-b8ba6d2162bf/scoreboard) — Missing map 2  
- [Team Liquid vs NTMR](https://www.faceit.com/en/ow2/room/1-31dfb3e9-74c0-4c63-8985-8e1484a8eaba/scoreboard) — Missing map 1 (Stage 1)  
- [Team Liquid vs Spacestation](https://www.faceit.com/en/ow2/room/1-1d2eaf73-7387-4f1f-b6c8-659dae4ca25f/scoreboard) — Missing map 1 (Stage 2)  
- [Team Liquid vs DhillDucks](https://www.faceit.com/en/ow2/room/1-6b6c093c-1473-464d-859e-dbb92aee4aba/scoreboard) — Entire scoreboard missing (Stage 2)  
- [Twisted Minds vs Team Peps](https://www.faceit.com/en/ow2/room/1-9a84ec70-9596-4879-91d4-b96aedf6002a/scoreboard) — Missing map 3 (Stage 2)  
- [Alqadsiah vs Frost Tails](https://www.faceit.com/en/ow2/room/1-a9cfc90c-2290-427d-ad82-3c30346e8cb7/scoreboard) — Missing map 3 (Stage 2)

---

## Prediction Models

This project provides **two types** of predictive systems:  
one based on **advanced ML (Python scripts)** and another on **statistical rolling averages (Jupyter)**.

---

### 1. `.py` Simulation Files

Implements both **Logistic Regression** and **Random Forest** models (with calibration).  
Also includes a **custom Elo rating system** for dynamic performance weighting.

#### Example Output
Twisted Minds vs Al qadsiah
Twisted Minds win probability: 0.76
Al qadsiah win probability: 0.24

Per Map Type Predictions:
Hybrid: Twisted Minds 0.80 | Al qadsiah 0.20 -> Favored: Twisted Minds
Control: Twisted Minds 0.72 | Al qadsiah 0.28 -> Favored: Twisted Minds
This script provides console-based predictions and map-by-map win probabilities for any matchup.

---

### 2. `.jupyter` Notebook Version

Uses **RandomForestClassifier** to predict match outcomes:
- **Win (1)**  
- **Loss (0)**  
- **Draw (2)**  

#### Predictors
- Damage Dealt  
- Healing Done  
- Final Blows  
- KD Ratio  
- Encoded team, player, and role identifiers  

#### Dataset Split
- **Training set:** Matches before July 2025 (Stages 1 & 2)  
- **Test set:** Matches after July 2025 (Stage 3)

#### Output
Exports a file containing predicted outcomes for all head-to-head matchups.

---

## Outputs

| File | Description |
|------|--------------|
Prediction Notebook.jupyter:
| `predictions_all.csv` | Predicted outcomes for every possible team matchup |
Other .py files:
| `latest_team_stats.csv` | Rolling team performance averages for Power BI dashboard |
| `faceit_all_matches_emea_na_all_stages.csv` | Cleaned master dataset including all matches |
| `team_match.csv` / `team_map.csv` | Aggregated team-level statistics by match and map |

---

## Tools and Technologies

- **Python**: pandas, numpy, scikit-learn, matplotlib  
- **Power BI**: Interactive dashboards (Players, Teams, Maps, Bans, Predictions)  
- **Data Source**: FaceIT Overwatch Champions Series API  

---

## Credits

- **Author:** Majed Almusayhil 
- **Data Source:** FaceIT (Overwatch Champions Series)  
- **License:** MIT  
- **Year:** 2025  

---
