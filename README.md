# ⚽ FIFA World Cup 2026: Data-Driven Analysis & Winner Prediction

An interactive Streamlit application that analyzes 96 years of FIFA World Cup history (1930–2026) and predicts the most likely winner of the 2026 tournament using a data-driven scoring model — validated against real, live 2026 match results.

🔗 **Live App:** *(add your Render link here after deployment)*

## 📊 Project Overview

This project combines five datasets covering World Cup editions, historical matches, the 2026 qualified teams, the full 2026 fixture list, and Golden Boot winners, to:

- Explore historical trends (champions, goals per match, tournament growth)
- Identify which countries have been most consistently strong (semi-final appearances)
- Build a weighted prediction model for the WC 2026 winner
- Validate that model against real matches already played in the 2026 tournament
- Let users interactively check any country's predicted win probability

## 🧠 Workflow

1. **Data Loading & Cleaning** — Loaded and validated 5 CSV datasets spanning 1930–2026
2. **Exploratory Data Analysis** — Champions, goals trends, tournament size growth, semi-final consistency
3. **Prediction Model** — Weighted score: FIFA Ranking (40%) + Semi-Final History (30%) + World Cup Titles (20%) + Host Advantage (10%)
4. **Model Validation** — Tested against 30 real WC 2026 group stage matches (through June 28, 2026) — **76.2% accuracy** on decisive outcomes
5. **Interactive Deployment** — Built as a multi-page Streamlit app and deployed publicly

## 🔮 Key Result

The model predicts **Brazil** as the most likely WC 2026 champion, driven by their historical dominance (5 titles, 9 semi-final appearances, 89 goals scored across all tournaments) combined with a strong current FIFA ranking — followed by Germany and Argentina.

## 🛠️ Tech Stack

- **Python** — pandas, NumPy for data processing
- **Matplotlib** — visualizations
- **Streamlit** — interactive web application
- **Render** — deployment

## 📁 Datasets

| File | Description |
|------|-------------|
| `wc_all_editions.csv` | All 22 World Cup tournaments, 1930–2026 (champions, goals, attendance) |
| `wc_all_matches.csv` | 184 historical match results with stage, score, venue |
| `wc_2026_teams.csv` | All 48 qualified teams with FIFA ranking and confederation |
| `wc_2026_fixtures.csv` | All 104 scheduled 2026 matches |
| `wc_top_scorers.csv` | Golden Boot winners for every tournament |

Source: Kaggle FIFA World Cup Dataset

## 🚀 Running Locally

```bash
git clone https://github.com/Aasthapandit/worldcup-2026-prediction.git
cd worldcup-2026-prediction
pip install -r requirements.txt
streamlit run app.py
```

## ⚠️ Disclaimer

This model is based on historical patterns and current FIFA rankings. Football is inherently unpredictable — the model achieved 76.2% accuracy on decisive matches, meaning roughly 1 in 4 matches resulted in an upset. Predictions should be read as data-driven probabilities, not certainties.

## 👩‍💻 Author

**Aastha Pandit**
MS in Data Science, Eastern University
[LinkedIn](https://www.linkedin.com/in/aastha-pandit-842aa4228) | [GitHub](https://github.com/Aasthapandit)
