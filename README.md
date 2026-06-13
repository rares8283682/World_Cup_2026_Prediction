cat > README.md <<'EOF'
# World Cup 2026 Prediction

Machine learning project for predicting FIFA World Cup 2026 match outcomes, likely scores and eventually player scoring probabilities.

## Project Goals

- Build a clean match-level training dataset from historical international football data.
- Predict match result probabilities: home win, draw, away win.
- Predict likely scorelines using goal models.
- Extend later to player scorer predictions using lineup and player-level data.

## Current Workflow

1. Load historical datasets.
2. Clean dates, duplicates and team-name compatibility.
3. Perform exploratory data analysis.
4. Build a merged match-level training table.
5. Engineer team-strength, form, goal, rest and home-advantage features.
6. Train baseline models.
7. Predict World Cup 2026 fixtures.

## Main Datasets

Training datasets:

- `Historical international match results.csv`
- `Elo ratings.csv`
- `FIFA rankings.csv`
- `Recent form.csv`
- `Goals scored conceded.csv`
- `Home region advantage.csv`
- `Rest days.csv`
- `Tournament stage.csv`

Prediction datasets:

- `WC 2026 fixtures.csv`
- `WC 2026 match schedule.csv`
- `Team strength ratings.csv`
- `Team expected goals.csv`
- `Market match odds.csv`
- `Travel distances.csv`
- `Venue characteristics.csv`

Player scorer datasets are stored separately in:

- `player_scorer_predictions_datasets/`

## Modeling Plan

Winner prediction:

- Logistic Regression baseline
- Random Forest
- XGBoost / LightGBM / CatBoost

Score prediction:

- Poisson regression baseline
- Dixon-Coles style model
- Gradient boosting regression for expected goals

Player scorer prediction:

- Expected minutes
- xG per 90
- shots per 90
- penalty/free-kick taker status
- opponent defensive strength

## Notes

The notebook currently focuses on data validation, EDA and feature preparation. The project uses time-aware splitting because football data is chronological.
EOF
