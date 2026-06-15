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

## 🏆 World Cup 2026 Predictions & Live Performance

### 📈 Live Accuracy Tracking (Group Stage: June 11 - June 14)
Out of the first 12 matches, the Random Forest model correctly predicted the outcome of **8 matches (66.7% accuracy)**, including three spot-on draw overrides:
* 🇲🇽 Mexico vs. South Africa 🇿🇦 (Predicted: Mexico Win) ➔ 2-0 ✅
* 🇰🇷 South Korea vs. Czechia 🇨🇿 (Predicted: South Korea Win) ➔ 2-1 ✅
* 🇨🇦 Canada vs. Bosnia 🇧🇦 (Predicted: Canada Win) ➔ 1-1 ❌
* 🇺🇸 USA vs. Paraguay 🇵🇾 (Predicted: USA Win) ➔ 4-1 ✅
* 🇶🇦 Qatar vs. Switzerland 🇨🇭 (Predicted: Draw) ➔ 1-1 ✅ *(Custom Draw override)*
* 🇧🇷 Brazil vs. Morocco 🇲🇦 (Predicted: Draw) ➔ 1-1 ✅ *(Custom Draw override)*
* 🇭🇹 Haiti vs. Scotland 🏴󠁧󠁢󠁳󠁣󠁴󠁿 (Predicted: Draw) ➔ 0-1 ❌
* 🇦🇺 Australia vs. Turkey 🇹🇷 (Predicted: Draw) ➔ 2-0 ❌
* 🇩🇪 Germany vs. Curaçao 🇨🇼 (Predicted: Germany Win) ➔ 7-1 ✅
* 🇨🇮 Ivory Coast vs. Ecuador 🇪🇨 (Predicted: Ecuador Win) ➔ 1-0 ❌
* 🇳🇱 Netherlands vs. Japan 🇯🇵 (Predicted: Draw) ➔ 2-2 ✅ *(Custom Draw override)*
* 🇸🇪 Sweden vs. Tunisia 🇹🇳 (Predicted: Sweden Win) ➔ 5-1 ✅

### 🔮 Predicted Knockout Qualifiers (Round of 32)
Based on simulated group standings, these 32 teams are predicted to advance:

#### Direct Qualifiers (Top 2 in Group)
* **Group A:** Argentina 🇦🇷, Austria 🇦🇹
* **Group B:** Turkey 🇹🇷, United States 🇺🇸
* **Group C:** Iran 🇮🇷, Belgium 🇧🇪
* **Group D:** Switzerland 🇨🇭, Canada 🇨🇦
* **Group E:** Brazil 🇧🇷, Morocco 🇲🇦
* **Group F:** Spain 🇪🇸, Uruguay 🇺🇾
* **Group G:** Portugal 🇵🇹, Colombia 🇨🇴
* **Group H:** England 🏴󠁧󠁢󠁥󠁮󠁧󠁿, Croatia 🇭🇷
* **Group I:** Ecuador 🇪🇨, Germany 🇩🇪
* **Group J:** Mexico 🇲🇽, South Korea 🇰🇷
* **Group K:** France 🇫🇷, Norway 🇳🇴
* **Group L:** Japan 🇯🇵, Netherlands 🇳🇱

#### Wildcard Qualifiers (Best 3rd-Place Teams)
1. Panama 🇵🇦 (5 pts)
2. DR Congo 🇨🇩 (4 pts)
3. Paraguay 🇵🇾 (3 pts)
4. Scotland 🏴󠁧󠁢󠁳󠁣󠁴󠁿 (3 pts)
5. Jordan 🇯🇴 (3 pts)
6. Sweden 🇸🇪 (3 pts)
7. Qatar 🇶🇦 (3 pts)
8. Senegal 🇸🇳 (2 pts)

EOF
