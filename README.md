# 🌍 World Cup 2026 Prediction

Machine learning project for predicting FIFA World Cup 2026 match outcomes, scorelines and eventually player scoring probabilities.

---

## Project Goals

- Build a clean match-level training dataset from historical international football data.
- Predict match result probabilities: home win, draw, away win.
- Predict realistic scorelines using goal models.
- Extend later to player scorer predictions using lineup and player-level data.

---

## Current Workflow

1. Load historical datasets.
2. Clean dates, duplicates and team-name compatibility.
3. Perform exploratory data analysis.
4. Build a merged match-level training table.
5. Engineer team-strength, form, goal, rest and home-advantage features.
6. Train baseline models.
7. Predict World Cup 2026 fixtures.

---

## Main Datasets

**Training datasets:**
- `Historical international match results.csv`
- `Elo ratings.csv`
- `FIFA rankings.csv`
- `Recent form.csv`
- `Goals scored conceded.csv`
- `Home region advantage.csv`
- `Rest days.csv`
- `Tournament stage.csv`
- `fivethirtyeight_spi_international.csv` (SPI ratings & rolling xG)
- `international_football_odds_validated_best_match.csv` (historical betting odds)

**Prediction datasets:**
- `WC 2026 match schedule advanced.csv`
- `Team strength ratings.csv`
- `Team expected goals.csv`
- `Market match odds.csv`
- `wc_2026_squad_market_values.csv`
- `wc_2026_top_players_market_values.csv`

---

## 🤖 How the Predictions Were Made

### 🏆 Winner Prediction
> **Model:** `RandomForestClassifier` (300 trees, max_depth=6, class_weight=balanced)
> **Notebook:** `final_model_improved_WC_2026_group_predi.ipynb`

Trained on **~30,000 historical international matches** using these features:
- Elo rating difference
- FIFA ranking & points difference
- Recent form (last 5 and last 10 games)
- Head-to-head record, win/unbeaten streaks
- Rolling xG (expected goals for/against)
- FiveThirtyEight SPI ratings (attack & defense)
- Squad & top player market values (Transfermarkt)
- Injury and suspension data
- Bookmaker implied probabilities (normalized)
- Home country advantage (Mexico, Canada, USA hosting)

Uses **Symmetric Double-Inference**: every match is predicted twice (home/away swapped), probabilities averaged, to remove listing-order bias on neutral venues.

A **betting edge override** is then applied: if the model finds a value edge ≥ 9% on an underdog win or ≥ 11% on a draw vs the bookmaker odds, the prediction is overridden. If model confidence is ≥ 60%, the model prediction is trusted directly without any market override.

---

### ⚽ Score Prediction
> **Model:** `HistGradientBoostingRegressor` → Poisson Sampling
> **Notebook:** `final_model_improved_WC_2026_group_predi+score_predi.ipynb`

Two separate `HistGradientBoostingRegressor` models (one for home goals, one for away goals) are trained on the same historical dataset. They output **expected goals (λ_home, λ_away)** for each team.

These λ values are then fed into **Poisson sampling**: the model randomly draws scorelines from `Poisson(λ_home)` and `Poisson(λ_away)` until it finds a score that is **consistent with the predicted winner**. This produces diverse, realistic scorelines (3-1, 2-0, 4-2, 0-0, etc.) instead of always predicting 1-0 or 1-1.

---

## 📈 Live Accuracy Tracking (Group Stage: June 11 – June 14)

Out of the first 12 matches, the model correctly predicted the outcome of **8 matches (66.7% accuracy)**:

| Match | Predicted Winner | Predicted Score | Actual Result | ✅/❌ |
|---|---|---|---|---|
| 🇲🇽 Mexico vs. South Africa 🇿🇦 | Mexico Win | 2-1 | 2-0 | ✅ |
| 🇰🇷 South Korea vs. Czechia 🇨🇿 | South Korea Win | 2-0 | 2-1 | ✅ |
| 🇨🇦 Canada vs. Bosnia 🇧🇦 | Canada Win | 2-1 | 1-1 | ❌ |
| 🇺🇸 USA vs. Paraguay 🇵🇾 | USA Win | 1-0 | 4-1 | ✅ |
| 🇶🇦 Qatar vs. Switzerland 🇨🇭 | Draw *(override)* | 1-1 | 1-1 | ✅ |
| 🇧🇷 Brazil vs. Morocco 🇲🇦 | Draw *(override)* | 0-0 | 1-1 | ✅ |
| 🇭🇹 Haiti vs. Scotland 🏴󠁧󠁢󠁳󠁣󠁴󠁿 | Draw *(override)* | 0-0 | 0-1 | ❌ |
| 🇦🇺 Australia vs. Turkey 🇹🇷 | Draw *(override)* | 1-1 | 2-0 | ❌ |
| 🇩🇪 Germany vs. Curaçao 🇨🇼 | Germany Win | 1-0 | 7-1 | ✅ |
| 🇨🇮 Ivory Coast vs. Ecuador 🇪🇨 | Ecuador Win | 1-2 | 1-0 | ❌ |
| 🇳🇱 Netherlands vs. Japan 🇯🇵 | Draw *(override)* | 2-2 | 2-2 | ✅ |
| 🇸🇪 Sweden vs. Tunisia 🇹🇳 | Sweden Win | 1-0 | 5-1 | ✅ |

---

## 🔮 Full Group Stage Predictions

### Group A 🇦🇷 🇩🇿 🇦🇹 🇯🇴
| Match | Predicted Winner | Predicted Score |
|---|---|---|
| Argentina vs. Algeria | 🇦🇷 Argentina Win | 5-1 |
| Austria vs. Jordan | Draw | 2-2 |
| Argentina vs. Austria | 🇦🇷 Argentina Win | 4-0 |
| Jordan vs. Algeria | Draw | 1-1 |
| Algeria vs. Austria | 🇦🇹 Austria Win | 2-4 |
| Jordan vs. Argentina | Draw | 2-2 |

### Group B 🇺🇸 🇵🇾 🇦🇺 🇹🇷
| Match | Predicted Winner | Predicted Score |
|---|---|---|
| United States vs. Paraguay | 🇺🇸 USA Win | 1-0 |
| Australia vs. Turkey | Draw | 1-1 |
| United States vs. Australia | Draw | 1-1 |
| Turkey vs. Paraguay | 🇹🇷 Turkey Win | 3-0 |
| Paraguay vs. Australia | 🇵🇾 Paraguay Win | 2-1 |
| United States vs. Turkey | Draw | 2-2 |

### Group C 🇮🇷 🇧🇪 🇳🇿 🇪🇬
| Match | Predicted Winner | Predicted Score |
|---|---|---|
| Belgium vs. Egypt | Draw | 0-0 |
| Iran vs. New Zealand | 🇮🇷 Iran Win | 4-1 |
| Belgium vs. Iran | Draw | 0-0 |
| New Zealand vs. Egypt | Draw | 2-2 |
| Egypt vs. Iran | 🇮🇷 Iran Win | 0-1 |
| New Zealand vs. Belgium | Draw | 1-1 |

### Group D 🇨🇭 🇨🇦 🇧🇦 🇶🇦
| Match | Predicted Winner | Predicted Score |
|---|---|---|
| Canada vs. Bosnia and Herzegovina | 🇨🇦 Canada Win | 2-1 |
| Qatar vs. Switzerland | Draw | 1-1 |
| Switzerland vs. Bosnia and Herzegovina | 🇨🇭 Switzerland Win | 3-0 |
| Canada vs. Qatar | Draw | 1-1 |
| Bosnia and Herzegovina vs. Qatar | Draw | 1-1 |
| Canada vs. Switzerland | 🇨🇭 Switzerland Win | 1-3 |

### Group E 🇧🇷 🇲🇦 🇭🇹 🏴󠁧󠁢󠁳󠁣󠁴󠁿
| Match | Predicted Winner | Predicted Score |
|---|---|---|
| Brazil vs. Morocco | Draw | 0-0 |
| Haiti vs. Scotland | Draw | 0-0 |
| Scotland vs. Morocco | Draw | 0-0 |
| Brazil vs. Haiti | 🇧🇷 Brazil Win | 3-1 |
| Morocco vs. Haiti | Draw | 0-0 |
| Scotland vs. Brazil | Draw | 0-0 |

### Group F 🇪🇸 🇺🇾 🇨🇻 🇸🇦
| Match | Predicted Winner | Predicted Score |
|---|---|---|
| Saudi Arabia vs. Uruguay | 🇺🇾 Uruguay Win | 0-3 |
| Spain vs. Cape Verde | 🇪🇸 Spain Win | 4-0 |
| Spain vs. Saudi Arabia | 🇪🇸 Spain Win | 5-0 |
| Uruguay vs. Cape Verde | 🇺🇾 Uruguay Win | 4-0 |
| Cape Verde vs. Saudi Arabia | Draw | 0-0 |
| Uruguay vs. Spain | 🇪🇸 Spain Win | 1-2 |

### Group G 🇵🇹 🇨🇴 🇨🇩 🇺🇿
| Match | Predicted Winner | Predicted Score |
|---|---|---|
| Portugal vs. DR Congo | Draw | 1-1 |
| Uzbekistan vs. Colombia | Draw | 1-1 |
| Portugal vs. Uzbekistan | Draw | 1-1 |
| Colombia vs. DR Congo | 🇨🇴 Colombia Win | 3-1 |
| DR Congo vs. Uzbekistan | 🇨🇩 DR Congo Win | 3-2 |
| Colombia vs. Portugal | 🇵🇹 Portugal Win | 2-4 |

### Group H 🏴󠁧󠁢󠁥󠁮󠁧󠁿 🇭🇷 🇬🇭 🇵🇦
| Match | Predicted Winner | Predicted Score |
|---|---|---|
| England vs. Croatia | Draw | 1-1 |
| Ghana vs. Panama | 🇵🇦 Panama Win | 1-3 |
| England vs. Ghana | 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England Win | 3-0 |
| Panama vs. Croatia | Draw | 1-1 |
| Panama vs. England | 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England Win | 1-2 |
| Croatia vs. Ghana | 🇭🇷 Croatia Win | 4-0 |

### Group I 🇪🇨 🇩🇪 🇨🇮 🇨🇼
| Match | Predicted Winner | Predicted Score |
|---|---|---|
| Germany vs. Curaçao | 🇩🇪 Germany Win | 1-0 |
| Ivory Coast vs. Ecuador | 🇪🇨 Ecuador Win | 1-2 |
| Germany vs. Ivory Coast | Draw | 1-1 |
| Ecuador vs. Curaçao | 🇪🇨 Ecuador Win | 2-0 |
| Curaçao vs. Ivory Coast | Draw | 1-1 |
| Ecuador vs. Germany | Draw | 2-2 |

### Group J 🇲🇽 🇿🇦 🇰🇷 🇨🇿
| Match | Predicted Winner | Predicted Score |
|---|---|---|
| Mexico vs. South Africa | 🇲🇽 Mexico Win | 2-1 |
| South Korea vs. Czech Republic | 🇰🇷 South Korea Win | 2-0 |
| Czech Republic vs. South Africa | Draw | 1-1 |
| Mexico vs. South Korea | 🇲🇽 Mexico Win | 2-0 |
| South Africa vs. South Korea | Draw | 1-1 |
| Mexico vs. Czech Republic | 🇲🇽 Mexico Win | 1-0 |

### Group K 🇫🇷 🇳🇴 🇸🇳 🇮🇶
| Match | Predicted Winner | Predicted Score |
|---|---|---|
| France vs. Senegal | Draw | 1-1 |
| Iraq vs. Norway | Draw | 1-1 |
| France vs. Iraq | 🇫🇷 France Win | 2-0 |
| Norway vs. Senegal | 🇳🇴 Norway Win | 2-1 |
| Senegal vs. Iraq | Draw | 0-0 |
| Norway vs. France | 🇫🇷 France Win | 0-1 |

### Group L 🇯🇵 🇳🇱 🇸🇪 🇹🇳
| Match | Predicted Winner | Predicted Score |
|---|---|---|
| Netherlands vs. Japan | Draw | 2-2 |
| Sweden vs. Tunisia | 🇸🇪 Sweden Win | 1-0 |
| Netherlands vs. Sweden | 🇳🇱 Netherlands Win | 4-2 |
| Tunisia vs. Japan | 🇯🇵 Japan Win | 1-2 |
| Japan vs. Sweden | 🇯🇵 Japan Win | 4-1 |
| Tunisia vs. Netherlands | Draw | 1-1 |

---

## 🔮 Predicted Knockout Qualifiers (Round of 32)

Based on simulated group standings, these 32 teams are predicted to advance:

### Direct Qualifiers (Top 2 in Group)
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

### Wildcard Qualifiers (Best 3rd-Place Teams)
1. Panama 🇵🇦 (5 pts)
2. DR Congo 🇨🇩 (4 pts)
3. Paraguay 🇵🇾 (3 pts)
4. Scotland 🏴󠁧󠁢󠁳󠁣󠁴󠁿 (3 pts)
5. Jordan 🇯🇴 (3 pts)
6. Sweden 🇸🇪 (3 pts)
7. Qatar 🇶🇦 (3 pts)
8. Senegal 🇸🇳 (2 pts)

---

## Modeling Plan

**Winner prediction:**
- Logistic Regression baseline
- ✅ Random Forest (current production model)
- XGBoost / LightGBM / CatBoost

**Score prediction:**
- Poisson regression baseline
- ✅ HistGradientBoosting → Poisson sampling (current production model)
- Dixon-Coles style model

**Player scorer prediction:**
- Expected minutes
- xG per 90
- Shots per 90
- Penalty/free-kick taker status
- Opponent defensive strength

---

## Notes

The project uses time-aware splitting because football data is chronological. All models are trained only on matches before June 11, 2026 to avoid data leakage.
