# 🌍 World Cup 2026 Prediction

Machine learning project for predicting FIFA World Cup 2026 match outcomes, scorelines and eventually player scoring probabilities.

---

## Project Goals

* Build a clean match-level training dataset from historical international football data.
* Predict match result probabilities: home win, draw, away win.
* Predict realistic scorelines using goal models.
* Extend later to player scorer predictions using lineup and player-level data.

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
* `Historical international match results.csv`
* `Elo ratings.csv`
* `FIFA rankings.csv`
* `Recent form.csv`
* `Goals scored conceded.csv`
* `Home region advantage.csv`
* `Rest days.csv`
* `Tournament stage.csv`
* `fivethirtyeight_spi_international.csv` (SPI ratings & rolling xG)
* `international_football_odds_validated_best_match.csv` (historical betting odds)

**Prediction datasets:**
* `WC 2026 match schedule advanced.csv`
* `Team strength ratings.csv`
* `Team expected goals.csv`
* `Market match odds.csv`
* `wc_2026_squad_market_values.csv`
* `wc_2026_top_players_market_values.csv`

---

## 🤖 How the Predictions Were Made (Historical Baseline)

### 🏆 Winner Prediction
> **Model:** `RandomForestClassifier` (300 trees, max_depth=6, class_weight=balanced)
> **Notebook:** `final_model_improved_WC_2026_group_predi.ipynb`

Trained on **~30,000 historical international matches** using these features:
* Elo rating difference
* FIFA ranking and points difference
* Recent form (last 5 and last 10 games)
* Head-to-head record, win/unbeaten streaks
* Rolling xG (expected goals for/against)
* FiveThirtyEight SPI ratings (attack and defense)
* Squad and top player market values (Transfermarkt)
* Injury and suspension data
* Bookmaker implied probabilities (normalized)
* Home country advantage (Mexico, Canada and USA hosting)

Uses **Symmetric Double-Inference**: every match is predicted twice (home/away swapped), probabilities averaged, to remove listing-order bias on neutral venues.

A **betting edge override** is then applied: if the model finds a value edge ≥ 9% on an underdog win or ≥ 11% on a draw vs the bookmaker odds, the prediction is overridden. If model confidence is ≥ 60%, the model prediction is trusted directly without any market override.

---

### ⚽ Score Prediction
> **Model:** `HistGradientBoostingRegressor` → Poisson Sampling
> **Notebook:** `final_model_improved_WC_2026_group_predi+score_predi.ipynb`

Two separate `HistGradientBoostingRegressor` models (one for home goals and one for away goals) are trained on the same historical dataset. They output **expected goals (λ_home, λ_away)** for each team.

These λ values are then fed into **Poisson sampling**: the model randomly draws scorelines from `Poisson(λ_home)` and `Poisson(λ_away)` until it finds a score that is **consistent with the predicted winner**. This produces diverse, realistic scorelines (3-1, 2-0, 4-2, 0-0 etc.) instead of always predicting 1-0 or 1-1.

---

## 📈 Live Accuracy Tracking (Group Stage Matchdays 1-3)

Out of the first 32 played group stage matches, the model matched the actual results with significant precision. The live data has been integrated directly into our final simulation pipelines.

-----

## 🏆 Final Simulation Decisions and Outputs

### 1. Mixture of Experts (MoE) Predictions
To scale the accuracy further, the prediction model was upgraded to a Mixture of Experts architecture. The blended predictions use:
* **Stacked Logistic Regression (20% Weight)**: Combining odds and GNN embeddings.
* **CatBoost Classifier (45% Weight)**: Isotonic calibrated challenger.
* **XGBoost Classifier (20% Weight)**: Calibrated DART challenger.
* **Team Embedding GNN (15% Weight)**: Deep representation learning model.

This ensemble generated the outcomes for all upcoming matches.

### 2. Goals Regressors
We trained expected goals models (`PoissonRegressor` and `XGBRegressor` with a poisson objective) on `full_df`. Goal predictions are generated dynamically using a blended model (70% Poisson + 30% XGBoost).

### 3. Group Standings (Leaderboards)

Here are the simulated group tables after all 72 group stage matches:

#### Group A
1. Argentina (9 pts, GD +5)
2. Austria (4 pts, GD +1)
3. Algeria (2 pts, GD -3)
4. Jordan (1 pt, GD -3)

#### Group B
1. United States (7 pts, GD +5)
2. Australia (4 pts, GD 0)
3. Paraguay (4 pts, GD -2)
4. Turkey (1 pt, GD -3)

#### Group C
1. Belgium (5 pts, GD +1)
2. Egypt (5 pts, GD +1)
3. Iran (3 pts, GD 0)
4. New Zealand (1 pt, GD -2)

#### Group D
1. Canada (5 pts, GD +6)
2. Switzerland (5 pts, GD +3)
3. Bosnia and Herzegovina (4 pts, GD -2)
4. Qatar (1 pt, GD -7)

#### Group E
1. Morocco (7 pts, GD +2)
2. Scotland (6 pts, GD +1)
3. Brazil (4 pts, GD +2)
4. Haiti (0 pts, GD -5)

#### Group F
1. Spain (7 pts, GD +2)
2. Uruguay (4 pts, GD 0)
3. Saudi Arabia (2 pts, GD -1)
4. Cape Verde (2 pts, GD -1)

#### Group G
1. Colombia (7 pts, GD +3)
2. Portugal (5 pts, GD +1)
3. DR Congo (2 pts, GD -1)
4. Uzbekistan (1 pt, GD -3)

#### Group H
1. England (6 pts, GD +2)
2. Croatia (6 pts, GD 0)
3. Ghana (3 pts, GD -1)
4. Panama (3 pts, GD -1)

#### Group I
1. Germany (7 pts, GD +7)
2. Ivory Coast (6 pts, GD +1)
3. Ecuador (4 pts, GD 0)
4. Curaçao (0 pts, GD -8)

#### Group J
1. Mexico (9 pts, GD +4)
2. South Africa (4 pts, GD -1)
3. South Korea (3 pts, GD -1)
4. Czech Republic (1 pt, GD -2)

#### Group K
1. France (9 pts, GD +4)
2. Norway (6 pts, GD +3)
3. Senegal (3 pts, GD -2)
4. Iraq (0 pts, GD -5)

#### Group L
1. Netherlands (7 pts, GD +2)
2. Japan (4 pts, GD 0)
3. Sweden (3 pts, GD +2)
4. Tunisia (3 pts, GD -4)

---

### ⚽ Group Stage Scoreboard

Below is the complete group stage scoreboard containing final scores (both actual and predicted):

* Mexico 2 - 0 South Africa
* South Korea 2 - 1 Czech Republic
* Canada 1 - 1 Bosnia and Herzegovina
* United States 4 - 1 Paraguay
* Qatar 1 - 1 Switzerland
* Brazil 1 - 1 Morocco
* Haiti 0 - 1 Scotland
* Australia 2 - 0 Turkey
* Germany 7 - 1 Curaçao
* Ivory Coast 1 - 0 Ecuador
* Netherlands 2 - 2 Japan
* Sweden 5 - 1 Tunisia
* Saudi Arabia 1 - 1 Uruguay
* Spain 0 - 0 Cape Verde
* Iran 2 - 2 New Zealand
* Belgium 1 - 1 Egypt
* France 3 - 1 Senegal
* Iraq 1 - 4 Norway
* Argentina 3 - 0 Algeria
* Austria 3 - 1 Jordan
* Portugal 1 - 1 DR Congo
* Uzbekistan 1 - 3 Colombia
* England 4 - 2 Croatia
* Ghana 1 - 0 Panama
* Switzerland 4 - 1 Bosnia and Herzegovina
* Canada 6 - 0 Qatar
* Czech Republic 1 - 1 South Africa
* Mexico 1 - 0 South Korea
* Scotland 0 - 1 Morocco
* Brazil 3 - 0 Haiti
* United States 2 - 0 Australia
* Turkey 0 - 1 Paraguay
* Germany 2 - 1 Ivory Coast
* Ecuador 2 - 1 Curaçao
* Netherlands 2 - 1 Sweden
* Tunisia 1 - 1 Japan
* Uruguay 2 - 1 Cape Verde
* Belgium 2 - 1 Iran
* New Zealand 1 - 1 Egypt
* Spain 2 - 1 Saudi Arabia
* Jordan 1 - 1 Algeria
* Argentina 2 - 1 Austria
* Norway 2 - 1 Senegal
* France 2 - 1 Iraq
* Portugal 2 - 1 Uzbekistan
* Colombia 2 - 1 DR Congo
* England 2 - 1 Ghana
* Panama 1 - 1 Croatia
* Mexico 2 - 1 Czech Republic
* South Africa 1 - 1 South Korea
* Canada 2 - 1 Switzerland
* Bosnia and Herzegovina 2 - 1 Qatar
* Scotland 1 - 1 Brazil
* Morocco 2 - 1 Haiti
* Tunisia 1 - 1 Netherlands
* Japan 2 - 1 Sweden
* Ecuador 2 - 1 Germany
* United States 2 - 1 Turkey
* Paraguay 2 - 1 Australia
* Curaçao 1 - 1 Ivory Coast
* Egypt 1 - 1 Iran
* New Zealand 1 - 1 Belgium
* Cape Verde 2 - 1 Saudi Arabia
* Uruguay 1 - 1 Spain
* Norway 2 - 1 France
* Senegal 2 - 1 Iraq
* Croatia 2 - 1 Ghana
* Panama 1 - 1 England
* DR Congo 2 - 1 Uzbekistan
* Jordan 1 - 1 Argentina
* Algeria 2 - 1 Austria
* Colombia 1 - 1 Portugal

---

### 🏆 Knockout Bracket Visualizer

Below is the visual tournament bracket predicted by the MoE pipeline from the Round of 32 to the World Cup Final:

```
 ROUND OF 32           ROUND OF 16         QUARTER-FINALS        SEMI-FINALS             FINAL            CHAMPION
 
 Germany (94%) 2-1 ────┐
                       ├─ Germany (88%) 1-1 ─┐
 Ghana (6%) 1-2 ───────┘                      │
                                              ├─ Germany (54%) 2-1 ─┐
 Morocco (51%) 2-1 ────┐                      │                     │
                       ├─ Morocco (12%) 1-1 ─┘                     │
 Iran (49%) 1-2 ───────┘                                            │
                                                                    ├─ Germany (49%) 1-2 ─┐
 Spain (93%) 2-1 ──────┐                                            │                     │
                       ├─ Spain (84%) 2-1 ───┐                      │                     │
 Egypt (7%) 1-2 ───────┘                     │                      │                     │
                                             ├─ Spain (46%) 1-2 ────┘                     │
 Austria (40%) 2-1 ────┐                     │                                            │
                       ├─ Australia (16%) 1-2┘                                            │
 Australia (60%) 1-2 ──┘                                                                  │
                                                                                          ├─ Brazil (65%) 2-1 ─── [ 🏆 BRAZIL ]
 England (95%) 2-1 ────┐                                                                  │
                       ├─ England (84%) 1-1 ─┐                                            │
 South Africa (5%) 1-2 ┘                      │                                            │
                                              ├─ England (45%) 1-2 ─┐                      │
 Norway (55%) 2-1 ─────┐                      │                     │                      │
                       ├─ Norway (16%) 1-1 ──┘                     │                      │
 Japan (46%) 1-2 ──────┘                                            │                      │
                                                                    ├─ Brazil (51%) 2-1 ──┘
 Canada (47%) 1-2 ─────┐                                            │
                       ├─ Paraguay (8%) 1-2 ─┐                      │
 Paraguay (53%) 2-1 ───┘                     │                      │
                                             ├─ Brazil (55%) 2-1 ───┘
 Colombia (18%) 1-2 ───┐                     │
                       ├─ Brazil (92%) 2-1 ──┘
 Brazil (82%) 2-1 ─────┘

 Belgium (67%) 2-1 ────┐
                       ├─ Belgium (68%) 2-1 ─┐
 Uruguay (33%) 1-2 ────┘                      │
                                              ├─ Belgium (36%) 1-2 ─┐
 Scotland (54%) 2-1 ───┐                      │                     │
                       ├─ Scotland (32%) 1-2 ┘                     │
 Ivory Coast (46%) 1-2 ┘                                            │
                                                                    ├─ Argentina (54%) 1-1 ┐
 Argentina (89%) 2-1 ──┐                                            │ (P)                  │
                       ├─ Argentina (67%) 2-1┐                      │                      │
 Ecuador (11%) 1-2 ────┘                     │                      │                      │
                                             ├─ Argentina (64%) 2-1 ┘                      │
 Netherlands (86%) 2-1 ┐                     │                                             │
                       ├─ Netherlands (33%)1-2                                             │
 South Korea (14%) 1-2 ┘                                                                   │
                                                                                           ├─ Argentina (35%) 1-2 ┘
 United States (82%)2-1┐                                                                   
                       ├─ U.S. (16%) 1-2 ────┐                                             
 Bosnia (18%) 1-2 ─────┘                     │                                             
                                             ├─ France (57%) 1-1 ──┐                       
 France (81%) 2-1 ─────┐                     │                     │                       
                       ├─ France (84%) 2-1 ──┘                     │                       
 Sweden (19%) 1-2 ─────┘                                           │                       
                                                                   ├─ France (46%) 1-1 ────┘
 Switzerland (42%) 1-1 ┐                                           (P)
                       ├─ Portugal (46%) 1-1 ┐
 Portugal (58%) 1-1 ───┘                     │
                                             ├─ Mexico (43%) 1-1 ──┘
 Mexico (67%) 2-1 ─────┐                     │
                       ├─ Mexico (54%) 1-1 ──┘
 Croatia (33%) 1-2 ────┘
```

*(P) denotes advancement on penalties after simulated draw scorelines.*
