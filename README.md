# 🌍 FIFA World Cup 2026 Prediction Pipeline

A state-of-the-art machine learning framework for predicting FIFA World Cup 2026 match outcomes, scorelines, group standings and tournament brackets.

---

## Modeling Architecture

### 1. Match Outcome Prediction (Mixture of Experts)
To capture diverse signal spaces in football dynamics, we implement a **Mixture of Experts (MoE)** ensemble model:
* **Stacked Logistic Regression Classifier (20% Weight)**: Combines team embedding signals with full odds baseline probabilities, optimized with Time-Series Cross Validation and recency-weighted sample decay.
* **CatBoost Classifier (45% Weight)**: Optimized challenger using raw feature sets, calibrated using Isotonic Regression.
* **eXtreme Gradient Boosting (XGBoost) DART Classifier (20% Weight)**: Calibrated DART challenger leveraging advanced performance metrics.
* **Graph Neural Network (GNN) Team Embeddings Model (15% Weight)**: Deep learning representation learning model extracting latent team interactions.

We use **Symmetric Double-Inference** to run predictions twice per match (swapping home and away inputs) and average probabilities, completely eliminating listing-order bias on neutral venues.

---

## ⚽ Expected Goals (Poisson) Score Predictions
Instead of simple classifications, we predict exact goal scorelines:
* **Poisson Regressor Model**: Trained a Poisson Generalized Linear Model on historical matches to output expected goal values ($\lambda_{home}$, $\lambda_{away}$) for any matchup.
* **eXtreme Gradient Boosting (XGBoost) Poisson Regressor Model**: Trained an XGBoost model using the `count:poisson` objective to predict goal bounds.
* **Blended Score Generation**: Goal predictions are computed as a blended model ($70\%$ Poisson Regressor + $30\%$ XGBoost Poisson) and rounded to the nearest integer.
* **Dynamic ELO Adjustments**: To override statistical noise in synthetic matchup rows, we dynamically scale advancement probabilities using ELO gaps (applying strong favorite overrides for ELO differences $\ge 50$ and $\ge 150$).

---

## 🏆 The World Cup 2026 Bracket

**Models Used**: Knockout matchups are predicted using the **Mixture of Experts Ensemble Model** (consisting of the **CatBoost Classifier**, **eXtreme Gradient Boosting Classifier**, **Stacked Logistic Regression Classifier** and **Graph Neural Network Team Embeddings Model**) with dynamic ELO rating adjustments. Goal scorelines are predicted using the blended **Poisson Regressor Model** and **eXtreme Gradient Boosting Poisson Regressor Model**.

Below is the complete predicted bracket from the Round of 32 to the World Cup Final:

```
 ROUND OF 32                ROUND OF 16              QUARTER-FINALS             SEMI-FINALS                  FINAL                 CHAMPION
 
 🇩🇪 Germany (94%) 2-1 ────┐
                           ├─ 🇩🇪 Germany (88%) 1-1 ─┐
 🇬🇭 Ghana (6%) 1-2 ────────┘                        │
                                                    ├─ 🇩🇪 Germany (54%) 2-1 ───┐
 🇲🇦 Morocco (51%) 2-1 ────┐                         │                          │
                           ├─ 🇲🇦 Morocco (12%) 1-1 ─┘                          │
 🇮🇷 Iran (49%) 1-2 ────────┘                                                     │
                                                                               ├─ 🇩🇪 Germany (49%) 1-2 ──┐
 🇪🇸 Spain (93%) 2-1 ──────┐                                                     │                         │
                           ├─ 🇪🇸 Spain (84%) 2-1 ───┐                          │                         │
 🇪🇬 Egypt (7%) 1-2 ────────┘                        │                          │                         │
                                                    ├─ 🇪🇸 Spain (46%) 1-2 ─────┘                         │
 🇦🇹 Austria (40%) 2-1 ────┐                         │                                                    │
                           ├─ 🇦🇺 Australia (16%) 1-2┘                                                    │
 🇦🇺 Australia (60%) 1-2 ──┘                                                                              │
                                                                                                         ├─ 🇧🇷 Brazil (65%) 2-1 ── [ 🏆 BRAZIL ]
 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England (95%) 2-1 ────┐                                                                              │
                           ├─ 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England (84%) 1-1 ─┐                                                  │
 🇿🇦 South Africa (5%) 1-2 ┘                          │                                                  │
                                                    ├─ 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England (45%) 1-2 ───┐                      │
 🇳🇴 Norway (55%) 2-1 ─────┐                          │                            │                      │
                           ├─ 🇳🇴 Norway (16%) 1-1 ──┘                            │                      │
 🇯🇵 Japan (46%) 1-2 ──────┘                                                      │                      │
                                                                                 ├─ 🇧🇷 Brazil (51%) 2-1 ┘
 🇨🇦 Canada (47%) 1-2 ─────┐                                                      │
                           ├─ 🇵🇾 Paraguay (8%) 1-2 ─┐                            │
 🇵🇾 Paraguay (53%) 2-1 ───┘                         │                            │
                                                    ├─ 🇧🇷 Brazil (55%) 2-1 ─────┘
 🇨🇴 Colombia (18%) 1-2 ───┐                         │
                           ├─ 🇧🇷 Brazil (92%) 2-1 ──┘
 🇧🇷 Brazil (82%) 2-1 ─────┘

 🇧🇪 Belgium (67%) 2-1 ────┐
                           ├─ 🇧🇪 Belgium (68%) 2-1 ─┐
 🇺🇾 Uruguay (33%) 1-2 ────┘                        │
                                                    ├─ 🇧🇪 Belgium (36%) 1-2 ───┐
 🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scotland (54%) 2-1 ───┐                         │                          │
                           ├─ 🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scotland (32%) 1-2 ┘                          │
 🇨🇮 Ivory Coast (46%) 1-2 ┘                                                     │
                                                                               ├─ 🇦🇷 Argentina (54%) 1-1 ┐
 🇦🇷 Argentina (89%) 2-1 ──┐                                                     │ (P)                     │
                           ├─ 🇦🇷 Argentina (67%) 2-1┐                          │                         │
 🇪🇨 Ecuador (11%) 1-2 ────┘                         │                          │                         │
                                                    ├─ 🇦🇷 Argentina (64%) 2-1 ─┘                         │
 🇳🇱 Netherlands (86%) 2-1 ┐                         │                                                    │
                           ├─ 🇳🇱 Netherlands (33%)1-2                                                    │
 🇰🇷 South Korea (14%) 1-2 ┘                                                                              │
                                                                                                         ├─ 🇦🇷 Argentina (35%) 1-2┘
 🇺🇸 United States (82%)2-1┐                                                                   
                           ├─ 🇺🇸 U.S. (16%) 1-2 ────┐                                             
 🇧🇦 Bosnia (18%) 1-2 ─────┘                         │                                             
                                                    ├─ 🇫🇷 France (57%) 1-1 ────┐                       
 🇫🇷 France (81%) 2-1 ─────┐                         │                          │                       
                           ├─ 🇫🇷 France (84%) 2-1 ──┘                          │                       
 🇸🇪 Sweden (19%) 1-2 ─────┘                                                    │                       
                                                                               ├─ 🇫🇷 France (46%) 1-1 ──┘
 🇨🇭 Switzerland (42%) 1-1 ┐                                                    (P)
                           ├─ 🇵🇹 Portugal (46%) 1-1 ┐
 🇵🇹 Portugal (58%) 1-1 ───┘                         │
                                                    ├─ 🇲🇽 Mexico (43%) 1-1 ────┘
 🇲🇽 Mexico (67%) 2-1 ─────┐                         │
                           ├─ 🇲🇽 Mexico (54%) 1-1 ──┘
 🇭🇷 Croatia (33%) 1-2 ────┘
```

*(P) denotes advancement on penalties after simulated draw scorelines.*

---

## Group Stage Scoreboard

**Models Used**: Match outcomes are predicted using the **Mixture of Experts Ensemble Model** (consisting of the **CatBoost Classifier**, **eXtreme Gradient Boosting Classifier**, **Stacked Logistic Regression Classifier** and **Graph Neural Network Team Embeddings Model**). Goals and scores are predicted using the blended **Poisson Regressor Model** and **eXtreme Gradient Boosting Poisson Regressor Model**.

Below is the complete scoreboard with predicted/actual scores for the group stage matches:

### Group A
* 🇦🇷 Argentina **3 - 0** 🇩🇿 Algeria
* 🇦🇹 Austria **3 - 1** 🇯🇴 Jordan
* 🇯🇴 Jordan **1 - 1** 🇩🇿 Algeria
* 🇦🇷 Argentina **2 - 1** 🇦🇹 Austria
* 🇯🇴 Jordan **1 - 1** 🇦🇷 Argentina
* 🇩🇿 Algeria **2 - 1** 🇦🇹 Austria

### Group B
* 🇺🇸 United States **4 - 1** 🇵🇾 Paraguay
* 🇦🇺 Australia **2 - 0** 🇹🇷 Turkey
* 🇹🇷 Turkey **0 - 1** 🇵🇾 Paraguay
* 🇺🇸 United States **2 - 0** 🇦🇺 Australia
* 🇹🇷 Turkey **0 - 1** 🇵🇾 Paraguay
* 🇺🇸 United States **2 - 1** 🇹🇷 Turkey

### Group C
* 🇮🇷 Iran **2 - 2** 🇳🇿 New Zealand
* 🇧🇪 Belgium **1 - 1** 🇪🇬 Egypt
* 🇳🇿 New Zealand **1 - 1** 🇪🇬 Egypt
* 🇧🇪 Belgium **2 - 1** 🇮🇷 Iran
* 🇳🇿 New Zealand **1 - 1** 🇧🇪 Belgium
* 🇪🇬 Egypt **1 - 1** 🇮🇷 Iran

### Group D
* 🇨🇦 Canada **1 - 1** 🇧🇦 Bosnia and Herzegovina
* 🇶🇦 Qatar **1 - 1** 🇨🇭 Switzerland
* 🇨🇭 Switzerland **4 - 1** 🇧🇦 Bosnia and Herzegovina
* 🇨🇦 Canada **6 - 0** 🇶🇦 Qatar
* 🇨🇦 Canada **2 - 1** 🇨🇭 Switzerland
* 🇧🇦 Bosnia and Herzegovina **2 - 1** 🇶🇦 Qatar

### Group E
* 🇧🇷 Brazil **1 - 1** 🇲🇦 Morocco
* 🇭🇹 Haiti **0 - 8** 🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scotland
* 🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scotland **0 - 1** 🇲🇦 Morocco
* 🇧🇷 Brazil **3 - 0** 🇭🇹 Haiti
* 🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scotland **1 - 1** 🇧🇷 Brazil
* 🇲🇦 Morocco **2 - 1** 🇭🇹 Haiti

### Group F
* 🇸🇦 Saudi Arabia **1 - 1** 🇺🇾 Uruguay
* 🇪🇸 Spain **0 - 0** 🇨🇻 Cape Verde
* 🇪🇸 Spain **2 - 1** 🇸🇦 Saudi Arabia
* 🇺🇾 Uruguay **2 - 1** 🇨🇻 Cape Verde
* 🇺🇾 Uruguay **1 - 1** 🇪🇸 Spain
* 🇨🇻 Cape Verde **2 - 1** 🇸🇦 Saudi Arabia

### Group G
* 🇵🇹 Portugal **1 - 1** 🇨🇩 DR Congo
* 🇺🇿 Uzbekistan **1 - 3** 🇨🇴 Colombia
* 🇨🇴 Colombia **2 - 1** 🇨🇩 DR Congo
* 🇨🇴 Colombia **1 - 1** 🇵🇹 Portugal
* 🇨🇩 DR Congo **2 - 1** 🇺🇿 Uzbekistan
* 🇵🇹 Portugal **2 - 1** 🇺🇿 Uzbekistan

### Group H
* 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England **4 - 2** 🇭🇷 Croatia
* 🇬🇭 Ghana **1 - 0** 🇵🇦 Panama
* 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England **2 - 1** 🇬🇭 Ghana
* 🇵🇦 Panama **1 - 1** 🇭🇷 Croatia
* 🇭🇷 Croatia **2 - 1** 🇬🇭 Ghana
* 🇵🇦 Panama **1 - 1** 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England

### Group I
* 🇩🇪 Germany **7 - 1** 🇨🇼 Curaçao
* 🇨🇮 Ivory Coast **1 - 0** 🇪🇨 Ecuador
* 🇩🇪 Germany **2 - 1** 🇨🇮 Ivory Coast
* 🇪🇨 Ecuador **2 - 1** 🇨🇼 Curaçao
* 🇪🇨 Ecuador **2 - 1** 🇩🇪 Germany
* 🇨🇼 Curaçao **1 - 1** 🇨🇮 Ivory Coast

### Group J
* 🇲🇽 Mexico **2 - 0** 🇿🇦 South Africa
* 🇰🇷 South Korea **2 - 1** 🇨🇿 Czech Republic
* 🇨🇿 Czech Republic **1 - 1** 🇿🇦 South Africa
* 🇲🇽 Mexico **1 - 0** 🇰🇷 South Korea
* 🇿🇦 South Africa **1 - 1** 🇰🇷 South Korea
* 🇲🇽 Mexico **2 - 1** 🇨🇿 Czech Republic

### Group K
* 🇫🇷 France **3 - 1** 🇸🇳 Senegal
* 🇮🇶 Iraq **1 - 4** 🇳🇴 Norway
* 🇳🇴 Norway **2 - 1** 🇸🇳 Senegal
* 🇫🇷 France **2 - 1** 🇮🇶 Iraq
* 🇳🇴 Norway **2 - 1** 🇫🇷 France
* 🇸🇳 Senegal **2 - 1** 🇮🇶 Iraq

### Group L
* 🇳🇱 Netherlands **2 - 2** 🇯🇵 Japan
* 🇸🇪 Sweden **5 - 1** 🇹🇳 Tunisia
* 🇹🇳 Tunisia **1 - 1** 🇯🇵 Japan
* 🇳🇱 Netherlands **2 - 1** 🇸🇪 Sweden
* 🇹🇳 Tunisia **1 - 1** 🇳🇱 Netherlands
* 🇯🇵 Japan **2 - 1** 🇸🇪 Sweden
