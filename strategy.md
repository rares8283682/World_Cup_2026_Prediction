# World Cup 2026 Prediction Strategy and Methodological Paper

This document details the machine learning strategy, pipeline architectures, model definitions and limitations across all notebooks in the World Cup 2026 Prediction project.

---

## 1. Project Overview and Methodological Approach

Football match outcomes are notoriously difficult to predict due to high variance, low-scoring environments and the non-additive nature of team performance. This project addresses these challenges by combining historical international match data, FiveThirtyEight Soccer Power Index (SPI) statistics, Transfermarkt squad market valuations and real-time betting market implied probabilities.

Our core methodology divides the prediction space into two tasks:
1. **Match Outcome (1x2 Classification)**: Predicting the probability of a home win, draw or away win.
2. **Scoreline Regression (Goal Models)**: Predicting the exact number of goals scored by each team to generate realistic score distributions.

---

## 2. Guide to the Project Notebooks

### 1. `wc_2026.ipynb`
* **Purpose**: Core exploratory data analysis (EDA) and baseline feature engineering.
* **Methods**: Cleaning historical international match tables, mapping team names across disparate data sources and aligning FIFA rankings.
* **Why**: Establishes the primary format for the chronological dataset, ensuring that matches are ordered by date to prevent future data leakage.

### 2. `wc_2026_advanced_data_integration.ipynb`
* **Purpose**: Feature enrichment using advanced metrics.
* **Methods**: Merging rolling expected goals (xG) statistics, defensive ratings from FiveThirtyEight and market valuations of national squads from Transfermarkt.
* **Why**: Market values act as strong proxies for individual player quality, while xG captures team performance quality beyond simple scores.

### 3. `wc_2026_model_training.ipynb`
* **Purpose**: Sandbox for initial model training and tuning.
* **Methods**: Testing baseline classifiers (Logistic Regression, Random Forest and Gradient Boosting) and tuning hyperparameters using time-series cross validation splits.
* **Why**: Standard random cross validation is inappropriate for football match predictions because it creates chronological data leakage. We use rolling time-series splits to validate models.

### 4. `final_model_improved_WC_2026_group_predi.ipynb`
* **Purpose**: Optimized model for winner prediction with market odds adjustments.
* **Methods**: Employs a Random Forest Classifier trained on symmetric double-inference inputs. Implements a betting value edge override (adjusting predictions if there is a massive discrepancy between model probabilities and normalized bookmaker odds).
* **Why**: Betting markets consolidate vast amounts of public information. The model acts as a corrector to bookmaker odds, identifying value edges where the bookmaker underestimates draws or underdogs.

### 5. `final_model_improved_WC_2026_group_predi+score_predi.ipynb`
* **Purpose**: Goal modeling and scoreline sampling.
* **Methods**: Trains separate HistGradientBoostingRegressor models for home goals and away goals. The predicted expected goals (λ_home, λ_away) are used to draw scorelines from independent Poisson distributions. It filters draws to find a scoreline matching the predicted winner.
* **Why**: Independent Poisson distributions match football goal frequencies closely but can occasionally predict scorelines inconsistent with the overall match winner (e.g. predicting a draw scoreline for a home win outcome). The sampling loop resolves this.

### 6. `WC_2026_complete_modeling_pipeline.ipynb`
* **Purpose**: The final production tournament simulation pipeline.
* **Methods**:
  * **Winner Expert Ensemble**: Blends Stacked Logistic Regression, CatBoost Classifier, XGBoost Classifier and Team Embedding Neural Network outputs.
  * **Goal Models**: Blends a PoissonRegressor and an XGBRegressor (with poisson objective) to predict scores.
  * **Simulation Engine**: Automatically computes group standings, determines wildcard qualifiers and dynamically builds synthetic feature rows for knockout matches using historical averages.
* **Why**: This is the single source of truth for the entire World Cup simulation, handling all merge steps, standings calculations and brackets.

---

## 3. Modeling Architecture Deep Dive

### Mixture of Experts (MoE) Outcome Model
We combine four distinct classifiers to create a robust prediction ensemble:
1. **CatBoost (45% Weight)**: Highly robust to categorical features and non-linear relationships.
2. **XGBoost DART (20% Weight)**: Uses dropout trees to prevent overfitting to recent match results.
3. **Stacked Logistic Regression (20% Weight)**: Acts as a linear baseline, blending bookmaker probabilities and GNN outputs.
4. **GNN Team Embeddings (15% Weight)**: Extracts latent structural features of match history graphs.

### Scoreline Prediction Model
Expected goals are generated using:
$$\text{Expected Goals} = 0.70 \times \text{PoissonRegressor}(\mathbf{X}) + 0.30 \times \text{XGBRegressor}(\mathbf{X})$$
This blended goal output combines the smooth linear properties of a Poisson Generalized Linear Model with the non-linear capability of XGBoost.

---

## 4. Key Limitations and Challenges

1. **Sample Size for 2026 Tournament**:
   The group stage consists of only 72 matches. Evaluation metrics on such a small validation set have high variance. Performance can change significantly based on a few unexpected upsets.
2. **Knockout Stage Synthetic Features**:
   Since knockout matchups are not scheduled in advance, the model must predict outcomes on synthetic feature rows built by averaging a team's last 5 matches. This averages out sudden team changes (such as injuries or tactical shifts).
3. **Betting Odds Dependency**:
   Several models rely heavily on normalized market implied probabilities. If odds data is unavailable or highly illiquid for certain international friendlies, the model's baseline predictions can shift.
4. **Draw Bias**:
   Machine learning classifiers trained on cross-entropy loss tend to under-predict draws because draws are high-entropy, low-probability outcomes. While draw weights and boosts were implemented, finding the correct threshold remains a balancing act.
5. **No Live Lineup Data**:
   The current models rely on historical rolling statistics rather than live matchday lineups. Sudden benchings of star players or tactical rotations are not captured.
