import os
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error

# Define directories
datasets_dir = "/Users/raresolteanu/Desktop/WC-2026/datasets"
output_table_path = os.path.join(datasets_dir, "wc_2026_training_table.csv")

print("--- 1. LOADING DATASETS ---")
df_res = pd.read_csv(os.path.join(datasets_dir, "Historical international match results.csv"))
df_elo = pd.read_csv(os.path.join(datasets_dir, "Elo ratings.csv"))
df_fifa = pd.read_csv(os.path.join(datasets_dir, "FIFA rankings.csv"))
df_form = pd.read_csv(os.path.join(datasets_dir, "Recent form.csv"))
df_goals = pd.read_csv(os.path.join(datasets_dir, "Goals scored conceded.csv"))
df_home = pd.read_csv(os.path.join(datasets_dir, "Home region advantage.csv"))
df_rest = pd.read_csv(os.path.join(datasets_dir, "Rest days.csv"))
df_stage = pd.read_csv(os.path.join(datasets_dir, "Tournament stage.csv"))

print(f"Results: {len(df_res)} rows")
print(f"Elo ratings: {len(df_elo)} rows")
print(f"FIFA rankings: {len(df_fifa)} rows")
print(f"Recent form: {len(df_form)} rows")
print(f"Goals scored/conceded: {len(df_goals)} rows")
print(f"Home region advantage: {len(df_home)} rows")
print(f"Rest days: {len(df_rest)} rows")
print(f"Tournament stage: {len(df_stage)} rows")

# Convert date columns to datetime
for df in [df_res, df_elo, df_fifa, df_form, df_goals, df_home, df_rest, df_stage]:
    df['date'] = pd.to_datetime(df['date'])

# Sort chronologically
df_res = df_res.sort_values('date').reset_index(drop=True)
df_fifa = df_fifa.sort_values('date').reset_index(drop=True)

print("\n--- 2. DE-DUPLICATING DATASETS ---")
# Drop duplicates on the unique (date, team, opponent, is_home) combination
df_elo = df_elo.drop_duplicates(subset=['date', 'team', 'opponent', 'is_home'])
df_form = df_form.drop_duplicates(subset=['date', 'team', 'opponent', 'is_home'])
df_goals = df_goals.drop_duplicates(subset=['date', 'team', 'opponent', 'is_home'])
df_home = df_home.drop_duplicates(subset=['date', 'home_team', 'away_team'])
df_rest = df_rest.drop_duplicates(subset=['date', 'home_team', 'away_team'])
df_stage = df_stage.drop_duplicates(subset=['date', 'home_team', 'away_team'])

print(f"Cleaned Elo ratings: {len(df_elo)} rows")
print(f"Cleaned Recent form: {len(df_form)} rows")
print(f"Cleaned Goals scored/conceded: {len(df_goals)} rows")

print("\n--- 3. JOINING TEAM-LEVEL FEATURES ---")
# Helper columns for Elo join
df_res['home_elo_is_home'] = ~df_res['neutral']
df_res['away_elo_is_home'] = False

# Join Elo ratings
df_merged = pd.merge(
    df_res,
    df_elo[['date', 'team', 'opponent', 'is_home', 'elo_rating']].rename(columns={'elo_rating': 'home_elo'}),
    left_on=['date', 'home_team', 'away_team', 'home_elo_is_home'],
    right_on=['date', 'team', 'opponent', 'is_home'],
    how='left'
).drop(columns=['team', 'opponent', 'is_home'])

df_merged = pd.merge(
    df_merged,
    df_elo[['date', 'team', 'opponent', 'is_home', 'elo_rating']].rename(columns={'elo_rating': 'away_elo'}),
    left_on=['date', 'away_team', 'home_team', 'away_elo_is_home'],
    right_on=['date', 'team', 'opponent', 'is_home'],
    how='left'
).drop(columns=['team', 'opponent', 'is_home'])

df_merged = df_merged.drop(columns=['home_elo_is_home', 'away_elo_is_home'])

# Join Recent Form
df_form_home = df_form[df_form['is_home'] == True].rename(columns={
    'form_score_5': 'home_form_score_5', 'form_score_10': 'home_form_score_10'
})
df_form_away = df_form[df_form['is_home'] == False].rename(columns={
    'form_score_5': 'away_form_score_5', 'form_score_10': 'away_form_score_10'
})

df_merged = pd.merge(
    df_merged,
    df_form_home[['date', 'team', 'opponent', 'home_form_score_5', 'home_form_score_10']],
    left_on=['date', 'home_team', 'away_team'],
    right_on=['date', 'team', 'opponent'],
    how='left'
).drop(columns=['team', 'opponent'])

df_merged = pd.merge(
    df_merged,
    df_form_away[['date', 'team', 'opponent', 'away_form_score_5', 'away_form_score_10']],
    left_on=['date', 'away_team', 'home_team'],
    right_on=['date', 'team', 'opponent'],
    how='left'
).drop(columns=['team', 'opponent'])

# Join Goals Scored/Conceded
df_goals_home = df_goals[df_goals['is_home'] == True].rename(columns={
    'average_goals_scored': 'home_avg_goals_scored', 'average_goals_conceded': 'home_avg_goals_conceded'
})
df_goals_away = df_goals[df_goals['is_home'] == False].rename(columns={
    'average_goals_scored': 'away_avg_goals_scored', 'average_goals_conceded': 'away_avg_goals_conceded'
})

df_merged = pd.merge(
    df_merged,
    df_goals_home[['date', 'team', 'opponent', 'home_avg_goals_scored', 'home_avg_goals_conceded']],
    left_on=['date', 'home_team', 'away_team'],
    right_on=['date', 'team', 'opponent'],
    how='left'
).drop(columns=['team', 'opponent'])

df_merged = pd.merge(
    df_merged,
    df_goals_away[['date', 'team', 'opponent', 'away_avg_goals_scored', 'away_avg_goals_conceded']],
    left_on=['date', 'away_team', 'home_team'],
    right_on=['date', 'team', 'opponent'],
    how='left'
).drop(columns=['team', 'opponent'])

print("\n--- 4. JOINING FIFA RANKINGS (pd.merge_asof) ---")
# Map team names to FIFA spelling where applicable
fifa_name_mapping = {
    'China': 'China PR', 'Taiwan': 'Chinese Taipei', 'North Korea': 'Korea DPR', 'Kyrgyzstan': 'Kyrgyz Republic',
    'Brunei': 'Brunei Darussalam', 'Saint Kitts and Nevis': 'St. Kitts and Nevis', 'Saint Lucia': 'St. Lucia',
    'Saint Vincent and the Grenadines': 'St. Vincent and the Grenadines', 'United States Virgin Islands': 'US Virgin Islands'
}
df_merged['fifa_home_name'] = df_merged['home_team'].replace(fifa_name_mapping)
df_merged['fifa_away_name'] = df_merged['away_team'].replace(fifa_name_mapping)

df_fifa_home = df_fifa.rename(columns={'team': 'fifa_home_name', 'rank': 'home_fifa_rank', 'total_points': 'home_fifa_points'}).drop(columns=['team_short'])
df_fifa_away = df_fifa.rename(columns={'team': 'fifa_away_name', 'rank': 'away_fifa_rank', 'total_points': 'away_fifa_points'}).drop(columns=['team_short'])

df_merged = pd.merge_asof(
    df_merged,
    df_fifa_home,
    on='date',
    by='fifa_home_name',
    direction='backward'
)

df_merged = pd.merge_asof(
    df_merged,
    df_fifa_away,
    on='date',
    by='fifa_away_name',
    direction='backward'
)

df_merged = df_merged.drop(columns=['fifa_home_name', 'fifa_away_name'])

print("\n--- 5. JOINING MATCH-LEVEL DATASETS ---")
# Join Home region advantage
df_merged = pd.merge(
    df_merged,
    df_home[['date', 'home_team', 'away_team', 'is_home_country', 'is_away_country', 'is_home_confed', 'is_away_confed']],
    on=['date', 'home_team', 'away_team'],
    how='left'
)

# Join Rest days
df_merged = pd.merge(
    df_merged,
    df_rest[['date', 'home_team', 'away_team', 'home_rest_days_capped', 'away_rest_days_capped']],
    on=['date', 'home_team', 'away_team'],
    how='left'
)

# Join Tournament stage
df_merged = pd.merge(
    df_merged,
    df_stage[['date', 'home_team', 'away_team', 'stage']],
    on=['date', 'home_team', 'away_team'],
    how='left'
)

print("\n--- 6. IMPUTING MISSING VALUES ---")
# Fill tournament stage
df_merged['stage'] = df_merged['stage'].fillna('Not World Cup')

# Fill missing FIFA rankings (mostly for pre-1993 or non-FIFA members)
df_merged['home_fifa_rank'] = df_merged['home_fifa_rank'].fillna(215)
df_merged['away_fifa_rank'] = df_merged['away_fifa_rank'].fillna(215)
df_merged['home_fifa_points'] = df_merged['home_fifa_points'].fillna(0.0)
df_merged['away_fifa_points'] = df_merged['away_fifa_points'].fillna(0.0)

# Fill other potential missing values in Home region advantage or Rest days (fallbacks)
df_merged['is_home_country'] = df_merged['is_home_country'].fillna(False)
df_merged['is_away_country'] = df_merged['is_away_country'].fillna(False)
df_merged['is_home_confed'] = df_merged['is_home_confed'].fillna(False)
df_merged['is_away_confed'] = df_merged['is_away_confed'].fillna(False)
df_merged['home_rest_days_capped'] = df_merged['home_rest_days_capped'].fillna(7.0)
df_merged['away_rest_days_capped'] = df_merged['away_rest_days_capped'].fillna(7.0)

print("\n--- 7. FEATURE ENGINEERING ---")
# 1. Target Outcomes (only for rows that have scores)
has_scores = df_merged['home_score'].notnull() & df_merged['away_score'].notnull()

df_merged['result'] = None
df_merged.loc[has_scores & (df_merged['home_score'] > df_merged['away_score']), 'result'] = 'home_win'
df_merged.loc[has_scores & (df_merged['home_score'] == df_merged['away_score']), 'result'] = 'draw'
df_merged.loc[has_scores & (df_merged['home_score'] < df_merged['away_score']), 'result'] = 'away_win'

df_merged['goal_diff'] = np.nan
df_merged.loc[has_scores, 'goal_diff'] = df_merged['home_score'] - df_merged['away_score']

# 2. Difference Features
df_merged['elo_diff'] = df_merged['home_elo'] - df_merged['away_elo']
df_merged['fifa_rank_diff'] = df_merged['home_fifa_rank'] - df_merged['away_fifa_rank']
df_merged['fifa_points_diff'] = df_merged['home_fifa_points'] - df_merged['away_fifa_points']
df_merged['form_score_5_diff'] = df_merged['home_form_score_5'] - df_merged['away_form_score_5']
df_merged['form_score_10_diff'] = df_merged['home_form_score_10'] - df_merged['away_form_score_10']
df_merged['rest_days_diff'] = df_merged['home_rest_days_capped'] - df_merged['away_rest_days_capped']
df_merged['attack_diff'] = df_merged['home_avg_goals_scored'] - df_merged['away_avg_goals_scored']
df_merged['defense_diff'] = df_merged['home_avg_goals_conceded'] - df_merged['away_avg_goals_conceded']

# Save final table
df_merged.to_csv(output_table_path, index=False)
print(f"Saved clean training table to {output_table_path} (shape: {df_merged.shape})")

print("\n--- 8. MODEL TRAINING ---")
# We train models on the modern era (since 1993) where FIFA rankings are established
df_train_full = df_merged[(df_merged['date'] >= '1993-01-01') & df_merged['result'].notnull()].copy()
print(f"Total rows available for training: {len(df_train_full)}")

features = [
    'elo_diff', 'fifa_rank_diff', 'fifa_points_diff',
    'form_score_5_diff', 'form_score_10_diff',
    'attack_diff', 'defense_diff',
    'is_home_country', 'is_away_country', 'is_home_confed', 'is_away_confed',
    'rest_days_diff', 'neutral'
]

# Classification target: result (home_win, draw, away_win)
X = df_train_full[features].copy()
# Convert boolean columns to int
for col in ['is_home_country', 'is_away_country', 'is_home_confed', 'is_away_confed', 'neutral']:
    X[col] = X[col].astype(int)

y_class = df_train_full['result']
y_home_goals = df_train_full['home_score'].astype(int)
y_away_goals = df_train_full['away_score'].astype(int)

# Split data
X_train, X_test, y_class_train, y_class_test, y_hg_train, y_hg_test, y_ag_train, y_ag_test = train_test_split(
    X, y_class, y_home_goals, y_away_goals, test_size=0.2, random_state=42
)

# Scale continuous features
scaler = StandardScaler()
continuous_cols = [
    'elo_diff', 'fifa_rank_diff', 'fifa_points_diff',
    'form_score_5_diff', 'form_score_10_diff',
    'attack_diff', 'defense_diff', 'rest_days_diff'
]

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])

# 1. Classification Model (Logistic Regression)
clf_lr = LogisticRegression(max_iter=1000, random_state=42)
clf_lr.fit(X_train_scaled, y_class_train)
class_preds = clf_lr.predict(X_test_scaled)
class_probs = clf_lr.predict_proba(X_test_scaled)

acc = accuracy_score(y_class_test, class_preds)
loss = log_loss(y_class_test, class_probs)

print("\nClassification Model (Logistic Regression) Metrics:")
print(f"  Accuracy: {acc:.4f}")
print(f"  Log Loss: {loss:.4f}")

# 2. Score Prediction Models (Poisson Regression)
poisson_home = PoissonRegressor()
poisson_home.fit(X_train_scaled, y_hg_train)
pred_hg = poisson_home.predict(X_test_scaled)

poisson_away = PoissonRegressor()
poisson_away.fit(X_train_scaled, y_ag_train)
pred_ag = poisson_away.predict(X_test_scaled)

mae_home = mean_absolute_error(y_hg_test, pred_hg)
mae_away = mean_absolute_error(y_ag_test, pred_ag)

print("\nScore Prediction Model (Poisson Regression) Metrics:")
print(f"  Home Goals MAE: {mae_home:.4f}")
print(f"  Away Goals MAE: {mae_away:.4f}")

# 3. Random Forest Classifier as alternative
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
clf_rf.fit(X_train_scaled, y_class_train)
rf_preds = clf_rf.predict(X_test_scaled)
rf_probs = clf_rf.predict_proba(X_test_scaled)
rf_acc = accuracy_score(y_class_test, rf_preds)
rf_loss = log_loss(y_class_test, rf_probs)

print("\nClassification Model (Random Forest Classifier) Metrics:")
print(f"  Accuracy: {rf_acc:.4f}")
print(f"  Log Loss: {rf_loss:.4f}")

print("\n--- MODEL TRAINING AND EVALUATION COMPLETED ---")
