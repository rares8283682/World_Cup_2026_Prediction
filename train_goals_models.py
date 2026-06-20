import json
import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor
from xgboost import XGBRegressor

with open("WC_2026_complete_modeling_pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

code_lines = []
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        code_lines.extend(cell["source"])

code_str = "".join(code_lines)

full_df = pd.read_csv("datasets/wc_2026_training_table_clean_with_spi_and_odds.csv")

xgb_cat_features = [
    "fifa_rank_diff", "fifa_points_zscore_diff", "form_score_5_diff", "form_score_10_diff",
    "home_form_trend", "away_form_trend", "form_trend_diff", "home_cumulative_matches",
    "home_avg_goals_scored", "home_avg_goals_conceded", "away_cumulative_matches",
    "away_avg_goals_scored", "away_avg_goals_conceded", "attack_diff", "defense_diff",
    "is_home_country", "is_home_confed", "is_away_confed", "home_rest_days_capped",
    "away_rest_days_capped", "rest_days_diff", "home_spi_prob", "away_spi_prob",
    "draw_spi_prob", "home_spi_proj_score", "away_spi_proj_score", "home_spi_importance",
    "away_spi_importance", "home_spi", "away_spi", "home_xg_for_rolling",
    "home_xg_against_rolling", "away_xg_for_rolling", "away_xg_against_rolling",
    "spi_diff", "spi_proj_score_diff", "spi_importance_diff", "spi_prob_diff",
    "xg_attack_diff", "xg_defense_diff", "xg_balance_diff", "home_overall_rating",
    "away_overall_rating", "strength_rating_diff", "home_attack_rating", "home_defense_rating",
    "away_attack_rating", "away_defense_rating", "attack_rating_diff", "defense_rating_diff",
    "home_total_unavailable", "home_missing_player_value", "away_total_unavailable",
    "away_missing_player_value", "unavailable_diff", "missing_player_value_diff",
    "h2h_matches_played", "h2h_wins_diff", "h2h_draws", "h2h_goals_diff", "h2h_avg_goals_diff",
    "home_win_streak", "home_unbeaten_streak", "away_win_streak", "away_unbeaten_streak",
    "win_streak_diff", "unbeaten_streak_diff", "home_gd", "away_gd", "is_draw",
    "head_to_head_draw_rate", "is_world_cup", "ranking_gap", "home_days_rest",
    "away_days_rest", "is_neutral_venue", "year", "group_stage_match_day",
    "recent_draw_rate_home", "recent_draw_rate_away", "goal_diff_last5",
    "squad_value_diff", "top_player_value_diff"
]

xgb_feature_cols = [c for c in xgb_cat_features if c in full_df.columns]

train_df = full_df[full_df["home_score"].notna() & full_df["away_score"].notna()].copy()
X = train_df[xgb_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
y_home = train_df["home_goals"] if "home_goals" in train_df.columns else train_df["home_score"]
y_away = train_df["away_goals"] if "away_goals" in train_df.columns else train_df["away_score"]
y_home = y_home.astype(int)
y_away = y_away.astype(int)

poisson_home = PoissonRegressor(max_iter=1000)
poisson_home.fit(X, y_home)
poisson_away = PoissonRegressor(max_iter=1000)
poisson_away.fit(X, y_away)

xgb_home = XGBRegressor(objective="count:poisson", random_state=42)
xgb_home.fit(X, y_home)
xgb_away = XGBRegressor(objective="count:poisson", random_state=42)
xgb_away.fit(X, y_away)

print("Training completed successfully.")

insertion_code = """from sklearn.linear_model import PoissonRegressor
from xgboost import XGBRegressor

train_df_goals = full_df[full_df["home_score"].notna() & full_df["away_score"].notna()].copy()
X_goals = train_df_goals[xgb_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
y_home_goals = train_df_goals["home_goals"] if "home_goals" in train_df_goals.columns else train_df_goals["home_score"]
y_away_goals = train_df_goals["away_goals"] if "away_goals" in train_df_goals.columns else train_df_goals["away_score"]
y_home_goals = y_home_goals.astype(int)
y_away_goals = y_away_goals.astype(int)

poisson_home_model = PoissonRegressor(max_iter=1000)
poisson_home_model.fit(X_goals, y_home_goals)
poisson_away_model = PoissonRegressor(max_iter=1000)
poisson_away_model.fit(X_goals, y_away_goals)

xgb_home_model = XGBRegressor(objective="count:poisson", random_state=42)
xgb_home_model.fit(X_goals, y_home_goals)
xgb_away_model = XGBRegressor(objective="count:poisson", random_state=42)
xgb_away_model.fit(X_goals, y_away_goals)"""

print("Code to be inserted:")
print(insertion_code)
