import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

#1: Load Data
team_map = pd.read_csv("team_map.csv")
print(f"Loaded TEAM × MAP: {team_map.shape}")

team_map["Result"] = team_map["Result"].apply(lambda x: 1 if x == 1 else 0)
team_map = team_map.sort_values(["team", "match_id", "round_num"])

team_map["match_id"] = pd.to_numeric(team_map["match_id"], errors="coerce").fillna(0).astype(int)

if "match_date" in team_map.columns:
    team_map["match_date"] = pd.to_datetime(team_map["match_date"], errors="coerce")
    team_map["days"] = (team_map["match_date"] - team_map["match_date"].min()).dt.days.fillna(0).astype(int)
else:
    team_map["days"] = team_map["match_id"] - team_map["match_id"].min()

#2: Feature Engineering
team_map["rolling_wr"] = (
    team_map.groupby("team")["Result"]
    .transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
)
team_map["team_past_wr"] = (
    team_map.groupby("team")["Result"]
    .transform(lambda x: x.shift().expanding().mean())
)

opp_wr = (
    team_map.groupby(["match_id", "round_num", "team"])["team_past_wr"].mean().reset_index()
)
opp_wr = opp_wr.rename(columns={"team": "opp_team", "team_past_wr": "opp_wr"})
team_map = team_map.merge(opp_wr, on=["match_id", "round_num"])
team_map = team_map[team_map["team"] != team_map["opp_team"]]

if "hero_bans" in team_map.columns:
    team_map["ban_count"] = team_map["hero_bans"].fillna("").apply(
        lambda x: len(x.split(",")) if isinstance(x, str) and x else 0
    )
else:
    team_map["ban_count"] = 0

team_map["kd_ratio"] = team_map["Eliminations"] / team_map["Deaths"].replace(0, 1)
team_map["dmg_eff"] = team_map["Damage Dealt"] / (team_map["Damage Mitigated"] + 1)
team_map["heal_eff"] = team_map["Healing Done"] / team_map["Deaths"].replace(0, 1)

team_map["schedule_strength"] = team_map.groupby("team")["opp_wr"].transform("mean")

team_map = pd.get_dummies(team_map, columns=["map_type"], prefix="map_type")

for col in ["rolling_wr", "team_past_wr", "opp_wr", "schedule_strength"]:
    team_map[col] = team_map[col].fillna(0.5)

#3: Features
feature_cols = [
    "Eliminations","Assists","Final Blows","Deaths",
    "Damage Dealt","Damage Mitigated","Healing Done","Objective Time",
    "rolling_wr","team_past_wr","opp_wr","ban_count",
    "kd_ratio","dmg_eff","heal_eff","schedule_strength"
] + [c for c in team_map.columns if c.startswith("map_type_")]

X = team_map[feature_cols].fillna(0)
y = team_map["Result"]
print("Using features:", feature_cols)

#4: Train/Test Split
if "stage" in team_map.columns:
    train_data = team_map[team_map["stage"].isin(["S1", "S2"])]
    test_data = team_map[team_map["stage"] == "S3"]
else:
    cutoff = int(team_map["match_id"].quantile(0.7))
    train_data = team_map[team_map["match_id"] <= cutoff]
    test_data = team_map[team_map["match_id"] > cutoff]

X_train, y_train = train_data[feature_cols], train_data["Result"]
X_test, y_test = test_data[feature_cols], test_data["Result"]

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

#5: Models
logreg = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", CalibratedClassifierCV(
        estimator=LogisticRegression(max_iter=2000, solver="lbfgs"),
        cv=5
    ))
])
logreg.fit(X_train, y_train)

print("\nLogistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, logreg.predict(X_test)))
print(classification_report(y_test, logreg.predict(X_test)))

rf = CalibratedClassifierCV(
    estimator=RandomForestClassifier(
        n_estimators=300, max_depth=6, random_state=42, class_weight="balanced"),
    cv=5
)
rf.fit(X_train, y_train)

print("\nRandom Forest Results:")
print("Accuracy:", accuracy_score(y_test, rf.predict(X_test)))
print(classification_report(y_test, rf.predict(X_test)))

#6: Elo Rating System (date-aware, playoff boost, dynamic K, clean dicts)
def initialize_elo(team_map, base_rating=1500, k=32, half_life_days=60):
    elo, elo_map = {}, {}
    games_played = {}
    max_days = team_map["days"].max()

    for _, row in team_map.iterrows():
        t, o, res, d = row["team"], row["opp_team"], row["Result"], row["days"]

        # detect map type
        map_type = [c.replace("map_type_", "") for c in row.index if c.startswith("map_type_") and row[c] == 1]
        map_type = map_type[0] if map_type else "General"

        for team in [t, o]:
            if team not in elo:
                elo[team] = base_rating
                games_played[team] = 0
            if team not in elo_map:
                elo_map[team] = {}
            if map_type not in elo_map[team]:
                elo_map[team][map_type] = base_rating

        # recency weighting
        age_days = max_days - d
        decay = 0.5 ** (age_days / half_life_days)

        # playoff weighting
        phase_text = str(row.get("phase", "")).lower()
        phase_boost = 2.5 if ("playoff" in phase_text or "final" in phase_text) else 1.0

        # dynamic K-factor
        games_played[t] += 1
        games_played[o] += 1
        k_dynamic = k / np.sqrt(games_played[t])
        if phase_boost > 1:
            k_dynamic *= 2

        # global Elo update
        exp_t = 1 / (1 + 10 ** ((elo[o] - elo[t]) / 400))
        elo[t] += k_dynamic * decay * phase_boost * (res - exp_t)
        elo[o] += k_dynamic * decay * phase_boost * ((1 - res) - (1 - exp_t))

        # map-specific Elo update
        exp_t_map = 1 / (1 + 10 ** ((elo_map[o][map_type] - elo_map[t][map_type]) / 400))
        elo_map[t][map_type] += k_dynamic * decay * phase_boost * (res - exp_t_map)
        elo_map[o][map_type] += k_dynamic * decay * phase_boost * ((1 - res) - (1 - exp_t_map))

    return elo, elo_map, games_played

elo_ratings, elo_map_ratings, games_played = initialize_elo(train_data)

def elo_probability(team1, team2, map_type=None):
    if map_type and team1 in elo_map_ratings and map_type in elo_map_ratings[team1]:
        r1 = elo_map_ratings[team1][map_type]
        r2 = elo_map_ratings.get(team2, {}).get(map_type, 1500)
    else:
        r1, r2 = elo_ratings.get(team1, 1500), elo_ratings.get(team2, 1500)
    return 1 / (1 + 10 ** ((r2 - r1) / 400))

print("\nCurrent Elo Ratings (global top 15):")
for team, rating in sorted(elo_ratings.items(), key=lambda x: -x[1])[:15]:
    print(f"{team}: {rating:.0f}")


#7: Simulation
def simulate_match(team1, team2, model=rf, last_n=5):
    def team_profile(team):
        df = team_map[team_map["team"] == team].sort_values("match_date", ascending=False)
        if len(df) > last_n: df = df.head(last_n)
        return df[feature_cols].mean()

    t1_stats, t2_stats = team_profile(team1), team_profile(team2)
    t1_df, t2_df = pd.DataFrame([t1_stats]), pd.DataFrame([t2_stats])

    # ML prediction
    p1_ml, p2_ml = model.predict_proba(t1_df)[0][1], model.predict_proba(t2_df)[0][1]
    # Elo prediction
    p1_elo = elo_probability(team1, team2)
    p2_elo = 1 - p1_elo

    # Blend weight by team strength
    avg_rating = (elo_ratings.get(team1,1500) + elo_ratings.get(team2,1500)) / 2
    blend = 0.7 if avg_rating > 1550 else 0.5 if avg_rating > 1450 else 0.3

    p1 = blend*p1_ml + (1-blend)*p1_elo
    p2 = blend*p2_ml + (1-blend)*p2_elo
    total = p1+p2
    p1, p2 = p1/total, p2/total

    print(f"\n⚔️ {team1} vs {team2}")
    print(f"➡️ {team1} win probability: {p1:.2f}")
    print(f"➡️ {team2} win probability: {p2:.2f}")

    print("\n📊 Per Map Type Predictions (map-specific Elo):")
    for mt in [c.replace("map_type_","") for c in feature_cols if c.startswith("map_type_")]:
        pm1_elo = elo_probability(team1, team2, map_type=mt)
        pm2_elo = 1 - pm1_elo
        pm1_ml = model.predict_proba(t1_df.assign(**{f"map_type_{mt}":1}))[0][1]
        pm2_ml = model.predict_proba(t2_df.assign(**{f"map_type_{mt}":1}))[0][1]
        pm1 = blend*pm1_ml + (1-blend)*pm1_elo
        pm2 = blend*pm2_ml + (1-blend)*pm2_elo
        total = pm1+pm2
        pm1, pm2 = pm1/total, pm2/total
        favored = team1 if pm1 > pm2 else team2
        print(f"   {mt}: {team1} {pm1:.2f} | {team2} {pm2:.2f} -> Favored: {favored}")

#8: Example Sims
simulate_match("Twisted Minds", "Al qadsiah")
simulate_match("NTMR", "Geekay Esports")
simulate_match("Team Liquid", "NTMR")
simulate_match("Geekay Esports", "Team Liquid")
simulate_match("Twisted Minds", "Virtuspro")
simulate_match("Twisted Minds", "Quick Esports")
simulate_match("Al qadsiah", "Virtuspro")
simulate_match("Al qadsiah", "Quick Esports")
