import pandas as pd

master = pd.read_csv("faceit_all_matches_emea_na_all_stages.csv")
print(f"Loaded {len(master)} rows")
print("Regions before fix:", master["region"].unique())
if master["region"].isna().any():
    master["region"] = master["region"].fillna(
        master["match_id"].str.split("_").str[0]
    )

print("Regions after fix:", master["region"].unique())
print("Sample columns:", master.columns[:15].tolist())

id_cols = [
    "region", "stage", "phase", "match_id", "match_date",
    "round_num", "map_name", "map_type", "team", "player", "hero_bans"
]
stat_cols = [c for c in master.columns if c not in id_cols]

print(f"Detected {len(stat_cols)} stat columns")

#team_match
team_match = (
    master.groupby(["region", "stage", "phase", "match_id", "match_date", "team"])[stat_cols]
    .sum()
    .reset_index()
)

if "Result" in master.columns:
    result = (
        master.groupby(["region", "stage", "phase", "match_id", "match_date", "team"])["Result"]
        .max()
        .reset_index()
    )
    team_match = (
        team_match.drop(columns="Result", errors="ignore")
        .merge(result, on=["region", "stage", "phase", "match_id", "match_date", "team"])
    )

team_match.to_csv("team_match.csv", index=False)
print(f"Saved team_match -> {len(team_match)} rows")

#player_match
player_match = (
    master.groupby(["region", "stage", "phase", "match_id", "match_date", "team", "player"])[stat_cols]
    .sum()
    .reset_index()
)

if "Result" in master.columns:
    result = (
        master.groupby(["region", "stage", "phase", "match_id", "match_date", "team", "player"])["Result"]
        .max()
        .reset_index()
    )
    player_match = (
        player_match.drop(columns="Result", errors="ignore")
        .merge(result, on=["region", "stage", "phase", "match_id", "match_date", "team", "player"])
    )

player_match.to_csv("player_match.csv", index=False)
print(f"Saved player_match -> {len(player_match)} rows")

#team_map
team_map = (
    master.groupby([
        "region", "stage", "phase", "match_id", "match_date", "round_num", "map_name", "map_type", "team"
    ])[stat_cols]
    .sum()
    .reset_index()
)
if "hero_bans" in master.columns:
    bans = (
        master.groupby([
            "region", "stage", "phase", "match_id", "match_date", "round_num", "map_name", "map_type", "team"
        ])["hero_bans"]
        .apply(lambda x: ", ".join(sorted(set(x.dropna().astype(str)))))
        .reset_index()
    )
    team_map = team_map.merge(
        bans,
        on=["region", "stage", "phase", "match_id", "match_date", "round_num", "map_name", "map_type", "team"]
    )

if "Result" in master.columns:
    result = (
        master.groupby([
            "region", "stage", "phase", "match_id", "match_date", "round_num", "map_name", "map_type", "team"
        ])["Result"]
        .max()
        .reset_index()
    )
    team_map = (
        team_map.drop(columns="Result", errors="ignore")
        .merge(result, on=["region", "stage", "phase", "match_id", "match_date", "round_num", "map_name", "map_type", "team"])
    )

team_map.to_csv("team_map.csv", index=False)
print(f"Saved team_map -> {len(team_map)} rows")

print("All aggregations done successfully!")

print("\nRegion counts (team_match):")
print(team_match["region"].value_counts())
print("\nStages distribution (team_match):")
print(team_match["stage"].value_counts())
print("\nEarliest/LATEST dates:", team_match["match_date"].min(), "->", team_match["match_date"].max())
