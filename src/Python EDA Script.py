import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

team_match = pd.read_csv("team_match.csv")
team_map = pd.read_csv("team_map.csv")
player_match = pd.read_csv("player_match.csv")

print("Datasets loaded:")
print(f"team_match: {team_match.shape}")
print(f"team_map: {team_map.shape}")
print(f"player_match: {player_match.shape}")
print("Regions before fix (team_match):", team_match["region"].unique())

#Fix missing region labels
for df in [team_match, team_map, player_match]:
    if df["region"].isna().any() or (df["region"] == "").any():
        df["region"] = df["region"].fillna(
            df["match_id"].str.split("_").str[0]
        )

print("Regions after fix (team_match):", team_match["region"].unique())
sns.set(style="whitegrid")

#3: TEAM WINRATES
team_wr = (
    team_match.groupby(["region", "stage", "team"])["Result"]
    .mean()
    .reset_index()
    .rename(columns={"Result": "Winrate"})
)

plt.figure(figsize=(14,6))
top_teams = team_wr.groupby("team")["Winrate"].mean().sort_values(ascending=False).head(12).index
sns.barplot(data=team_wr[team_wr["team"].isin(top_teams)], x="team", y="Winrate", hue="stage")
plt.xticks(rotation=45, ha="right")
plt.title("Top 12 Teams Winrates by Stage")
plt.ylabel("Winrate")
plt.tight_layout()
plt.show()

#2: MAP WINRATES
map_wr = (
    team_map.groupby(["map_type", "team"])["Result"]
    .mean()
    .reset_index()
    .rename(columns={"Result": "Winrate"})
)

plt.figure(figsize=(12,6))
sns.barplot(data=map_wr, x="map_type", y="Winrate", hue="team", dodge=False)
plt.xticks(rotation=45)
plt.title("Winrate by Map Type (All Teams)")
plt.ylabel("Winrate")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

#3: HERO BAN FREQUENCY + META EVOLUTION
if "hero_bans" in team_map.columns:
    all_bans = team_map["hero_bans"].fillna("").str.split(", ")
    all_bans = [ban.strip() for sublist in all_bans for ban in sublist if ban.strip()]
    ban_counts = pd.Series(all_bans).value_counts().head(15)

    plt.figure(figsize=(8,6))
    ban_counts.plot(kind="barh", color="red")
    plt.title("Most Common Hero Bans")
    plt.xlabel("Count")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    ban_stage = (
        team_map.dropna(subset=["hero_bans"])
        .assign(hero_bans=team_map["hero_bans"].str.split(", "))
        .explode("hero_bans")
        .groupby(["stage", "hero_bans"]).size()
        .reset_index(name="count")
    )
    top5_bans = ban_stage.groupby("hero_bans")["count"].sum().nlargest(5).index
    plt.figure(figsize=(10,6))
    sns.lineplot(data=ban_stage[ban_stage["hero_bans"].isin(top5_bans)], 
                 x="stage", y="count", hue="hero_bans", marker="o")
    plt.title("Top 5 Hero Bans Over Stages")
    plt.show()


#4: PLAYER PERFORMANCE LEADERBOARDS
metrics = ["Eliminations", "Damage Dealt", "Healing Done"]
for metric in metrics:
    top_players = (
        player_match.groupby("player")[metric]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    plt.figure(figsize=(10,6))
    top_players.plot(kind="barh")
    plt.title(f"Top 10 Players by Avg {metric}")
    plt.xlabel(metric)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

#5: TEAM vs TEAM HEATMAP
match_results = team_match.pivot_table(
    index="team", columns="match_id", values="Result", aggfunc="max"
).fillna(0)

plt.figure(figsize=(12,8))
sns.heatmap(match_results, cmap="Blues", cbar=False)
plt.title("Team Participation/Results Matrix (1=Win, 0=Loss)")
plt.show()

#6: REGION COMPARISON (K/D RATIO)
player_match["KDR"] = player_match["Eliminations"] / player_match["Deaths"].replace(0, 1)
plt.figure(figsize=(8,6))
sns.boxplot(data=player_match, x="region", y="KDR")
plt.ylim(0, player_match["KDR"].quantile(0.95))
plt.title("Player K/D Ratio Distribution by Region")
plt.show()

#7: TEAM PROGRESSION OVER STAGES
progression = team_wr.groupby(["team", "stage"])["Winrate"].mean().reset_index()
plt.figure(figsize=(12,6))
sns.lineplot(data=progression, x="stage", y="Winrate", hue="team", marker="o")
plt.title("Team Winrate Progression Across Stages")
plt.xticks(rotation=45)
plt.show()

#8: ROLE-BASED ANALYSIS (if Role exists)
if "Role" in player_match.columns:
    plt.figure(figsize=(10,6))
    sns.boxplot(data=player_match, x="Role", y="Eliminations")
    plt.title("Eliminations by Role")
    plt.show()

    plt.figure(figsize=(10,6))
    sns.boxplot(data=player_match, x="Role", y="Damage Dealt")
    plt.title("Damage by Role")
    plt.show()

    plt.figure(figsize=(10,6))
    sns.boxplot(data=player_match, x="Role", y="Healing Done")
    plt.title("Healing by Role")
    plt.show()
