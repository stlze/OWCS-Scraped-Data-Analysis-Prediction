import requests
import pandas as pd
from datetime import datetime

API_KEY = "API_KEY" # from faceit
MATCH_IDS = [
    "1-d1c1f469-bef7-42cb-bce0-9d64706b427a",  # example https://www.faceit.com/en/ow2/room/1-c360b538-f56a-4ed3-b63d-b8ba6d2162bf
    # add more match IDs here
]

headers = {"Authorization": f"Bearer {API_KEY}"}
all_rows = []

for mid in MATCH_IDS:
    print(f"Processing {mid}")

    m = requests.get(f"https://open.faceit.com/data/v4/matches/{mid}", headers=headers).json()
    match_date = m.get("started_at")
    if match_date:
        match_date = datetime.utcfromtimestamp(match_date).isoformat()

    map_entities = {e["guid"]: e["name"] for e in m.get("voting", {}).get("map", {}).get("entities", [])}
    map_picks = m.get("voting", {}).get("map", {}).get("pick", [])
    maps_ordered = [map_entities.get(g, g) for g in map_picks]

    hero_entities = {e["guid"]: e["name"] for e in m.get("voting", {}).get("heroes", {}).get("entities", [])}
    ban_sets = []
    for game_bans in m.get("voting", {}).get("heroes", {}).get("pick", []):
        names = []
        if isinstance(game_bans, list):
            for guid in game_bans:
                names.append(hero_entities.get(guid, guid))
        else:
            names.append(hero_entities.get(game_bans, game_bans))
        ban_sets.append(", ".join(names))

    s = requests.get(f"https://open.faceit.com/data/v4/matches/{mid}/stats", headers=headers).json()

    if not s.get("rounds"):
        for team in m.get("teams", []):
            team_name = team.get("name") or team.get("team_id")
            all_rows.append({
                "match_id": mid,
                "match_date": match_date,
                "round_num": None,
                "map_name": None,
                "hero_bans": None,
                "team": team_name,
                "player": None,
                "Result": None,
                "data_quality": "metadata_only"
            })
        continue

    for r_i, rnd in enumerate(s.get("rounds", []), start=1):
        raw_guid = rnd.get("round_stats", {}).get("Map", "")
        map_name = map_entities.get(raw_guid, "") or (maps_ordered[r_i-1] if r_i-1 < len(maps_ordered) else "")
        hero_bans = ban_sets[r_i-1] if r_i-1 < len(ban_sets) else ""

        for team in rnd.get("teams", []):
            team_name = team.get("team_stats", {}).get("Team") or team.get("team_id")
            result = 1 if team.get("team_stats", {}).get("Team Win", "0") == "1" else 0

            if not team.get("players"):
                all_rows.append({
                    "match_id": mid,
                    "match_date": match_date,
                    "round_num": r_i,
                    "map_name": map_name,
                    "hero_bans": hero_bans,
                    "team": team_name,
                    "player": None,
                    "Result": result,
                    "data_quality": "metadata_only"
                })
                continue

            for p in team.get("players", []):
                row = {
                    "match_id": mid,
                    "match_date": match_date,
                    "round_num": r_i,
                    "map_name": map_name,
                    "hero_bans": hero_bans,
                    "team": team_name,
                    "player": p.get("nickname"),
                    "Result": result,
                    "data_quality": "full"
                }
                row.update(p.get("player_stats", {})) 
                all_rows.append(row)

df = pd.DataFrame(all_rows)
df = df.loc[:, ~df.columns.duplicated()]
df.to_csv("faceit_matches_stage_.csv", index=False)

print(f"Saved {len(df)} rows across {len(MATCH_IDS)} matches -> faceit_matches_stage_.csv")
