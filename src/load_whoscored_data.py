import numpy
import pandas

def load_merge_data():
    bundesliga_2018_2019_off = pandas.read_csv("data/bundesliga_2018_2019_off.csv")
    bundesliga_2018_2019_off["league"] = "bundesliga"
    premier_league_2018_2019_off = pandas.read_csv("data/premier_league_2018_2019_off.csv")
    premier_league_2018_2019_off["league"] = "premier_league"
    ligue1_2018_2019_off = pandas.read_csv("data/ligue1_2018_2019_off.csv")
    ligue1_2018_2019_off["league"] = "ligue1"
    serie_a_2018_2019_off = pandas.read_csv("data/serie_a_2018_2019_off.csv")
    serie_a_2018_2019_off["league"] = "serie_a"
    la_liga_2018_2019_off = pandas.read_csv("data/la_liga_2018_2019_off.csv")
    la_liga_2018_2019_off["league"] = "la_liga"

    bundesliga_2018_2019_def = pandas.read_csv("data/bundesliga_2018_2019_def.csv")
    bundesliga_2018_2019_def["league"] = "bundesliga"
    premier_league_2018_2019_def = pandas.read_csv("data/premier_league_2018_2019_def.csv")
    premier_league_2018_2019_def["league"] = "premier_league"
    ligue1_2018_2019_def = pandas.read_csv("data/ligue1_2018_2019_def.csv")
    ligue1_2018_2019_def["league"] = "ligue1"
    serie_a_2018_2019_def = pandas.read_csv("data/serie_a_2018_2019_def.csv")
    serie_a_2018_2019_def["league"] = "serie_a"
    la_liga_2018_2019_def = pandas.read_csv("data/la_liga_2018_2019_def.csv")
    la_liga_2018_2019_def["league"] = "la_liga"

    full_2018_2019_off = pandas.concat([bundesliga_2018_2019_off, premier_league_2018_2019_off, ligue1_2018_2019_off, serie_a_2018_2019_off, la_liga_2018_2019_off], axis=0)
    full_2018_2019_def = pandas.concat([bundesliga_2018_2019_def, premier_league_2018_2019_def, ligue1_2018_2019_def, serie_a_2018_2019_def, la_liga_2018_2019_def], axis=0)

    full_2018_2019 = full_2018_2019_off.merge(full_2018_2019_def, how="outer", on="player").dropna(axis=0)
    return full_2018_2019

def clean_apparition(data):
    try:
        return int(data[:data.index("(")])
    except:
        return int(data)
    
def clean_float(data):
    try:
        if "-" in data:
            return 0.0
        else:
            return float(data)
    except:
        return data

def clean_text(data):
    return data.replace(",", "")

def clean_data(data):

    data["apparitions"] = data["apps_y"].apply(clean_apparition)
    data["minutes"] = data["minutes_y"]
    data["goals"] = data["goals"].apply(clean_float)
    data["assists"] = data["assists"].apply(clean_float)
    data["shot_per_game"] = data["shot_per_game"].apply(clean_float)
    data["key_passes_per_game"] = data["key_passes_per_game"].apply(clean_float)
    data["dribbles_per_game"] = data["dribbles_per_game"].apply(clean_float)
    data["fouled_per_game"] = data["fouled_per_game"].apply(clean_float)
    data["offside_per_game"] = data["offside_per_game"].apply(clean_float)
    data["dispossessed_per_game"] = data["dispossessed_per_game"].apply(clean_float)
    data["bad_control_per_game"] = data["bad_control_per_game"].apply(clean_float)
    data["bad_control_per_game"] = data["bad_control_per_game"].apply(clean_float)
    data["team"] = data["team_y"].apply(clean_text)
    data["age"] = data["age_y"].apply(clean_float)
    data["tackles_per_game"] = data["tackles_per_game"].apply(clean_float)
    data["interception_per_game"] = data["interception_per_game"].apply(clean_float)
    data["fouls_per_game"] = data["fouls_per_game"].apply(clean_float)
    data["offsides_per_game"] = data["offsides_per_game"].apply(clean_float)
    data["clear_per_game"] = data["clear_per_game"].apply(clean_float)
    data["dribbled_past_per_game"] = data["dribbled_past_per_game"].apply(clean_float)
    data["blocks_per_game"] = data["blocks_per_game"].apply(clean_float)
    data["own_goal"] = data["own_goal"].apply(clean_float)
    data["league"] = data["league_y"]    
    columns = ["player", "team", "league", "age", "apparitions", "minutes", "goals", "assists", "shot_per_game", "key_passes_per_game", "dribbles_per_game", "fouled_per_game", "offside_per_game", "dispossessed_per_game", "bad_control_per_game", "tackles_per_game", "interception_per_game", "fouls_per_game", "offsides_per_game", "clear_per_game", "dribbled_past_per_game", "blocks_per_game", "own_goal"]
    return data[columns]

if __name__ == "__main__":
    data = load_merge_data()
    data = clean_data(data)
    data.to_csv("data/full_2018_2019.csv", index=False)


