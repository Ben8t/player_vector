import streamlit
import numpy
import pandas
import scipy.spatial
from sklearn import preprocessing
from sklearn.decomposition import PCA

from src.similarity import SimilarEngine

"""

# Player Similarity

## Load and process data
"""

fifa_data = pandas.read_csv("data/players_20.csv")
whoscored_data = pandas.read_csv("data/full_2018_2019.csv")
football_manager_data = pandas.read_csv("data/football_manager_2017.csv")

@streamlit.cache(persist=True)
def process_fifa_data(data):
    features = ["weight_kg", "height_cm", "pace", "shooting","passing","dribbling","defending","physic","gk_diving","gk_handling","gk_kicking","gk_reflexes","gk_speed","gk_positioning","attacking_crossing","attacking_finishing","attacking_heading_accuracy","attacking_short_passing","attacking_volleys","skill_dribbling","skill_curve","skill_fk_accuracy","skill_long_passing","skill_ball_control","movement_acceleration","movement_sprint_speed","movement_agility","movement_reactions","movement_balance","power_shot_power","power_jumping","power_stamina","power_strength","power_long_shots","mentality_aggression","mentality_interceptions","mentality_positioning","mentality_vision","mentality_penalties","mentality_composure","defending_marking","defending_standing_tackle","defending_sliding_tackle","goalkeeping_diving","goalkeeping_handling","goalkeeping_kicking","goalkeeping_positioning","goalkeeping_reflexes"]
    processed_data = data.copy()
    processed_data[features] = processed_data[features].div(data.overall, axis=0)
    imputer = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    processed_data[features] = imputer.fit_transform(processed_data[features])
    processed_data[features] = processed_data[features].apply(preprocessing.scale)
    pca = PCA(n_components=2)
    pca.fit(processed_data[features])
    processed_data["component_1"], processed_data["component_2"] = pca.transform(processed_data[features])[:,0], pca.transform(processed_data[features])[:,1]
    return processed_data.copy()[["short_name", "player_positions"] + features] , pca.explained_variance_ratio_

@streamlit.cache(persist=True)
def process_whoscored_data(data):
    features = ["goals", "assists", "shot_per_game", "key_passes_per_game", "dribbles_per_game", "fouled_per_game", "offside_per_game", "dispossessed_per_game", "bad_control_per_game", "tackles_per_game", "interception_per_game", "fouls_per_game", "offsides_per_game", "clear_per_game", "dribbled_past_per_game", "blocks_per_game", "own_goal"]
    data = data.copy()
    data[features] = data[features].apply(preprocessing.scale)
    pca = PCA(n_components=2)
    pca.fit(data[features])
    data["component_1"], data["component_2"] = pca.transform(data[features])[:,0], pca.transform(data[features])[:,1]
    return data.copy()[["player", "team", "age", "league"] + features], pca.explained_variance_ratio_

@streamlit.cache(persist=True)
def process_football_manager_data(data):
    features = ["Height", "Weight", "AerialAbility", "CommandOfArea", "Communication", "Eccentricity", "Handling", "Kicking", "OneOnOnes", "Reflexes", "RushingOut", "TendencyToPunch", "Throwing", "Corners", "Crossing", "Dribbling", "Finishing", "FirstTouch", "Freekicks", "Heading", "LongShots", "Longthrows", "Marking", "Passing", "PenaltyTaking", "Tackling" ,"Technique", "Aggression", "Anticipation", "Bravery", "Composure", "Concentration", "Vision", "Decisions", "Determination", "Flair", "Leadership", "OffTheBall", "Positioning", "Teamwork", "Workrate", "Acceleration", "Agility", "Balance", "Jumping", "LeftFoot", "NaturalFitness", "Pace", "RightFoot", "Stamina", "Strength", "Consistency", "Dirtiness", "ImportantMatches", "InjuryProness", "Versatility", "Adaptability", "Ambition", "Loyalty", "Pressure", "Professional", "Sportsmanship", "Temperament", "Controversy"]
    processed_data = data.copy()
    processed_data = processed_data.query('IntCaps > 0')
    processed_data[features] = processed_data[features].apply(preprocessing.scale)
    pca = PCA(n_components=2)
    pca.fit(processed_data[features])
    processed_data["component_1"], processed_data["component_2"] = pca.transform(processed_data[features])[:,0], pca.transform(processed_data[features])[:,1]
    return processed_data.copy()[["Name", "Age"] + features], pca.explained_variance_ratio_

"""
Load data
"""
data_to_load = streamlit.radio("Data to load", ["Fifa", "WhoScored", "Football Manager"])
if data_to_load == "Fifa":
    player_vectors, pca_result = process_fifa_data(fifa_data)
    vector_keys = ["weight_kg", "height_cm", "pace", "shooting","passing","dribbling","defending","physic","gk_diving","gk_handling","gk_kicking","gk_reflexes","gk_speed", "gk_positioning","attacking_crossing","attacking_finishing","attacking_heading_accuracy","attacking_short_passing","attacking_volleys","skill_dribbling","skill_curve","skill_fk_accuracy","skill_long_passing","skill_ball_control","movement_acceleration","movement_sprint_speed","movement_agility","movement_reactions","movement_balance","power_shot_power","power_jumping","power_stamina","power_strength","power_long_shots","mentality_aggression","mentality_interceptions","mentality_positioning","mentality_vision","mentality_penalties","mentality_composure","defending_marking","defending_standing_tackle","defending_sliding_tackle","goalkeeping_diving","goalkeeping_handling","goalkeeping_kicking","goalkeeping_positioning","goalkeeping_reflexes"]
    name_column = "short_name"
elif data_to_load == "WhoScored":
    player_vectors, pca_result = process_whoscored_data(whoscored_data)
    vector_keys = ["goals", "assists", "shot_per_game", "key_passes_per_game", "dribbles_per_game", "fouled_per_game", "offside_per_game", "dispossessed_per_game", "bad_control_per_game", "tackles_per_game", "interception_per_game", "fouls_per_game", "offsides_per_game", "clear_per_game", "dribbled_past_per_game", "blocks_per_game", "own_goal"]
    name_column = "player"
else:
    player_vectors, pca_result = process_football_manager_data(football_manager_data)
    vector_keys = ["Height", "Weight", "AerialAbility", "CommandOfArea", "Communication", "Eccentricity", "Handling", "Kicking", "OneOnOnes", "Reflexes", "RushingOut", "TendencyToPunch", "Throwing", "Corners", "Crossing", "Dribbling", "Finishing", "FirstTouch", "Freekicks", "Heading", "LongShots", "Longthrows", "Marking", "Passing", "PenaltyTaking", "Tackling" ,"Technique", "Aggression", "Anticipation", "Bravery", "Composure", "Concentration", "Vision", "Decisions", "Determination", "Flair", "Leadership", "OffTheBall", "Positioning", "Teamwork", "Workrate", "Acceleration", "Agility", "Balance", "Jumping", "LeftFoot", "NaturalFitness", "Pace", "RightFoot", "Stamina", "Strength", "Consistency", "Dirtiness", "ImportantMatches", "InjuryProness", "Versatility", "Adaptability", "Ambition", "Loyalty", "Pressure", "Professional", "Sportsmanship", "Temperament", "Controversy"]
    name_column = "Name"

similar_engine = SimilarEngine(player_vectors, vector_keys, name_column)

streamlit.write(player_vectors.head())
player_vectors.to_csv("data/processed_data.csv", index=False)

streamlit.write(f"Applying PCA here is not that good (PCA explained variance ratio : `{pca_result}`). We could up the number of component, but only to reduce the overall data dimension (can't be used to 2D plot...).")



"""
## Test & Samples

In first tries, I applied euclidean distance to compute player similarities. However, it might be better to use cosine similarity, especially to capture similarity between young and more experimented players. For example :

* Player 1 has 100 pace, 100 shooting
* Player 2 has 50 pace, 50 shooting
* Player 3 has 90 pace, 40 shooting

By cosine similarity, player 1 and player 2 are more similar. By euclidean similarity, player 3 is more similar to player 1.
"""

player_1 = [100, 100]
player_2 = [50, 50]
player_3 = [90, 40]

print(scipy.spatial.distance.cosine(player_1, player_2))
print(scipy.spatial.distance.euclidean(player_1, player_2))

print(scipy.spatial.distance.cosine(player_1, player_3))
print(scipy.spatial.distance.euclidean(player_1, player_3))


"""
### Find similar player
"""
player = streamlit.selectbox("Players", player_vectors[name_column].unique(), key="player")
if streamlit.button("Compute similarity"):
    streamlit.write(similar_engine.find_similar_players(player, scipy.spatial.distance.cosine, 10))

"""
### Gradient embedding
"""
 
player_1 = streamlit.selectbox("Players", player_vectors[name_column].unique(), key="player_1")
player_2 = streamlit.selectbox("Players", player_vectors[name_column].unique(), key="player_2")
gradient = streamlit.slider("Gradient rate", 0.0, 1.0, 0.1)
similar = similar_engine.gradient_embedding(player_1, player_2, gradient)
if streamlit.button("Compute gradient"):
    streamlit.write(similar_engine.find_similar_players(similar, scipy.spatial.distance.cosine, 10, [player_1, player_2]))

"""
### Interpolated Players
"""

if streamlit.button("Compute interpolated players"):
    streamlit.write(similar_engine.interpolated_players(player_1, player_2))

"""
## Ressources

* https://towardsdatascience.com/how-to-write-web-apps-using-simple-python-for-data-scientists-a227a1a01582
* https://datascience.stackexchange.com/questions/27726/when-to-use-cosine-simlarity-over-euclidean-similarity 
"""
