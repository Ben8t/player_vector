import streamlit
import numpy
import pandas
import scipy.spatial
from sklearn import preprocessing
from sklearn.decomposition import PCA

"""

# Player Similarity

## Load and process data

Data come from [Fifa 20 player database](https://www.kaggle.com/stefanoleone992/fifa-20-complete-player-dataset#players_20.csv). While it's quite difficult
to gather a huge player database with features (mainly stats) to define players, Fifa player base can be a good starting point.

"""

data = pandas.read_csv("data/players_20.csv")

def process_data(data):
    features = ["weight_kg", "height_cm", "pace", "shooting","passing","dribbling","defending","physic","gk_diving","gk_handling","gk_kicking","gk_reflexes","gk_speed","gk_positioning","attacking_crossing","attacking_finishing","attacking_heading_accuracy","attacking_short_passing","attacking_volleys","skill_dribbling","skill_curve","skill_fk_accuracy","skill_long_passing","skill_ball_control","movement_acceleration","movement_sprint_speed","movement_agility","movement_reactions","movement_balance","power_shot_power","power_jumping","power_stamina","power_strength","power_long_shots","mentality_aggression","mentality_interceptions","mentality_positioning","mentality_vision","mentality_penalties","mentality_composure","defending_marking","defending_standing_tackle","defending_sliding_tackle","goalkeeping_diving","goalkeeping_handling","goalkeeping_kicking","goalkeeping_positioning","goalkeeping_reflexes"]
    data[features] = data[features].div(data.overall, axis=0)
    imputer = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    data[features] = imputer.fit_transform(data[features])
    data[features] = data[features].apply(preprocessing.scale)
    pca = PCA(n_components=2)
    pca.fit(data[features])
    data["component_1"], data["component_2"] = pca.transform(data[features])[:,0], pca.transform(data[features])[:,1]
    return data.copy()[["short_name", "player_positions"] + features] , pca.explained_variance_ratio_

player_vectors, pca_result = process_data(data)
streamlit.write(player_vectors.head())
player_vectors.to_csv("data/processed_data.csv", index=False)

streamlit.write(f"Applying PCA here is not that good (PCA explained variance ratio : `{pca_result}`). We could up the number of component, but only to reduce the overall data dimension (can't be used to 2D plot...).")

def get_player_vector(data, name):
    vector_keys = ["weight_kg", "height_cm", "pace", "shooting","passing","dribbling","defending","physic","gk_diving","gk_handling","gk_kicking","gk_reflexes","gk_speed","gk_positioning","attacking_crossing","attacking_finishing","attacking_heading_accuracy","attacking_short_passing","attacking_volleys","skill_dribbling","skill_curve","skill_fk_accuracy","skill_long_passing","skill_ball_control","movement_acceleration","movement_sprint_speed","movement_agility","movement_reactions","movement_balance","power_shot_power","power_jumping","power_stamina","power_strength","power_long_shots","mentality_aggression","mentality_interceptions","mentality_positioning","mentality_vision","mentality_penalties","mentality_composure","defending_marking","defending_standing_tackle","defending_sliding_tackle","goalkeeping_diving","goalkeeping_handling","goalkeeping_kicking","goalkeeping_positioning","goalkeeping_reflexes"]
    return data[data["short_name"]==name][vector_keys].get_values()

def find_similar_players(data, player, distance_callback=scipy.spatial.distance.cosine, n=2):
    vector_keys = ["weight_kg", "height_cm", "pace", "shooting","passing","dribbling","defending","physic","gk_diving","gk_handling","gk_kicking","gk_reflexes","gk_speed","gk_positioning","attacking_crossing","attacking_finishing","attacking_heading_accuracy","attacking_short_passing","attacking_volleys","skill_dribbling","skill_curve","skill_fk_accuracy","skill_long_passing","skill_ball_control","movement_acceleration","movement_sprint_speed","movement_agility","movement_reactions","movement_balance","power_shot_power","power_jumping","power_stamina","power_strength","power_long_shots","mentality_aggression","mentality_interceptions","mentality_positioning","mentality_vision","mentality_penalties","mentality_composure","defending_marking","defending_standing_tackle","defending_sliding_tackle","goalkeeping_diving","goalkeeping_handling","goalkeeping_kicking","goalkeeping_positioning","goalkeeping_reflexes"]
    if type(player)==str:
        main_player_vector = get_player_vector(data, player)
    else:
        main_player_vector = player
    for index, row in data.iterrows():
        player_vector = row[vector_keys].get_values()
        player_vector = [float(x) for x in player_vector]
        distance = distance_callback(main_player_vector, player_vector)
        data.at[index, "distance"] = distance
    return data.sort_values("distance", ascending = True).head(n)

def gradient_embedding(data, name1, name2, alpha):
    player1_vector = get_player_vector(data, name1)
    player2_vector = get_player_vector(data, name2)
    interpolated_vector = alpha * player1_vector + (1 - alpha) * player2_vector
    return interpolated_vector

def interpolated_players(data, name1, name2, alpha_range=10):
    df = data.drop(data[data.short_name == name1].index)
    df = df.drop(df[df.short_name == name2].index)
    alphas = numpy.linspace(1, 0, alpha_range, endpoint=False)
    print(alphas)
    players = []
    for a in alphas:
        vector = gradient_embedding(data, name1, name2, a)
        player = find_similar_players(df, vector, n=10).iloc[[1]]
        players.append(player.short_name.get_values())
        print(player.short_name.get_values()[0])
    return players

def render_latex(formula, fontsize=12, dpi=300):
    """Renders LaTeX formula into Streamlit."""
    fig = plt.figure()
    text = fig.text(0, 0, '$%s$' % formula, fontsize=fontsize)

    fig.savefig(BytesIO(), dpi=dpi)  # triggers rendering

    bbox = text.get_window_extent()
    width, height = bbox.size / float(dpi) + 0.05
    fig.set_size_inches((width, height))

    dy = (bbox.ymin / float(dpi)) / height
    text.set_position((0, -dy))

    buffer = BytesIO()
    fig.savefig(buffer, dpi=dpi, format='jpg')
    plt.close(fig)

    streamlit.image(buffer)


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
player = streamlit.selectbox("Players", player_vectors["short_name"].unique()[0:100], key="player")
streamlit.write(find_similar_players(player_vectors, player, scipy.spatial.distance.cosine, 10))

"""
### Gradient embedding
"""
 
player_1 = streamlit.selectbox("Players", player_vectors["short_name"].unique()[0:100], key="player_1")
player_2 = streamlit.selectbox("Players", player_vectors["short_name"].unique()[0:100], key="player_2")
gradient = streamlit.slider("Gradient rate", 0.0, 1.0, 0.1)
similar = gradient_embedding(player_vectors, player_1, player_2, gradient)
streamlit.write(find_similar_players(player_vectors, similar, scipy.spatial.distance.cosine, 10))

"""
### Interpolated Players
"""

streamlit.write(interpolated_players(player_vectors, "Santi Cazorla", "P. Pogba"))

"""
## Ressources

* https://towardsdatascience.com/how-to-write-web-apps-using-simple-python-for-data-scientists-a227a1a01582
* https://datascience.stackexchange.com/questions/27726/when-to-use-cosine-simlarity-over-euclidean-similarity 
"""
