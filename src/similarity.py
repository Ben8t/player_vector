import numpy
import pandas
import scipy.spatial


class SimilarEngine:

    def __init__(self, data, vector_keys, name_column):
        self.__data = data
        self.__vector_keys = vector_keys
        self.__name_column = name_column

    def get_player_vector(self, player):
        return self.__data[self.__data[self.__name_column]==player][self.__vector_keys].get_values()

    def find_similar_players(self, player, distance_callback=scipy.spatial.distance.cosine, n=2):
        data = self.__data.copy()
        if type(player)==str:
            main_player_vector = self.get_player_vector(player)
        else:
            main_player_vector = player
        for index, row in data.iterrows():
            player_vector = row[self.__vector_keys].get_values()
            player_vector = [float(x) for x in player_vector]
            distance = distance_callback(main_player_vector, player_vector)
            data.at[index, "distance"] = distance
        return data.sort_values("distance", ascending=True).head(n)


    def gradient_embedding(self, player1, player2, alpha):
        player1_vector = self.get_player_vector(player1)
        player2_vector = self.get_player_vector(player2)
        interpolated_vector = alpha * player1_vector + (1 - alpha) * player2_vector
        return interpolated_vector

    def interpolated_players(self, player1, player2, alpha_range=10):
        df = self.__data.drop(self.__data[self.__data[name_column] == player1].index)
        df = df.drop(df[df[name_column] == player2].index)
        alphas = numpy.linspace(1, 0, alpha_range, endpoint=False)
        players = []
        for a in alphas:
            vector = self.gradient_embedding(player1, player2, a)
            player = self.find_similar_players(vector, n=10).iloc[[1]]
            players.append(player[name_column].get_values())
        return pandas.DataFrame(players)
