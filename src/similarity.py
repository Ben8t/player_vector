"""similarity.py
Define functions to compute similarity between players
"""
import numpy
import pandas
import scipy.spatial


class SimilarEngine:
    """Wrap similarity computation

    Args:
        data (pandas.DataFrame): full player database.
        vector_keys (list): key to define features that characterize a player.
        name_column (str): define player index (usually the 'name').

    """
    def __init__(self, data, vector_keys, name_column):
        self.__data = data
        self.__vector_keys = vector_keys
        self.__name_column = name_column

    def get_player_vector(self, player):
        """ Return player vector

        Args:
            player (str): a player name
        
        Returns:
            list: player vector
        """
        return self.__data[self.__data[self.__name_column]==player][self.__vector_keys].get_values()

    def find_similar_players(self, player, distance_callback=scipy.spatial.distance.cosine, n=2):
        """ Compute similarity for a given player

        Args:
            player (str): player name or vector.
            distance_callback (function): a distance function.
            n (int): number of similar players to return (ordered by similarity).
        
        Returns:
            pandas.DataFrame: similar players
        """
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
        """ Compute gradient similarity.
        Given two players, compute a 'gradient' vector representing a shaded or mix of the tow players.

        Args:
            player1 (str): a player name
            player2 (str): a player name
            alpha (float): degree of gradient (more it's close to 0 more the gradient vector will be close to the first player).

        Returns:
            list: interpolated vector
        """
        player1_vector = self.get_player_vector(player1)
        player2_vector = self.get_player_vector(player2)
        interpolated_vector = alpha * player1_vector + (1 - alpha) * player2_vector
        return interpolated_vector

    def interpolated_players(self, player1, player2, alpha_range=10):
        """Compute gradient for two players with different values of alpha

        Args:
            player1 (str): a player name.
            player2 (str): a player name.
            alpha_range (int): number of different value to compute on.

        Returns:
            pandas.DataFrame: players
        """
        df = self.__data.drop(self.__data[self.__data[self.__name_column] == player1].index)
        df = df.drop(df[df[self.__name_column] == player2].index)
        alphas = numpy.linspace(1, 0, alpha_range, endpoint=False)
        players = []
        for a in alphas:
            vector = self.gradient_embedding(player1, player2, a)
            player = self.find_similar_players(vector, n=10).iloc[[1]]
            players.append(player[self.__name_column].get_values())
        return pandas.DataFrame(players)
