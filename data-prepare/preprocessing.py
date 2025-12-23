import random

import numpy as np
import pandas as pd


class TMDBPreProcessor:
    def __init__(self, movies: list, user_count=100, max_select_count=20):
        random.seed(0)
        self._movies = movies
        self._features = pd.DataFrame()
        self._users = list(range(1, user_count+1))
        self._max_select_count = max_select_count
        self._max_runtime_seconds = 120 * 60

    @staticmethod
    def augmentation(movie):
        rating = movie["vote_average"]
        count = int(pow(2, rating))
        data = {
            "content_id": movie["id"],
            "rating": rating,
            "popularity": movie["popularity"]
        }
        return [data] * count

    def generate_watch_second(self, rating):
        base = 1.1
        noise_level = 0.1
        base_time = self._max_runtime_seconds \
		        * (base ** (rating - 5) - base ** -5) \
		        / (base ** 5 - base ** -5)
        noise = np.random.normal(0, noise_level * base_time)
        watch_second = base_time + noise

        watch_second = int(np.clip(watch_second, 0, self._max_runtime_seconds))
        print(f"{rating}/{watch_second}")
        return watch_second

    def selection(self, user_id, features):
        select_count = random.randint(0, self._max_select_count)
        print(f"user [{user_id}] is select [{select_count}] contents")
        if select_count == 0:
            return []

        selected_feature = random.choices(features, k=select_count)

        result = [
            {
                "user_id": str(user_id),
                "content_id": str(feature["content_id"]),
                "watch_seconds": self.generate_watch_second(feature["rating"]),
                "rating": feature["rating"],
                "popularity": feature["popularity"],
            } for feature in selected_feature
        ]
        return result

    def run(self):
        features = []
        selected_features = []
        for movie in self._movies:
            features.extend(self.augmentation(movie))

        for user_id in self._users:
            selected_features.extend(self.selection(user_id, features))

        df = pd.DataFrame.from_records(selected_features)

        self._features = df

    def save(self, filename):
        if not self._features.empty:
            self._features.to_csv(f"./result/{filename}.csv", header=True, index=False)

    @property
    def features(self):
        return self._features


