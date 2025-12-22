import os
import json
import time

import requests


class TMDBCrawler:
    def __init__(
		    self, region="KR", 
		    language="ko-KR", 
		    request_interval_seconds=0.4
		):
        self._base_url = os.environ.get("TMDB_BASE_URL")
        self._api_key = os.environ.get("TMDB_API_KEY")
        self._region = region
        self._language = language
        self._request_interval_seconds = request_interval_seconds

    def get_popular_movies(self, page):
        params = {
            "api_key": self._api_key,
            "language": self._language,
            "region": self._region,
            "page": page
        }
        response = requests.get(f"{self._base_url}/popular", params=params)

        if not response.status_code == 200:
            return

        return json.loads(response.text)["results"]

    def get_bulk_popular_movies(self, start_page, end_page):
        movies = []

        for page in range(start_page, end_page+1):
            movies.extend(self.get_popular_movies(page))
            time.sleep(self._request_interval_seconds)

        return movies

    @staticmethod
    def save_movies_to_json_file(movies, dst="./result", filename="popular"):
        data = {"movies": movies}
        with open(f"{os.path.join(dst, filename)}.json", "w", encoding='utf-8') as f:
            f.write(json.dumps(data))
