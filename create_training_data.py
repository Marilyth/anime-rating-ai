from time import sleep
import requests
import csv

base_url = "https://api.jikan.moe/v4/"

def request_api(endpoint: str, query:list[tuple[str, str]]) -> dict:
    url = base_url + endpoint
    response = requests.get(url, params = query)
    return response.json()


def create_genre_flags() -> dict[int, int]:
    """
    Determines all existing genres of MyAnimeList and assigns each mal-id to a corresponding bit-flag.
    """
    genre_flags = {}
    
    genres = request_api("genres/anime", [("filter","genres")])["data"]
    genres_explicit = request_api("genres/anime", [("filter","explicit_genres")])["data"]
    genres.extend(genres_explicit)
    
    for i, genre in enumerate(sorted(genres, key=lambda x: x["mal_id"])):
        genre_flags[genre["mal_id"]] = 2**i
    
    return genre_flags


def get_data(type_name: str, genre_flags: dict[int,int], max_pages: int = -1) -> list[list[object]]:
    """
    Collects title, title_length, score and genres from the titles of the MyAnimeList Top list of the given type.
    """
    has_next_page = True
    page = 1
    entries = []

    while has_next_page and (max_pages == -1 or page <= max_pages):
        page_content = request_api("top/anime", [("type",type_name), ("page",f"{page}")])
        has_next_page = page_content["pagination"]["has_next_page"]
        page += 1

        for entry in page_content["data"]:
            title =  entry["title_english"] if entry["title_english"] else entry["title_japanese"] if entry["title_japanese"] else entry["title"]
            title_length = len(title)
            score = entry["score"]
            genres = 0
            for genre in entry["genres"]:
                genres += genre_flags[genre["mal_id"]]
            
            entries.append([title, title_length, genres, score])
        
        sleep(1)
    
    return entries


def write_data(file_name: str, types: list[str], max_pages: int = -1):
    """
    Creates a CSV file named file_name containing title, title_length, score, genres from the MyAnimeList Top list for the given types.
    """
    with open(file_name, "w+", newline="", encoding="utf-8") as file:
        writer = csv.writer(file,delimiter=",",escapechar="\\")
        genre_flags = create_genre_flags()
        writer.writerow(["Title", "TitleLength", "Genres", "Score"])

        for type_name in types:
            entries = get_data(type_name, genre_flags, max_pages)
            writer.writerows(entries)


if __name__ == "__main__":
    write_data("anime_data.csv", ["tv", "ona", "movie"], -1)