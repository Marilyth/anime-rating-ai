import matplotlib.pyplot as plt
import pandas

random_seed = 1

def load_data(max_samples=100000) -> pandas.DataFrame:
    data = pandas.read_csv("anime_data.csv", escapechar="\\").dropna()
    if len(data) > max_samples:
        data = data.sample(max_samples)
    return data

def rating_distribution():
    """Displays the distribution of MyAnimeList ratings.

    Args:
        data (List[str]): A list of strings to be displayed.
    """
    data = load_data()
    data.hist("Score")
    plt.tight_layout()
    plt.show()
