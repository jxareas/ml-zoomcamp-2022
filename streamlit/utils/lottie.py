import json
import requests


def load_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


def load_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
