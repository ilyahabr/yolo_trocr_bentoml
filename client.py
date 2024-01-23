import json
from collections import defaultdict
from pathlib import Path
from utils.utils import timing_decorator
from bentoml.client import Client
from loguru import logger


BENTOML_URL = "http://0.0.0.0:8996"
IMG_PATH = "./image_example"

@timing_decorator
def call(json_data):
    client = Client.from_url(BENTOML_URL)
    return client.process_batch(folder_name=IMG_PATH, results_container=json_data)

if __name__ == "__main__":
    results_container = defaultdict(lambda: {"idx": None, "name": None, "price": None, "file": None})
    # Check if all images recived right

    images = list(Path(IMG_PATH).glob("*.jpg"))

    for idx in range(0,30):
        if images[0].exists():
            results_container[idx]["idx"] = idx
        results_container[idx]["file"] = images[0].name

    # Convert defaultdict to JSON
    json_data = json.dumps(dict(results_container), indent=2)

    logger.info(call(json_data))
