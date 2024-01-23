import json
from collections import defaultdict
from pathlib import Path
from bentoml.client import Client
from loguru import logger
import random

BENTOML_URL = "http://0.0.0.0:8995"
IMG_PATH = "./img_examples"
BATCH_SIZE = 10

def call(json_data):
    client = Client.from_url(BENTOML_URL)
    return client.process_batch(folder_name=IMG_PATH, results_container=json_data)

if __name__ == "__main__":
    results_container = defaultdict(lambda: {"idx": None, "name": None, "text": None, "file": None})
    # Check if all images recived right

    images = list(Path(IMG_PATH).glob("*.png"))
    logger.debug(images)

    # Create batch
    for idx in range(0,BATCH_SIZE):
        results_container[idx]["idx"] = idx
        results_container[idx]["file"] = images[random.choice([0,1])].name

    # Convert defaultdict to JSON
    json_data = json.dumps(dict(results_container), indent=2)


    logger.info(call(json_data))
