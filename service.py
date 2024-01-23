import json
import os
import typing as tp
from collections import defaultdict
from pathlib import Path

import bentoml
import torch
from bentoml.io import JSON, Multipart, Text
from clearml import Task
from loguru import logger
from model_wrappers import TrOCRWrapper, YoloDetWrapper, YoloSegWrapper


GPU_NUM = os.getenv("GPU_NUM", "0")
DEVICE = f"cuda:{GPU_NUM}" if torch.cuda.is_available() else "cpu"


class Yolov8RunnableSeg(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self, task_id: str, device: str = "cpu"):
        self.tag_segmentator = YoloSegWrapper(task_id=task_id, device=device)
        self.tag_segmentator.load_model()

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def inference(self, image_file_names):
        results = self.tag_segmentator.predict(image_file_names, save=False)
        return results


class Yolov8RunnableDet(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self, task_id: str, device: str = "cpu"):
        logger.debug(device)
        self.text_detector = YoloDetWrapper(task_id=task_id, device=device)
        self.text_detector.load_model()

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def inference(self, image_file_names):
        results = self.text_detector.predict(image_file_names, save=False)
        return results


class TrOCRRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self, task_id: str, device: str = "cpu"):
        self.char_recognizer = TrOCRWrapper(task_id=task_id, device=device)
        self.char_recognizer.load_model()

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def inference(self, images):
        results = self.char_recognizer.predict(images)
        return results


yolo_v8_runner_seg = bentoml.Runner(
    Yolov8RunnableSeg,
    max_batch_size=30,
    runnable_init_params={
        "task_id": SEG_TASK_ID,
        "device": DEVICE,
    },
)


yolo_v8_runner_det = bentoml.Runner(
    Yolov8RunnableDet,
    max_batch_size=30,
    runnable_init_params={
        "task_id": DET_TASK_ID,
        "device": DEVICE,
    },
)

TrOCR_runner = bentoml.Runner(
    TrOCRRunnable,
    max_batch_size=30,
    runnable_init_params={
        "task_id": OCR_TASK_ID,
        "device": DEVICE,
    },
)


svc = bentoml.Service(
    "pricetag_pipeline_service",
    runners=[yolo_v8_runner_seg, yolo_v8_runner_det, TrOCR_runner],
)

input_spec = Multipart(folder_name=Text(), results_container=JSON())


@svc.api(input=input_spec, output=JSON())
def process_batch(folder_name: str, results_container: json):
    loaded_dict = json.loads(results_container)
    results_container = defaultdict(
        lambda: {"idx": None, "name": None, "price": None, "file": None},
        loaded_dict,
    )
    paths = [
        Path(folder_name) / value["file"]
        for k, value in results_container.items()
        if value["file"] is not None
    ]


    # Segmentation
    results_seg = yolo_v8_runner_seg.inference.run(paths)
    tag_segmentator = YoloSegWrapper(task_id=SEG_TASK_ID, device=DEVICE)
    res_images = tag_segmentator.post_process(results_seg, results_container)

    # Detection
    results_det = yolo_v8_runner_det.inference.run(res_images)
    text_detector = YoloDetWrapper(task_id=DET_TASK_ID, device=DEVICE)
    crop_data, results_container = text_detector.post_process(
        results_det, res_images, results_container
    )

    # OCR
    char_recognizer = TrOCRWrapper(task_id=OCR_TASK_ID, device=DEVICE)
    for image_idx in results_container:
        images = char_recognizer.pre_process(crop_data, image_idx, save=False)
        ocr_result = TrOCR_runner.inference.run(images)
        text = char_recognizer.post_process(ocr_result)
        if text is None:
            continue
        results_container[image_idx]["text"] = text

    return results_container
