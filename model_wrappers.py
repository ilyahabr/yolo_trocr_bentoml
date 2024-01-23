import typing as tp
from collections import defaultdict

import albumentations as A
import numpy as np
import torch
import ultralytics
from loguru import logger
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class YoloSegWrapper:
    def __init__(self, task_id: str, device: str = "cpu"):
        self.task_id = task_id
        self.device = device
        self.model = None
        self.infer_res = []  # TODO rename

    def load_model(self, task: str = "segment") -> ultralytics.YOLO:
        # weights = download_from_s3(self.task_id)
        self.model = ultralytics.YOLO('yolov8n-seg.pt', task=task)
        return self.model

    def predict(self, batch: tp.List, save: bool = False) -> tp.List:
        results = self.model.predict(
            batch, stream=False, save=save, conf=0.2, verbose=False, device=self.device
        )
        return results

    def post_process(self, results, container: defaultdict) -> tp.List:
        self.infer_res = []
        for result, image_idx in zip(results, container):
            processed_image = self.post_process_item(result, image_idx)
            self.infer_res.append(processed_image)
        return self.infer_res

    def post_process_item(self, result, image_idx: int) -> np.ndarray:
        if result.masks is None:
            logger.debug(f"No mask has found for image id {image_idx}!")
            return result.orig_img  # TODO resize first

        return result


class YoloDetWrapper:
    def __init__(self, task_id: str, device: str = "cpu"):
        self.task_id = task_id
        self.device = device
        self.model = None

    def load_model(self, task: str = "detect") -> ultralytics.YOLO:
        # weights = download_from_s3(self.task_id)
        self.model = ultralytics.YOLO('yolov8n.pt', task=task)
        return self.model

    def predict(self, batch: tp.List, save: bool = False) -> tp.List:
        results = self.model.predict(
            batch,
            stream=False,
            save=save,
            conf=0.05,
            verbose=False,
            device=self.device,
        )
        return results

    def post_process(
        self, results, images: tp.List[np.ndarray], container: defaultdict
    ) -> tp.List:
        crop_data = {}
        empty_result_idx = []
        for result, image, image_idx in zip(results, images, container):
            if result.boxes.data.numel() == 0:  # no det
                # TODO log failed image name from result
                empty_result_idx.append(image_idx)
                continue

            crops_classes, image_crops = self.post_process_item(result, image)

            crop_data[image_idx] = {
                "texts_classes": crops_classes,
                "image_crops": image_crops,
            }

        for image_idx in empty_result_idx:
            del container[image_idx]
        return crop_data, container

    def post_process_item(
        self, result, image
    ) -> tp.Tuple[tp.List[str], tp.List[np.ndarray]]:
        crops_classes = []
        image_crops = []
        for res_det in result:
            boxes = res_det.boxes
            for box in boxes:
                roi = box.xyxy[0].type(torch.int32)
                text_class = res_det.names[int(box.cls)]
                crop = image[roi[1] : roi[3], roi[0] : roi[2]]
                crops_classes.append(text_class)
                image_crops.append(crop)
        return crops_classes, image_crops


class TrOCRWrapper:
    def __init__(self, task_id: str, device: str = "cpu"):
        self.task_id = task_id
        self.device = device
        self.processor = None
        self.model = None
        self.results = []
        self.transform = A.Compose(
            [
                A.ToGray(p=1),
                A.Resize(height=384, width=384, p=1, interpolation=1),
            ]
        )

    def load_model(self):
        # weights = download_from_s3(self.task_id)
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-printed",
        )
        #self.model.load_state_dict(torch.load(weights, map_location=self.device))
        self.model.half()  # might hurt perfomance but 6x times faster and 2x less memory.
        return self.model.to(self.device)

    def predict(self, images) -> tp.List:
        with torch.no_grad():
            pixel_values = self.processor(images, return_tensors="pt").pixel_values
            pred = self.model.generate(
                pixel_values.to(self.device), num_beams=8, max_new_tokens=128
            )
            results = self.processor.batch_decode(pred, skip_special_tokens=True)
        return results

    def pre_process(
        self, raw_batch: dict, image_idx: str) -> tp.List[np.ndarray]:

        images = [
            self.transform(image=img)["image"]
            for img in raw_batch[image_idx]["image_crops"]
        ]
        return images

    def post_process(self, results) -> str:
        # just postprocess str
        pass
