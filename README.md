# yolo_trocr_bentoml

### Virtual env
#### create env
```bash
virtualenv -p python3.8 bentoml_env
```

#### install libs
```bash
pip install -r requirements.txt
```

#### download yolov8 weights
```bash
python download_models.py
```

#### change permissions (because in docker it won't work without it)
```bash
chmod 776 yolov8n-seg.pt
chmod 776 yolov8n.pt
```
---
### RUN WITH SERVE
```bash
BENTOML_CONFIG=configuration.yml bentoml serve service.py:svc -p 8995 --development --reload --debug 
```

#### Questions for serve:
1. How to allocate gpu: 1 with it?

---
### RUN WITH DOCKER
```bash
bentoml build
```
```bash
bentoml bentoml containerize ml_pipeline_service:latest
```
```bash
docker run --rm --gpus '"device=1"' -v ./configuration.yml:/home/bentoml/configuration.yml -v $(pwd):/home/bentoml/bento/src/  -p 8995:3000 ml_pipeline_service:(your_tag)
```
---

### Problems:

1. Memory issues arise when executing within a container, witnessing a cumulative increase of 500MB with each inference attempt.

2. While performing a 30-batch operation, an error is encountered, surpassing the 60-second time limit.(can't reproduce in this repo because of different data and pre/post processing)

3. Running with `BENTOML_CONFIG=configuration.yml bentoml serve service.py:svc -p 8995 --development --reload --debug` consumes significantly less memory than execution within a Docker container. Approximately 3GB for serve of memory is utilized, and for docker 6-8GB, which escalates further with increased batch sizes

4. A RuntimeError is experienced with a batch size of 30, displaying the message: "Unexpected ASGI message 'http.response.start' sent after the response has already been completed." Despite successful functionality and the desired response, this error persists. (can't reproduce in this repo because of different data and pre/post processing)

5. Swagger functionality is impaired when using the Docker container, but it works seamlessly with `bentoml serve`. Investigating the reason for this disparity. 

6. What benefits I get while using custom runner with transformers lib and bentoml.transformers? 

7. Can't send defaultdict as input, but send as output is fine. 

