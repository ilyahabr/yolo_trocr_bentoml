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

### RUN WITH DOCKER
```bash
bentoml build
```
```bash
bentoml bentoml containerize ml_pipeline_service:latest
```
```bash
docker run --rm --gpus '"device=1"' -v ./configuration.yml:/home/bentoml/configuration.yml -v $(pwd):/home/bentoml/bento/src/  -p 8995:3000 ml_pipeline_service:mq46sqvzvknho7lb
```

### RUN WITH SERVE
```bash
BENTOML_CONFIG=configuration.yml bentoml serve service.py:svc -p 8995 --development --reload --debug 
```

#### Questions for serve:
1. How to allocate gpu: 1 with it?

### Challenges Encountered:

1. **Memory Accumulation in Container:**
   - Memory issues arise when executing within a container, witnessing a cumulative increase of 500MB with each inference attempt.

2. **Time Limit Exceeded during Batch Operation:**
   - While performing a 30-batch operation, an error is encountered, surpassing the 60-second time limit.

3. **Discrepancy in Memory Usage:**
   - Running with `BENTOML_CONFIG=configuration.yml bentoml serve service.py:svc -p 8995 --development --reload --debug` consumes significantly less memory than execution within a Docker container. Approximately 3GB of memory is utilized, which escalates further with increased batch sizes (e.g., 5500 for batch 30).

4. **RuntimeError with Batch Size 30:**
   - A RuntimeError is experienced with a batch size of 30, displaying the message: "Unexpected ASGI message 'http.response.start' sent after the response has already been completed." Despite successful functionality and the desired response, this error persists.

5. **Swagger Not Functional in Docker Container:**
   - Swagger functionality is impaired when using the Docker container, but it works seamlessly with `bentoml serve`. Investigating the reason for this disparity.

---
