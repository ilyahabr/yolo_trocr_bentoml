# YOLO-TrOCR BentoML Service

A machine learning service that combines YOLO object detection and TrOCR text recognition using BentoML.

## 🚀 Getting Started

### Prerequisites

1. Create Virtual Environment
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Download YOLO Models
```bash
python download_models.py
```

4. Set Model Permissions
```bash
chmod 776 yolov8n-seg.pt
chmod 776 yolov8n.pt
```

## 🛠️ Running the Service

### Local Development
Run the service locally with hot-reload:
```bash
BENTOML_CONFIG=configuration.yml bentoml serve service.py:svc -p 8995 --development --reload --debug 
```

### Docker Deployment
1. Build the Docker container:
```bash
bentoml build --containerize
```

2. Run the container:
```bash
docker run --rm \
  -v $(pwd)/configuration.yml:/home/bentoml/configuration.yml \
  -v $(pwd):/home/bentoml/bento/src/ \
  -p 8995:3000 \
  ml_pipeline_service:q6ttjkw3z6wnslql
```

### Testing
Run the test client:
```bash
python client.py
```