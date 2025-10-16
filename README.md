
# Multimodal Models for Robotics Vision

This is a project integrating YOLOv11 object detection and MolmoE-1B multimodal Mixture-of-Experts LLM. 

An image is first given to the orchestrator. It will then call YOLO to detect persons in the camera and Molmo to confirm if they are the same person or not. 




## Run Locally

First, download Python 3.11 and create a virtual environment 

```bash
  python -m venv .venv
```

```bash
  .venv\Scripts\activate
```

Clone the project

```bash
  git clone https://github.com/alsonjiang/Molmo-FYP.git
```

Go to the project directory

```bash
  cd Molmo-FYP
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Download finetuned MolmoE model

```bash
  python download_model.py
```

On separate terminals: 

1. Start the VLM service
```bash
  cd molmo-service
```
```bash
  python -m uvicorn app:app --host 0.0.0.0 --port 8000
```
2. Start the object detection service
```bash
  cd yolo-service
```
```bash
  python -m uvicorn app:app --host 0.0.0.0 --port 9000
```
3. Start the orchestrator script
```bash
  cd orchestrator
```
```bash
  set MOLMO_URL=http://localhost:8000/caption
```
```bash
  set YOLO_URL=http://localhost:9000/detect
```
```bash
  python modified_main.py
```

Save an image in the 'images' folder. 

In the modified_main.py terminal, when prompted, type the image path like so ../images/(your_image)



## Authors

- [@Alson Jiang](https://www.github.com/alsonjiang)
- [@Reuben Kway](https://www.github.com/reubzdubz)

