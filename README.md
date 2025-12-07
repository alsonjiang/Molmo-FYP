
# Application of Multimodal Models for Robotics Vision

This is a project integrating YOLOv11 object detection and Multimodal Models.

The two currently in use within this project are a custom-quantised [MolmoE-1B](https://huggingface.co/reubk/MolmoE-1B-0924-NF4) multimodal Mixture-of-Experts LLM and small vision language model [Moondream2](https://huggingface.co/vikhyatk/moondream2)

The main script has two modes: 

For semantic description, the VLM will describe any persons detected in the camera. 
  
For identity matching, an image is first given to the orchestrator. It will then call YOLO to detect persons in the camera and then the VLM to confirm if they are the same person or not. 



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

Download custom Molmo model

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
For live semantic description, use:
```bash
  python main.py caption
```
For live identity matching, use:
```bash
  python main.py identity

  ../images/(your_image)
```

## Running Tests

To run tests, run the following commands from the orchestrator folder.  
Activate the VLM service first from the folder of the VLM of interest.

1. Image Comparison Test:
```bash
  python compare_two_images.py (path_to_1st_image) (path_to_2nd_image)

```
2. Multiple Images Benchmarking Test:
```bash
  python multiple_images_benchmark.py

```
3. Prompt Engineering Test:
```bash
  python prompt_ablation_test.py

```
4. 20 Frames Caption-Only Test:
Requires the acivation of the object detection service.
```bash
  python live_caption_test.py

```


## Authors

- [@Alson Jiang](https://www.github.com/alsonjiang)
- [@Reuben Kway](https://www.github.com/reubzdubz)

