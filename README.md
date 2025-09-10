1. Install NVIDIA driver and reboot 
sudo ubuntu-drivers autoinstall 
sudo reboot 
 
2. Install Docker Engine and NVIDIA Container Toolkit 
Docker Engine 
curl -fsSL https://get.docker.com | sh 
sudo usermod -aG docker $USER 
newgrp docker 

NVIDIA Container Toolkit 
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

Verify GPU: 
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

3. Clone this repository
git clone https://github.com/alsonjiang/Molmo-FYP.git
cd Molmo-FYP

4. Build the Docker image (~13GB)
docker build -f Dockerfile.cuda -t molmo-test:latest .

5. Download model weights 
docker run --rm -it \
  -v "$PWD:/models" \
  -w /models \
  molmo-test:latest \
  python /app/download_model.py

6. Run inference 
docker run --rm -it --gpus all \
  -v "$PWD/MolmoE-1B-0924-NF4:/molmoE" \
  -e MODEL_DIR=/molmoE \
  molmo-test:latest



*Deprecated* 
Create a vritual environment:  
python -m venv .venv  
source .venv/bin/activate  # (Windows: .\.venv\Scripts\activate)  

Install torch:  
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0  

Install all other dependencies:  
pip install -r requirements.txt  

To run:  

1. Run download_model.py
This creates a folder named MolmoE-1B-0924-NF4 in your project root that stores the model

2. Run model_test.py 
This tests the model and the working environment

Expected output:
The image displays a solid gray square with no visible content or features. 
It appears to be a plain, unadorned gray square without any discernible elements, patterns, or variations in color or texture.
