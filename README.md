Create a vritual environment:  
python -m venv .venv  
source .venv/bin/activate  # (Windows: .\.venv\Scripts\activate)  

Install torch:  
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0  

Install all other dependencies:  
pip install -r requirements.txt  
