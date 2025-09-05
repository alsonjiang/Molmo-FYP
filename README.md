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
