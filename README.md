# Trade-Offs Between Compression and Accuracy for Skin Disease Classification in Edge Devices
### final report is in Final-Report_BIOE.pdf file

# Use user_test_colab.ipynb to run in Google Colab, no inputs needed - does a random pruning value
## Please make sure train and val fodlers are in same directory you are running the ipynb file in
## If you are using gpu on colab, there may be issues running the code, but with cpu should work fine

# How to Run Code on Terminal
### example how to run test file: python user-test.py pruning_ratio
### pruning_ratio should be replaced with a pruning ratio (float value) greater than 0 and less than or equal to 1.0
### e.g. for 20% pruning ratio: python user-test.py 0.2 

# Initial Implementations
### https://docs.google.com/document/d/1s4O57IubOKrpDEAyFQeiSYAJHuWbnvEPz5jnNCoKU7U/edit?usp=sharing

# Imports
### Need to install torch if not already installed: pip install torchvision
### May need to import torch_pruning: pip install torch_pruning
