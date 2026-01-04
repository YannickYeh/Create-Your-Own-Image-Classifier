# Project: Create You Own Image Classifier
This is the project code for Udacity's AI Programming with Python course.In this project, you train a image classifier on a dataset using PyTorch, then convert it into a command line app.

# Required Setup
You should have CUDA available on you local computer, otherwise it will likely run very slow and won't work well.You should also have all the python packages installed on your local environment.I recommend you install later PyTorch version like 2.9.1+cu128 so cuda is available.

# Dataset
I am using a dataset that contains 102 different types of flowers with ~25 each.It is so big that it is almost impossible to run it on CPU.You can download it in Download_the_Data.ipynb.

# Training
You will train the flower dataset by running train.py.

# Testing
You can test the trained model by running predict.py.
You have to provide the image path and the checkpoint path.

For example:

```python predict.py 'flowers/test/1/image_06752.jpg' checkpoint.pth```
