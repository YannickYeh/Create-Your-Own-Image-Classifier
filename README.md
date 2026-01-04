# Image Classifier Project

This project is a CLI application that allows users to train a deep learning model on a dataset of images and then use that model to predict the classes of new images. Built with Python and PyTorch.

## 🚀 Key Features
- **Transfer Learning:** Uses pre-trained models (VGG16, ResNet50, etc.) to achieve high accuracy with small datasets.
- **Customizable:** Choose your architecture, learning rate, and hidden units via CLI arguments.
- **GPU Support:** Automatically detects and uses CUDA for faster training.

## 🛠 Installation & Setup
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/YannickYeh/Create-Your-Own-Image-Classifier.git](https://github.com/YannickYeh/Create-Your-Own-Image-Classifier.git)
   cd Create-Your-Own-Image-Classifier

2. **Recommended Setup:**
   
   You should have cuda available on you local computer, otherwise it will likely run very slow and won't work well.I recommend you to install later PyTorch version like `2.9.1+cu128` so cuda is available.

## Dataset
The dataset contains 102 different types of flowers with ~25 each.It is so big that it is almost impossible to run it on CPU.You can download it in `Download_the_Data.ipynb`.

## Training
You will train the flower dataset by running `train.py`.

## Testing
You can test the trained model by running `predict.py`.
You have to provide the image path and the checkpoint path.

For example:

```python predict.py 'flowers/test/1/image_06752.jpg' checkpoint.pth```

## Explain Combined_Image_Classifier.ipynb
```Combined_Image_Classifier.ipynb``` is just ```Image_Classifier_Project.ipynb``` but I merged the training cell with the saving checkpoint cell, I also merged the testing cell with the loading checkpoint cell.
