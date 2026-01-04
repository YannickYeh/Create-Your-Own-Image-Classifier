# Image Classifier Project

## Udacity AI Programming with Python: Image Classifier

This repository contains the code for a command-line image classifier built with **PyTorch**. The application uses transfer learning (VGG16 or ResNet50) to identify different species of flowers.

## Key Features
- **Transfer Learning:** Uses pre-trained models (VGG16, ResNet50, etc.) to achieve high accuracy with small datasets.
- **Customizable:** Choose your architecture, learning rate, and hidden units via CLI arguments.
- **GPU Support:** Automatically detects and uses CUDA for faster training.

## Project Structure
- `train.py`: Trains a new network on a dataset and saves the model as a checkpoint.
- `predict.py`: Uses a trained network to predict the class for an input image.
- `Image_Classifier_Project.ipynb`: 'train.py' and 'predict.py' into one single file.
- 'Combined_Image_Classifier.ipynb': Image_Classifier_Project but I combined some cells together.

## Installation & Setup
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/YannickYeh/Create-Your-Own-Image-Classifier.git](https://github.com/YannickYeh/Create-Your-Own-Image-Classifier.git)
   cd Create-Your-Own-Image-Classifier

2. **Recommended Setup:**
   
   You should have cuda available on you local computer, otherwise it will likely run very slow and won't work well.I recommend you to install later PyTorch version like `2.9.1+cu128` so cuda is available.

3. **This project requires Python 3.9+ and the following libraries:**

   - PyTorch
   - Torchvision
   - NumPy
   - Pillow
   - Matplotlib
   
   Install all dependencies using:
   
   ```bash
   pip install -r requirements.txt

## Dataset
The dataset contains 102 different types of flowers with ~25 each.It is so big that it is almost impossible to run it on CPU.You can download it in `Download_the_Data.ipynb`.

Or you can follow these 3 steps:
1. Download the dataset [from this link](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz).
2. Create a folder named `flowers/` in the project root.
3. Extract the `train`, `valid`, and `test` folders into it.

## Command Line Arguments

The project uses two main scripts: `train.py` for model training and `predict.py` for inference.

### 1. Training (`train.py`)
This script allows you to train a new network on a dataset and save the model as a checkpoint.

| Argument | Description | Default |
| :--- | :--- | :--- |
| `data_dir` | **Required.** Path to the dataset (e.g., `flowers`) | N/A |
| `--save_dir` | Directory path or filename to save the checkpoint | `checkpoint.pth` |
| `--arch` | Model architecture (choose `vgg16` or `resnet50`) | `resnet50` |
| `--learning_rate` | Learning rate for the optimizer | `0.001` |
| `--hidden_units` | Number of hidden units in the classifier | `150` |
| `--epochs` | Number of training epochs | `5` |
| `--batch_size` | Number of images per training batch | `16` |
| `--gpu` | Include this flag to use GPU (CUDA) for training | `False` |

**Example Command:**
```bash
python train.py flowers --arch resnet50 --learning_rate 0.01 --hidden_units 512 --gpu
```

---

### 2. Prediction (`predict.py`)
This script uses a saved checkpoint to predict the class of a single image.

| Argument | Description | Default |
| :--- | :--- | :--- |
| `path_to_image` | **Required.** Path to the image file | N/A |
| `checkpoint` | **Required.** Path to the saved model `.pth` file | N/A |
| `--top_k` | Return the top K most likely classes | `3` |
| `--category_names` | Path to a JSON file mapping categories to names | `cat_to_name.json` |
| `--gpu` | Include this flag to use GPU for inference | `False` |

**Example Command:**
```bash
python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --top_k 5 --gpu
```
