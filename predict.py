import argparse
import os
import torch
from torch import nn
from torchvision import datasets, models, transforms
from PIL import Image

def process_image(image_path):
    image_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    pil_image = Image.open(image_path)
    return image_transforms(pil_image)

def predict(image_path, model, device, topk=5):
    model.eval()
    tensor = process_image(image_path)
    tensor = tensor.unsqueeze_(0).float().to(device)
    with torch.no_grad():
        output = model(tensor)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)
        top_p = top_p.tolist()[0]
        top_class = top_class.tolist()[0]
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        top_classes = [idx_to_class[i] for i in top_class]
    return top_p, top_classes

def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    arch = checkpoint.get('model', 'resnet50')
    model = getattr(models, arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    if 'classifier' in checkpoint:
        model.fc = checkpoint['classifier']
    else:
        hidden_units = checkpoint.get('hidden_units', 1024)
        model.fc = nn.Sequential(
            nn.Linear(2048, hidden_units),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_units, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 102),
            nn.LogSoftmax(dim=1)
        )

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.to(device)
    return model

def parse_args():
    parser = argparse.ArgumentParser(description='Image Classifier.')
    parser.add_argument('image_path')
    parser.add_argument('checkpoint_path')
    parser.add_argument('--gpu', help='Use GPU for training', default='cpu')
    parser.add_argument('--topk', type=int, help='Return top K most likely classes', default=5)
    parser.add_argument('--category_names', help='Use a mapping of categories to real names', default='cat_to_name.json')
    return parser.parse_args()

def main():
    args = parse_args()
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

    model = load_checkpoint(args.checkpoint_path, device)

    if args.image_path:
        top_p, top_classes = predict(args.image_path, model, device, args.topk)
        class_names = None
        if args.category_names and os.path.exists(args.category_names):
            import json
            with open(args.category_names, 'r') as f:
                class_names = json.load(f)
        if class_names:
            named = [class_names.get(c, c) for c in top_classes]
            print('Top classes:', named)
        else:
            print('Top classes:', top_classes)
        print('Probabilities:', top_p)

if __name__ == '__main__':
    main()