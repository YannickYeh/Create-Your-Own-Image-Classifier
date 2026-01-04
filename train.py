import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms, utils
from torch.utils.data import DataLoader

def build_dataloaders(data_dir, batch_size):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    train_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    validation_transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_datasets = dict()
    image_datasets['train'] = datasets.ImageFolder(train_dir,transform=train_transform)
    image_datasets['validation'] = datasets.ImageFolder(valid_dir,transform=validation_transform)
    image_datasets['test'] = datasets.ImageFolder(test_dir,transform=test_transform)
    dataloaders = dict()
    dataloaders['train'] = DataLoader(image_datasets['train'],batch_size=batch_size,shuffle=True)
    dataloaders['validation'] = DataLoader(image_datasets['validation'],batch_size=batch_size,shuffle=True)
    dataloaders['test'] = DataLoader(image_datasets['test'],batch_size=batch_size,shuffle=True)
    return dataloaders, image_datasets

def build_model(hidden_units, output_features, learning_rate, device, arch):
    if not hasattr(models, arch):
        raise ValueError(f"Unsupported architecture: {arch}")

    model = getattr(models, arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, 'fc'):
        input_features = model.fc.in_features
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            input_features = model.classifier[0].in_features
        else:
            input_features = model.classifier.in_features
    else:
        raise ValueError(f"Architecture {arch} does not expose a classifier head")

    classifier = nn.Sequential(
        nn.Linear(input_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_units, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, output_features),
        nn.LogSoftmax(dim=1)
    )

    if hasattr(model, 'fc'):
        model.fc = classifier
        trainable_params = model.fc.parameters()
    else:
        model.classifier = classifier
        trainable_params = model.classifier.parameters()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(trainable_params, lr=learning_rate)
    model.to(device)

    return model, criterion, optimizer

def train_model(model, criterion, optimizer, dataloaders, device, epochs, print_every):
    steps = 0
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in dataloaders['train']:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for inputs, labels in dataloaders['validation']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        output = model.forward(inputs)
                        loss = criterion(output, labels)
                        test_loss += loss.item()

                        ps = torch.exp(output)
                        _, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += equals.float().mean().item()

                print(
                    f"Epoch {epoch + 1}/{epochs}.. "
                    f"Train loss: {running_loss / print_every:.3f}.. "
                    f"Validation loss: {test_loss / len(dataloaders['validation']):.3f}.. "
                    f"Validation accuracy: {accuracy / len(dataloaders['validation']):.3f}"
                )
                running_loss = 0
                model.train()

def save_checkpoint(model, optimizer, epochs, image_datasets, save_path, learning_rate, hidden_units, arch, output_features):
    model.class_to_idx = image_datasets['train'].class_to_idx
    classifier = model.fc if hasattr(model, 'fc') else model.classifier
    checkpoint = {
        'model': arch,
        'classifier': classifier,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx,
        'epochs': epochs,
        'hidden_units': hidden_units,
        'output_features': output_features
    }
    torch.save(checkpoint, save_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Image Classifier Training')
    parser.add_argument('data_dir')
    parser.add_argument('--save_dir', help='Set directory to save checkpoints', default="checkpoint.pth")
    parser.add_argument('--learning_rate', help='Set the learning rate', type=float, default=0.001)
    parser.add_argument('--hidden_units', help='Set the number of hidden units', type=int, default=150)
    parser.add_argument('--output_features', help='Specify the number of output features', type=int, default=102)
    parser.add_argument('--epochs', help='Set the number of epochs', type=int, default=5)
    parser.add_argument('--batch_size', help='Set the batch size', type=int, default=16)
    parser.add_argument('--print_every', help='Steps between status prints', type=int, default=40)
    parser.add_argument('--gpu', help='Use GPU for training', action='store_true')
    parser.add_argument('--arch', help='Choose architecture', default='resnet50')
    return parser.parse_args()

def main():
    args = parse_args()
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    dataloaders, image_datasets = build_dataloaders(args.data_dir, args.batch_size)
    model, criterion, optimizer = build_model(
        args.hidden_units,
        args.output_features,
        args.learning_rate,
        device,
        args.arch
    )

    train_model(model, criterion, optimizer, dataloaders, device, args.epochs, args.print_every)
    save_checkpoint(
        model,
        optimizer,
        args.epochs,
        image_datasets,
        args.save_dir,
        args.learning_rate,
        args.hidden_units,
        args.arch,
        args.output_features
    )

if __name__ == '__main__':
    main()
