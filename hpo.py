#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def test(model, test_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()    
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data=data.to(device)
            target=target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            pred = output.argmax(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, validation_loader, criterion, optimizer, epochs, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_samples=0

            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples+=len(inputs)
                if running_samples % 2000  == 0:
                    accuracy = running_corrects/running_samples
                    print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                            running_samples,
                            len(image_dataset[phase].dataset),
                            100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0*accuracy,
                        )
                    )
            phase_loss = running_loss / running_samples
            phase_acc = running_corrects / running_samples
            logger.info(f"\nEpoch {epoch}, Phase {phase}")
            logger.info(f"\nBest Loss: {best_loss:.4f}, Phase Loss: {phase_loss:.4f}, Phase Accuracy: {phase_acc:.4f}") 
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
    return model

def model_fn(model_dir):
    model = Net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)
    for param in model.parameters(): #freeze pretrained model weights
        param.requires_grad = False
        
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 200),
                             nn.ReLU(),
                             nn.Linear(200, 133),
                             nn.Softmax()
                            )
    return model
    
def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_data_path = os.path.join(data, 'train')
    validation_data_path=os.path.join(data, 'valid')
    test_data_path=os.path.join(data, 'test')
    
    train_transform = transforms.Compose(
        [transforms.RandomResizedCrop((224,224)),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()
        ]
    )
    test_transform = transforms.Compose(
        [transforms.Resize((224,224)),
         transforms.ToTensor()
        ]
    )
    train_set = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )
    valid_set = torchvision.datasets.ImageFolder(root=valid_data_path, transform = test_transform)
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False
    )
    test_set = torchvision.datasets.ImageFolder(root=test_data_path, transform = test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, validation_loader, test_loader
    

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    for key, value in vars(args).items():
        logger.info(f"{key}:{value}")
    model=model.to(device)
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr = args.lr )
    batch_size=args.batch_size
    epochs=args.epochs
    train_loader, validation_loader, test_loader = create_data_loaders(args.data_dir, batch_size)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    logger.info("Training the model")
    model=train(model, train_loader, validation_loader, loss_criterion, optimizer, epochs, device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    logger.info("Testing the model")
    test(model, test_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    
    path = os.path.join(args.model_dir, "model.pth")
    logger.info(f"Saving the model to path: {path}")
    torch.save(model.state_dict(), path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"] )
    parser.add_argument("--data_dir", type=str, default="s3://sagemaker-us-east-1-755391689112/dogImages/")
    
    args=parser.parse_args()
    
    main(args)
