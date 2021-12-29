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
            outputs = model(data)
            loss = criterion(outputs, target)
            _, preds = torch.max(outputs, 1)
            test_loss += loss.item() * data.size(0)
            correct += torch.sum(preds == target.data).item()

    total_loss = test_loss / len(test_loader.dataset)
    total_acc = correct / len(test_loader.dataset)
    
    logger.info(f"Test set: Average loss: {total_loss:.2f}, Accuracy: {total_acc:.2f}")

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

            for inputs, labels in image_dataset[phase]:
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
                running_corrects += torch.sum(preds == labels.data)
                #running_samples+=len(inputs)
                #if running_samples % 2000  == 0:
                    #accuracy = running_corrects/running_samples
                    #print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                            #running_samples,
                            #len(image_dataset[phase].dataset),
                            #100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            #loss.item(),
                            #running_corrects,
                            #running_samples,
                            #100.0*accuracy,
                        #)
                    #)
            phase_loss = running_loss / len(image_dataset[phase].dataset)
            phase_acc = running_corrects / len(image_dataset[phase].dataset)
            logger.info(f"\nEpoch {epoch}, Phase {phase}")
            logger.info(f"\nBest Loss: {best_loss:.2f}, Phase Loss: {phase_loss:.2f}, Phase Accuracy: {phase_acc:.2f}") 
            if phase=='valid':
                if phase_loss<best_loss:
                    best_loss=phase_loss
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
    model.fc = nn.Sequential(nn.Linear(num_features, 512),
                             nn.ReLU(inplace=True),
                             nn.Linear(512, 133),
                             )
    return model
    
def create_data_loaders(data, batch_size):
    
    train_data_path = os.path.join(data, 'train') 
    test_data_path = os.path.join(data, 'test')
    validation_data_path=os.path.join(data, 'valid')
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])
                                                            
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])

    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size) 
    
    return train_data_loader, validation_data_loader, test_data_loader 

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on Device {device}")
    for key, value in vars(args).items():
        logger.info(f"{key}:{value}")
    model=net()
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
    parser.add_argument("--batch_size", type=int, default=64, metavar = "N")
    parser.add_argument("--epochs", type=int, default=2, metavar="N")
    parser.add_argument("--lr", type=float, default=0.1, metavar = "LR")
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--model_dir", type=str, default=os.environ['SM_MODEL_DIR'] )
    parser.add_argument("--data_dir", type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    
    main(args)
