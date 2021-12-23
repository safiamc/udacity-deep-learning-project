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

from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook

def test(model, test_loader):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    pass

def train(model, train_loader, validation_loader, criterion, optimizer, epochs, device, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    hook=get_hook(create_if_not_exists=True)
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    hook.register_loss(criterion)
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                model.train()
                hook.set_mode(smd.modes.TRAIN)
            else:
                model.eval()
                hook.set_mode(smd.modes.EVAL)
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
    pass

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    for key, value in vars(args).items():
        logger.info(f"{key}:{value}")
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = None
    optimizer = None
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model, train_loader, validation_loader, loss_criterion, optimizer, epochs, device, hook)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, criterion)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--gpu", type=str2bool, default=True)
    parser.add_argument("--model", type=str, default="resnet50")
    args=parser.parse_args()
    
    main(args)
