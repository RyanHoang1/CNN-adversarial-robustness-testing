import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split

import random

# eliminate rng between runs
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Load the dataset and train, val, test splits
print("Loading datasets...")
FASHION_transform = transforms.Compose([
    transforms.ToTensor(), # Transform from [0,255] uint8 to [0,1] float
])
FASHION_trainval = datasets.FashionMNIST('.', download=True, train=True, transform=FASHION_transform)
FASHION_train = Subset(FASHION_trainval, range(50000))
FASHION_test = datasets.FashionMNIST('.', download=True, train=False, transform=FASHION_transform)
print("Done!")

# Create dataloaders
trainloader = DataLoader(FASHION_train, batch_size=100, shuffle=True)
testloader = DataLoader(FASHION_test, batch_size=100, shuffle=True)

# Specify model architecture layers
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.cv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(64*6*6, 2056)
        self.fc2 = nn.Linear(2056, 256)
        self.fc3 = nn.Linear(256,10)

    def forward(self,x):
        x = self.cv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.pool1(x)

        x = self.cv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1) # Flatten each image in the batch
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        # The loss layer will be applied outside LeNet class
        return x

device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
model = LeNet().to(device)
model.load_state_dict(torch.load('CNN_Base_15_Epoch.pt')) # load model trained on 15 epochs

criterion = nn.CrossEntropyLoss() # Specify the loss layer
optimizer = optim.SGD(model.parameters(), lr=.02)
num_epoch = 15
epsilon = 25/255 # epsilon value used in adversarial training(Change this value accordingly)

#fgsm evasion attack
def fgsm_evasion_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    return perturbed_image

def train(model, loader, testloader, num_epoch, epsilon): # Train the model
    print("Start training...")
    model.train() # Set the model to training mode
    epoch_train_loss = [] 
    epoch_test_loss = []
    for i in range(num_epoch):
        running_loss = []
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            
            # train on fgsm adversarial images
            batch.requires_grad = True
            optimizer.zero_grad() # Clear gradients
            output = model(batch)
            loss = criterion(output, label)
            model.zero_grad()
            loss.backward()
            data_grad = batch.grad.data
            adv_batch = fgsm_evasion_attack(batch, 10/255, data_grad)

            pred = model(adv_batch) # This will call LeNet.forward()
            loss = criterion(pred, label) # Calculate the loss
            running_loss.append(loss.item())
            loss.backward() # Backprop gradients to all tensors in the network
            optimizer.step() # Update trainable weights

            # train on pgd adversarial images
            optimizer.zero_grad() # Clear gradients
            
            output = model(batch)
            loss = criterion(output, label)
            model.zero_grad()
            loss.backward()
            adv_batch = pgd_evasion_attack(model, batch, label, 40/255)

            pred = model(adv_batch) # This will call LeNet.forward()
            loss = criterion(pred, label) # Calculate the loss
            running_loss.append(loss.item())
            loss.backward() # Backprop gradients to all tensors in the network
            optimizer.step() # Update trainable weights
        print("Epoch {} loss:{}".format(i+1,np.mean(running_loss))) # Print the average loss for this epoch
        epoch_train_loss.append(np.mean(running_loss))
        epoch_test_loss.append(evaluate(model, testloader))
    print("DONE TRAINING")
    plt.plot(epoch_train_loss, label="training loss")
    plt.plot(epoch_test_loss, label="testing loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('loss across epochs')
    plt.legend()
    plt.show()

def evaluate(model, loader): # Evaluate accuracy on test set
    model.eval() # Set the model to evaluation mode
    correct = 0
    with torch.no_grad(): # Do not calculate gradient to speed up computation
        running_loss = []
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            loss = criterion(pred, label) # Calculate the loss
            running_loss.append(loss.item())
            correct += (torch.argmax(pred,dim=1)==label).sum().item()
    acc = correct/len(loader.dataset)
    print("loss:{}".format(np.mean(running_loss)))
    print("Evaluation accuracy: {}".format(acc))
    return np.mean(running_loss)

#test fgsm attack efficacy on test set
def eval_fgsm_adv_images(model, device, testloader, epsilon):
    # Accuracy counter
    correct = 0

    # Loop over all examples in test set(in batches of 100)
    for data, label in tqdm(testloader):
        data, label = data.to(device), label.to(device)
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)

        # Calculate the loss
        loss = criterion(output, label)

        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        # perform fgsm attack
        perturbed_data = fgsm_evasion_attack(data, epsilon, data_grad)

        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1] # stores attacked predictions

        # Check success
        for i in range(len(final_pred)):
            if final_pred[i] == label[i]:
                correct += 1

    # Calculate final accuracy
    final_acc = correct/(float(len(testloader))*100)
    print("Accuracy = {} / {} = {}".format(correct, (len(testloader)*100), final_acc))

    # Return the accuracy
    return final_acc

#pgd evasion attack
def pgd_evasion_attack(model, images, labels, epsilon, steps=10, step_size=0.01):
    loss = nn.CrossEntropyLoss()

    original_images = images.data

    for i in range(steps):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adversarial_images = images + step_size*images.grad.sign()
        eta = torch.clamp(adversarial_images - original_images, min=-epsilon, max = epsilon)
        images = torch.clamp(original_images + eta, 0, 1).detach_()

    return images

#test pgd attack efficacy on test set
def eval_pgd_adv_images(model, device, testloader, epsilon):
    #Accuracy Counter
    correct = 0
    # Loop over all examples in test set(in batches of 100)
    for data, label in tqdm(testloader):
        data, label = data.to(device), label.to(device)
        # perform pgd attack
        perturbed_data = pgd_evasion_attack(model, data, label, epsilon)

        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1] # stores attacked predictions

        # Check success
        for i in range(len(final_pred)):
            if final_pred[i] == label[i]:
                correct += 1
    
    # Calculate final accuracy
    final_acc = correct/(float(len(testloader))*100)
    print("Steps 10 - Accuracy = {} / {} = {}".format(correct, (len(testloader)*100), final_acc))

    # Return the accuracy
    return final_acc

print('Training on 15 epoch model')
train(model, trainloader, testloader, num_epoch, epsilon)

print('------------------------------------------------------------------------------------------------------------------------------------------------------')
epsilon = 25/255 # use epsilon = 25/255 for attack
# Run fgsm-attack test
print('FGSM attack eval')
eval_fgsm_adv_images(model, device, testloader, epsilon)
# Run pgd-attack test
print('PGD attack eval')
eval_pgd_adv_images(model, device, testloader, epsilon)