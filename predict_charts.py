
import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from PIL import Image, ImageOps
from scipy.stats import gmean



import pickle
import pandas as pd
from PIL import Image, ImageOps




from torchvision import transforms
 
# define custom transform
# here we are using our calculated
# mean & std
transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.97, 0.135)
])


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

files = os.listdir('validation_plots')
df = pd.DataFrame()

#df['Path'] = [f for f in files if '200png.png' in f]
#df['Path2'] = [f for f in files if 

path = []
path2 = []

for f in os.listdir('Validation_Data'):
    f = '.'.join(f.split('.')[:-2])
    print(f)
    if os.path.exists('validation_plots/{}_recommendation_200png.png'.format(f)) and os.path.exists('validation_plots/{}_recommendation_400png.png'.format(f)):
        path.append('validation_plots/{}_recommendation_200png.png'.format(f))
        path2.append('validation_plots/{}_recommendation_400png.png'.format(f))
df['Path'] = path
df['Path2'] = path2
df['Score2'] = 1

class FinalDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        return (
            transform_norm(ImageOps.grayscale(Image.open(row["Path"])).resize((256, 256))),
            transform_norm(ImageOps.grayscale(Image.open(row["Path2"])).resize((256, 256))),
           torch.from_numpy(np.array([row["Score2"]])).float(),
            
        )
class FinalDataset2(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        return (
            torch.from_numpy(row['input']).float(),
           torch.from_numpy(np.array([row["Score2"]])).float(),
            
        )



# Define the CNN architecture
class MultiOutputCNN(nn.Module):
    def __init__(self):
        super(MultiOutputCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1)
        
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1)
        
        
        self.fc1 = nn.Linear(4 * 8 * 8, 16)  # Adjust input size based on your data
        self.fc2 = nn.Linear(16, 7)  
        #self.fc3 = nn.Linear(64, 7)
        
        self.act1 = nn.ReLU()
        


    def forward(self, x):
        #print(x.shape)
        
        x = self.pool(torch.relu(self.conv1(x)))
        
        #print(x.shape)
        x = self.pool(torch.relu(self.conv2(x)))
        #print(x.shape)
        x = x.view(-1, 4 * 8 * 8)  # Adjust size for the fully connected layer
        
        x = torch.relu(self.fc1(x))
        output = self.fc2(x)
        return output
    
    
    def forward2(self, x):
        print(x.shape)
        
        x = self.pool(torch.relu(self.conv1(x)))
        print('layer1:', x)
        
        #print(x.shape)
        x = self.pool(torch.relu(self.conv2(x)))
        print('layer 2:', x)
        #print(x.shape)
        x = x.view(-1, 4 * 8 * 8)  # Adjust size for the fully connected layer
        
        x = torch.relu(self.fc1(x))
        print('layer 3:', x)
        output = self.fc2(x)
        return output
# Instantiate the model
model = MultiOutputCNN()





# Define the CNN architecture
class FinalModel(nn.Module):
    def __init__(self):
        super(FinalModel, self).__init__()
        
        self.fc1 = nn.Linear(32, 10)  # Adjust input size based on your data
        
        self.fc2 = nn.Linear(10, 8)
        self.fc3 = nn.Linear(8, 4)
        self.fc4 = nn.Linear(4, 1)
        self.act1 = nn.ReLU()
        


    def forward(self, x):
        #print(x.shape)
        
        x = self.act1(self.fc1(x))
        
        #print(x.shape)
        x = self.act1(self.fc2(x))
        
        x = self.act1(self.fc3(x))
       
        output = self.fc4(x)
        return output
    
    
# Instantiate the model
fin_model = FinalModel()
model.load_state_dict(torch.load("model_stage_1.pt", map_location='cpu'))
fin_model.load_state_dict(torch.load("model_stage_2.pt", map_location='cpu'))
model.to(device)
fin_model.to(device)

print('model loaded')

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.fc1.register_forward_hook(get_activation('fc1'))

dataset_validation = FinalDataset(df)
dataset_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=32, shuffle=False)


input_data = np.array([[]])
with torch.no_grad():
    for images, images_400, labels in dataset_validation:
        images = images.to(device)
        outputs = model(images)
        inp = activation['fc1'].to('cpu').numpy()
        if input_data.shape[1] == 0:
            input_data = inp
        else:
            input_data = np.concatenate((input_data, inp), axis =0)



input_data2 = np.array([[]])
with torch.no_grad():
    for images, images_400, labels in dataset_validation:
        images_400 = images_400.to(device)
        outputs = model(images_400)
        inp = activation['fc1'].to('cpu').numpy()
        if input_data2.shape[1] == 0:
            input_data2 = inp
        else:
            input_data2 = np.concatenate((input_data2, inp), axis =0)

df['input']  = list(np.concatenate((input_data, input_data2), axis = 1))


print("stage 1 done")
dataset_validation = FinalDataset2(df)
dataset_validation_loader = torch.utils.data.DataLoader(dataset_validation, batch_size=32, shuffle=False)
all_labels = []
all_outputs = []

with torch.no_grad():

    for inputs, labels in dataset_validation_loader:
        inputs = inputs.to(device)

        labels = labels.to(device)


        outputs = fin_model(inputs)

        all_labels.extend(labels)
        all_outputs.extend(outputs)
df['preds'] = [np.array(_.to('cpu'))[0] for _ in all_outputs]

df['stock'] = df['Path'].apply(lambda x: x.split('/')[-1].split('_')[0])

df = df.sort_values('preds', ascending = False)


df.to_csv('stock_predictions.csv', index = False)
