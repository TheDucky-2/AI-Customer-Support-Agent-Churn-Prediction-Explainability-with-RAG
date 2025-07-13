import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

data = pd.read_csv("Processed_Data.csv")

X = data.drop('Churn', axis = 1).to_numpy()
y = data['Churn'].to_numpy()

X_tensor = torch.tensor(X, dtype= torch.float32)
y_tensor = torch.tensor(y, dtype = torch.float32).unsqueeze(1) # using unsqueeze 1 to ensure same shape (n, 1)

dataset = TensorDataset(X_tensor, y_tensor)

train_loader = DataLoader(dataset, batch_size= 40, shuffle= True)

class ChurnPredictor(nn.Module):

  def __init__(self):
    super().__init__()

    # Defining the input layers. 40 inputs because we have 40 columns in our processed data(excluding target), and 64 is just a random number we took, for output.

    self.layer1 = nn.Linear(40, 64)
    self.relu1 = nn.ReLU()                                # using ReLU() activation function, for the hidden layer to add non-linearity.
    self.layer2 = nn.Linear(64, 32)
    self.relu2 = nn.ReLU()

    # Defining the output layer.

    self.output = nn.Linear(32, 1)    # using 32 as inputs for the output layer, and to receive 1 output value.

    self.sigmoid = nn.Sigmoid()                   #### using sigmoid as activation function of output layer since predicting churn or not is a binary classification problem (0 or 1)


  def forward(self, input1):       # forward-pass

    output1 = self.layer1(input1)         # passing input through layer 1.
    activation1 = self.relu1(output1)     ## now, we will pass output1 through activation function for linearity
    output2 = self.layer2(activation1)    ### the output of first layer becomes input of 2nd layer
    activation2 = self.relu2(output2)     #### output of 2nd layer becomes input of the 3rd layer. It is the main output layer. Also called logit.
    logit = self.output(activation2)     # calculating logit/final output
    predictions = self.sigmoid(logit)             # calculating probability of churn(0), not churn(1)

    return predictions


churnpredictor = ChurnPredictor()
optimizer = torch.optim.Adam(churnpredictor.parameters(),lr=0.001)
loss_fn = nn.BCELoss()

epoch = 0

for epoch in range(100):

  for inputs, actuals in train_loader:

    # STEP 1: Making a prediction
    prediction = churnpredictor(inputs)
    print(torch.min(prediction), torch.max(prediction))
    print(torch.unique(actuals))

    # STEP 2: Calculating loss
    loss = loss_fn(prediction, actuals)

    # STEP 3: ZEROING ON GRADIENTS, otherwise gradients will add up in each epoch
    optimizer.zero_grad()

    # STEP 4: BACKPROPAGATION (making the network learn by itself)
    loss.backward()

    # STEP 5: UPDATE WEIGHTS
    optimizer.step()

    #print(f"Loss:{loss}")
  #print(f"Epoch{epoch+1}")





