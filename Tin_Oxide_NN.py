import torch
from torch import nn as nn
import Tin_Oxide_Data_Processing as d

#create test and train variables
dp = d.Data_Processor()
dp.Import_From_CSV('Tin_Oxide_Optimization.csv')
print(f"Group Names: {list(dp.grouped_data.keys())}")
dp.Select_Target(['Glass-FTO-Sn02-Perovskite-Spiro-OMeTAD-Gold'])
dp.Clean_Data()
dp.Drop_Bad_Pixels("Average")
dp.Sort_By_Device()
dp.Parameter_Encoder()
train_input, test_input, train_output, test_output = dp.Generate_Sets(.8)
    #-------------------------------------------------------------------------------------------------

tensor_input = torch.Tensor(train_input.values)
tensor_output = torch.Tensor(train_output.values)

print(tensor_input)
print(tensor_output)

x = tensor_input
y = tensor_output

n_samples, n_features = x.shape
print (x.shape)
input_size = n_features
output_size = 1

#model prediction
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x = torch.flatten(x, 1)
        # x = F.relu(self.fc1(x))
        x = self.fc1(x)
        return x

model = LinearRegression(input_size, output_size)
#training
learning_rate = .000000001
n_iters = 1000

#calc loss
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = model(x)
    #loss
    l = loss(y_pred, y)

    #gradients
    l.backward()

    #update weights
    optimizer.step()
    #zero gradients
    optimizer.zero_grad()

    if epoch % 1 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')
print(f'Prediciton after training: f(5) = {model(x):.3f}')