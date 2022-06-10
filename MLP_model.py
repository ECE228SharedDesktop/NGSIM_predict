import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Multi-layer perceptron model. Modified from code Lawson submitted for MLP homework in ECE 228
"""

def build_mlp(
          input_size,
          output_size,
          n_layers,
          size):
    """
    Args:
        input_size: int, dim(inputs)
        output_size: int, dim(outputs)
        n_layers: int, number of hidden layers
        size: int, number of nodes in each hidden layer
    Returns:
        An instance of (a subclass of) nn.Module representing the network.
    """
    class NN(nn.Module):
        def __init__(self, input_size, output_size, n_layers, size):
            super(NN, self).__init__()
            self.ourNN = nn.Sequential() # Make sequential module container
            self.ourNN.add_module("Hidden_0", nn.Linear(input_size,size,bias=True)) # Layer 1
            self.ourNN.add_module("ReLU_0", nn.ReLU(inplace=False))
            for n in range(1,n_layers):
                self.ourNN.add_module("Hidden_"+str(n), nn.Linear(size, size, bias=True))
                self.ourNN.add_module("ReLU_"+str(n), nn.ReLU(inplace=False))
            self.ourNN.add_module("Output_layer",nn.Linear(size, output_size, bias=True))# Last layer
            self.ourNN.to(device)
        def forward(self, x):
            x=self.ourNN(x)
            return x
    return NN(input_size, output_size, n_layers, size)

def np2torch(x, cast_double_to_float=True):
    """
    Utility function that accepts a numpy array and does the following:
        1. Convert to torch tensor
        2. Move it to the GPU (if CUDA is available)
        3. Optionally casts float64 to float32 (torch is picky about types)
    """
    x = torch.from_numpy(x).to(device)
    if cast_double_to_float and x.dtype is torch.float64:
        x = x.float()

    return x

def send_all_to_device(list,device):
    for thing in list:
        thing.to(device)


def train_on_data(x, y, num_iterations, n_layers, size, num_channels, device):
    """
    :param x: inputs to MLP
    :param y: desired outputs of MLP
    :param n_layers: number of hidden layers + ReLU
    :param size: width of MLP layers
    :param num_channels: number of channels in input data
    :param num_iterations: number of training iterations
    :return: tuple (MLP network object, array of training losses)
    """
    input_size = 13
    output_size = input_size
    n_layers = 13
    size = 30
    num_channels = 1

    input_data = np.random.random((input_size))
    target_data = np.random.random((output_size))
    print("input_data"+str(input_data))
    print("output_data:"+str(target_data))

    input_data = np2torch(input_data)
    target_data = np2torch(target_data)

    model = build_mlp(input_size, output_size, n_layers, size) # model
    criterion = nn.MSELoss(reduction="none") # loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # optimizer
    model.train() #sets model to train mode (dropout enabled)
    torch.set_grad_enabled(True) # L5kit does this, and I don't know why

    progress_bar = tqdm(range(num_iterations))#cfg["train_params"]["max_num_steps"]))
    losses_train = []

    for _ in progress_bar:
        model.eval()
        outputs = torch.reshape(model.forward(input_data.to(device)),target_data.size())
        loss = (criterion(outputs, target_data)).mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_train.append(loss.item())
        progress_bar.set_description(f"loss: {loss.item()} loss(avg ): {np.mean(losses_train)}")

    return (model, losses_train)

    # plt.figure()
    # plt.plot(np.log10(np.array(losses_train)))
    # plt.xlabel("Training iteration")
    # plt.ylabel("Log_10(Loss)")
    # plt.title("Loss per Training")
    # plt.show()