from networks import *
import torch

# run network of choice
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running with ' + str(device) + '...')

    continuous_discrete(device)
