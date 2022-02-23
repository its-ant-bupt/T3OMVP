import sys
import torch

def get_parms(argv):
    for i in range(len(argv)):
        print(argv[i])

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.rand(2,1,5).to(device)
    a.detach().numpy()
    print(a)
