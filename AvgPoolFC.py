import torch
import numpy as np

class AvgFC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nerual_net = torch.nn.Sequential(
            torch.nn.Linear(in_features=256,
                            out_features=128,
                            bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=128,
                            out_features=14,
                            bias=True))

    def forward(self,x):
        output = self.nerual_net(x)

        return output



def main():
    model = AvgFC()
    pytorch_total_params = np.sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model parameter num = {}'.format(pytorch_total_params))

if __name__ == '__main__':
    main()