import torch
import torch.nn as nn
import torch.nn.functional as F 

class ImageEncoder(nn.Module):
    def __init__(self):
        """
        Simple Image Encoder based on a Convenet

        """
        super().__init__()

        self.conv1 = nn.Conv2d(4, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(72, 30)
        self.fc2 = nn.Linear(30, 10)

    def forward(self, image):
        """
        Parameters
        ----------
        color_image: 

        Returns
        -------
        hidden : torch.FloatTensor
            These are the final hidden state of the RNN for this batch,
            shape `(m, p) where `m` is the batch_size and `p` is
             the color dimensionality.
        output, hidden = self.rnn(color_seqs)
        return hidden

        """

        x = self.pool(F.relu(self.conv1(image)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.flatten(start_dim=1,end_dim=-1)
        return  x.unsqueeze(0)