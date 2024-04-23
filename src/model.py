import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from src.utils import DSC


class HUWindowAndScale:
    def __init__(self, lower=-1000, upper=0):
        """
        Windows data based on Hounsfield Units and normalizes it to [-1, 1].

        Args:
            lower (int): Lower bound in HU for windowing.
            upper (int): Upper bound in HU for windowing.
        """

        self.lower = lower
        self.upper = upper

    def __call__(self, img):
        img = torch.clamp(img, self.lower, self.upper)
        img = (img - self.lower) / (self.upper - self.lower)
        img = (img - 0.5) / 0.5
        return img


class SegmentationDataset(Dataset):
    """
    A custom dataset class for the segmentation task.

    Args:
        data (dict): A dictionary containing the data.
        split (str): The split to use, one of "train", "val", or "test".
        img_transform (callable): A function/transform to apply to the image data.
    """

    def __init__(self, data, split, img_transform=None):
        self.data = data[split]
        self.keys = list(self.data.keys())
        self.transform = img_transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        case_id = self.keys[index]
        img = torch.tensor(self.data[case_id]["img"]).float().unsqueeze(0)
        seg = torch.tensor(self.data[case_id]["seg"]).float().unsqueeze(0)

        # No rotation or flipping as transforms applied to only image
        if self.transform:
            img = self.transform(img)

        return img, seg, case_id


class CombinedLoss(nn.Module):
    def __init__(self, lambda_DSC, lambda_BCE):
        """
        Conditionally combines the Dice loss and Binary Cross Entropy loss.

        Args:
            lambda_DSC (float): Weight for the Dice loss.
            lambda_BCE (float): Weight for the Binary Cross Entropy loss.
        """

        super().__init__()
        self.lambda_DSC = lambda_DSC
        self.lambda_BCE = lambda_BCE

    def forward(self, pred, target):
        return (
            self.lambda_BCE * F.binary_cross_entropy_with_logits(pred, target)
            + self.lambda_DSC * self.dice_loss(pred, target)
        ).mean()

    def dice_loss(self, pred, target):
        return (1 - DSC(pred, target, binary=False)).mean()


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        """
        A simple UNet model for image segmentation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """

        super().__init__()

        # Encoder
        self.conv1 = self.conv_block(in_channels, 16)
        self.maxpool1 = self.maxpool()
        self.conv2 = self.conv_block(16, 32)
        self.maxpool2 = self.maxpool()
        self.conv3 = self.conv_block(32, 64)
        self.maxpool3 = self.maxpool()
        self.middle = self.conv_block(64, 128)

        # Decoder
        self.upsample3 = self.transposed_block(128, 64)
        self.upconv3 = self.conv_block(128, 64)
        self.upsample2 = self.transposed_block(64, 32)
        self.upconv2 = self.conv_block(64, 32)
        self.upsample1 = self.transposed_block(32, 16)
        self.upconv1 = self.conv_block(32, 16)

        self.final = nn.Conv2d(16, out_channels, kernel_size=1, stride=1, padding=0)

    def conv_block(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def transposed_block(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,
    ):
        return nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )

    def maxpool(self, dropout_rate=0.5, kernel_size=2, stride=2, padding=0):
        return nn.Sequential(
            nn.MaxPool2d(kernel_size, stride, padding), nn.Dropout2d(dropout_rate)
        )

    def forward(self, x):
        # Encode
        x1 = self.conv1(x)

        x2 = self.maxpool1(x1)
        x2 = self.conv2(x2)

        x3 = self.maxpool2(x2)
        x3 = self.conv3(x3)

        out = self.maxpool3(x3)
        out = self.middle(out)

        # Decode
        out = self.upsample3(out)
        out = torch.cat([out, x3], dim=1)
        out = self.upconv3(out)

        out = self.upsample2(out)
        out = torch.cat([out, x2], dim=1)
        out = self.upconv2(out)

        out = self.upsample1(out)
        out = torch.cat([out, x1], dim=1)
        out = self.upconv1(out)

        out = self.final(out)

        return out
