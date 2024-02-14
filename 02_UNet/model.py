import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms.functional import center_crop

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ConvolutionBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        A double convolution block that is used both on the contracting and the
        expanding blocks.

        Args:
            in_channels (int): The number of input channels from the previous layer.
            out_channels (int): The number of output channels for the convolution.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class ContractingBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        An contracting block module for the U-Net.

        Args:
            in_channels (int): The number of input channels from the previous layer.
            out_channels (int): The number of output channels for the convolution.
        """
        super().__init__()
        self.conv = ConvolutionBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Used for the "copy and crop" operation.
        cc_out = self.conv(x)
        pool = self.pool(cc_out)
        return cc_out, pool

class ExpandingBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        """
        An expanding block module for the U-Net.

        Args:
            in_channels (int): The number of input channels from the previous layer.
            out_channels (int): The number of output channels for the convolution.
        """
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvolutionBlock(in_channels, out_channels)

    def forward(self, cc_input, x):
        x = self.upsample(x)
        # We perform a crop here, when we need it. x is (B,C,H,W) so we take the 
        # last two dims.
        crop_size = x.shape[-2], x.shape[-1]
        cc_input = center_crop(cc_input, crop_size)
        x = torch.cat([x, cc_input], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, restore_dims: bool = False):
        """
        A U-Net model.

        Args:
            in_channels (int): The number of input channels for the model (3 for RGB images)
            out_channels (int): The number of output channels (The number of classes for a multiclass dataset)
            restore_dims (bool): True if you want to restore the image dimensions for the output.
        """
        super().__init__()
        # Contains the input and output channels for each double convolution for
        # the downsampling and upsampling.
        self.restore_dims = restore_dims
        contracting_conv_channels = [(in_channels, 64), (64, 128), (128, 256), (256, 512)]
        expanding_conv_channels = [(1024, 512), (512, 256), (256, 128), (128, 64)]
        self.contracting_blocks = nn.ModuleList([ContractingBlock(in_c, out_c) for (in_c, out_c) in contracting_conv_channels])
        self.transition_conv = ConvolutionBlock(in_channels=512, out_channels=1024)
        self.expanding_blocks = nn.ModuleList([ExpandingBlock(in_c, out_c) for (in_c, out_c) in expanding_conv_channels])
        self.final_conv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)
        # This is dataset specific, the 0th class is the background and it is ignored 
        # for the loss calculation because the model tends to overfit to black pixels
        # otherwise.
        self.loss = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, x, targets=None):
        original_dims = (x.shape[-2], x.shape[-1])

        copy_crops = []
        for block in self.contracting_blocks:
            cc, x = block(x)
            copy_crops.append(cc)
        x = self.transition_conv(x)

        copy_crops.reverse()
        for idx, block in enumerate(self.expanding_blocks):
            x = block(copy_crops[idx], x)

        x = self.final_conv(x)

        # Restore original dims
        if self.restore_dims:
            x = F.interpolate(x, original_dims)

        if targets == None:
            # Prediction
            x = torch.argmax(x, dim=1)
            return x
        else:
            # Training
            targets = targets.squeeze(dim=1)
            loss = self.loss(x, targets)
            return x, loss