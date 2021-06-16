from layers import *

class CapNet(nn.Module):
    def __init__(self, in_channels = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size = 9, stride = 1)
        self.relu1 = nn.ReLU()
        self.primary_cap = PrimaryCapsule(256, 256, 8)
        self.digit_cap = DigitCapsule((32, 6, 6), 8, 10, 16)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.primary_cap(x)
        x = self.digit_cap(x)
        return x
        
class UCapsuleNet(nn.Module):
    """
        A simple architecture for MNIST
    """

    def __init__(self, image_shape, in_channels):
        def get_subsequent_shape(shape, ks, s):
            def fn(x):
                return (x - ks) // s + 1
            return fn(shape[0]), fn(shape[1])
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size = 9, padding = 4)
        self.relu1 = nn.ReLU(inplace = True)
        self.primary_capsule = PrimaryCapsule(256, 16, 8)
        shape1= get_subsequent_shape(image_shape, 9, 2)
        self.conv_cap1 = ConvCapsule((2, *shape1), 8, (4, 5, 5), 16)
        self.up_cap1 = ConvCapsule((4, 5, 5), 16, (2, *shape1), 8)
        self.up_conv = nn.ConvTranspose2d(32, 1, kernel_size = 9, stride = 2, output_padding = 1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        primary_capsule = self.primary_capsule(x)
        out_conv1 = self.conv_cap1(primary_capsule)
        out_up1 = self.up_cap1(out_conv1)
        out = torch.cat((primary_capsule, out_up1), dim = -1)
        out = out.view(-1, 32, 10, 10)
        out = self.up_conv(out)
        return torch.sigmoid(out)