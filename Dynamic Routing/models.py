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
    def __init__(self, image_shape, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size = 9, padding = 4)
        self.relu1 = nn.ReLU(inplace = True)
        self.primary_capsule = PrimaryCapsule(256, 16, 8)
        # shape1 = get_subsequent_shape(image_shape, 9, 2)

        self.down_cap1 = ConvCapsule((10, 10), 2, 8, 4, 16, 3, 2, 1)
        self.up_cap1 = ConvCapsule((10, 10), 4, 16, 2, 8, 3, 1, 1)

        self.up_conv = nn.ConvTranspose2d(32, 1, kernel_size = 9, stride = 2, output_padding = 1)

    def up(self, x, scale_factor):
        # N x H x W x C x d
        in_dim = x.shape[-1]
        in_caps = x.shape[-2]
        h = x.shape[1]
        w = x.shape[2]
        x = x.permute(0, 3, 4, 1, 2)
        x = x.reshape(x.shape[0], -1, h, w)
        x = F.interpolate(x, scale_factor = scale_factor).reshape(x.shape[0], in_caps, in_dim, h * scale_factor, w * scale_factor)
        x = x.permute(0, 3, 4, 1, 2)
        return x
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        primary_capsule = self.primary_capsule(x)
        out_down1 = self.down_cap1(primary_capsule)
        out_up1 = self.up_cap1(self.up(out_down1, 2))
        out = torch.cat((primary_capsule, out_up1), dim = -1)
        out = out.view(out.shape[0], out.shape[1], out.shape[2], -1)
        out = out.permute(0, 3, 1, 2)
        out = self.up_conv(out)

        return torch.sigmoid(out)