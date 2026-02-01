# Double Convolution
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

# ResNet Encoder
class ResNetEncoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.s1_conv1 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.s1_bn1 = nn.BatchNorm2d(64)
        self.s1_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.s1_bn2 = nn.BatchNorm2d(64)

        self.s2_conv1 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
        self.s2_bn1 = nn.BatchNorm2d(128)
        self.s2_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.s2_bn2 = nn.BatchNorm2d(128)
        self.s2_skip = nn.Conv2d(64, 128, 1, 2, bias=False)

        self.s3_conv1 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)
        self.s3_bn1 = nn.BatchNorm2d(256)
        self.s3_conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.s3_bn2 = nn.BatchNorm2d(256)
        self.s3_skip = nn.Conv2d(128, 256, 1, 2, bias=False)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x0 = x

        x = self.maxpool(x)
        y = self.relu(self.s1_bn1(self.s1_conv1(x)))
        y = self.s1_bn2(self.s1_conv2(y))
        x = self.relu(y + x)
        x1 = x

        y = self.relu(self.s2_bn1(self.s2_conv1(x)))
        y = self.s2_bn2(self.s2_conv2(y))
        x = self.relu(y + self.s2_skip(x))
        x2 = x

        y = self.relu(self.s3_bn1(self.s3_conv1(x)))
        y = self.s3_bn2(self.s3_conv2(y))
        x = self.relu(y + self.s3_skip(x))
        x3 = x

        return x0, x1, x2, x3

# UNet
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()

        self.encoder = ResNetEncoder(in_channels)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = DoubleConv(384, 128)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = DoubleConv(192, 64)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = DoubleConv(128, 64)
        self.up0 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec0 = DoubleConv(64, 64)

        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        ec0, ec1, ec2, ec3 = self.encoder(x)

        up3 = self.up3(ec3)
        dec3 = self.dec3(torch.cat([up3, ec2], dim=1))
        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([up2, ec1], dim=1))
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, ec0], dim=1))
        up0 = self.up0(dec1)
        dec0 = self.dec0(up0)

        out = self.out(dec0)

        return out