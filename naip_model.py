import torch.nn.functional as F
import os
import rasterio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from torch._C import memory_format
import numpy as np

class WaterBodyTIFDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.files = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        base, txt = os.path.splitext(img_name)

        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)
            image = image / 255.0

        with rasterio.open(mask_path) as src:
            mask = src.read().astype(np.int64)

        image, mask = torch.from_numpy(image), torch.from_numpy(mask).squeeze(0)
        
        mask = (mask > 0).long()

        return image, mask

class NAIPInferenceDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.files = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)
            image = image / 255.0

        image = torch.from_numpy(image)

        return image, img_name

train_dataset = WaterBodyTIFDataset("/naip/naip_train_input", "/naip/naip_train_mask")
val_dataset = NAIPInferenceDataset("/naip/naip_test_input")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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

print(f"Train dataset length: {len(train_loader.dataset)}, Val dataset length: {len(val_loader.dataset)}")

model = UNet(in_channels=4).to(device).to(memory_format=torch.channels_last)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 10
train_losses = []
best_train = float("inf")
scaler = torch.amp.GradScaler("cuda")

for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(device, memory_format=torch.channels_last, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            preds = model(xb)
            loss = criterion(preds, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)
    if train_loss < best_train:
        torch.save(model.state_dict(), "best_unet_naip.pth")
        best_train = train_loss

    print(f"Epoch {epoch+1}: Train Loss: {train_loss}")

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("train_test_curve_naip.png", dpi=300, bbox_inches="tight")
plt.show()

model = UNet(in_channels=4).to(device).to(memory_format=torch.channels_last)
state_dict = torch.load("best_unet_naip.pth", map_location=device)
model.load_state_dict(state_dict)

model.eval()
xb, img_name = next(iter(val_loader))
xb = xb.to(device, memory_format=torch.channels_last)

with torch.no_grad():
    with torch.amp.autocast("cuda"):
        logits = model(xb)
        preds = logits.argmax(dim=1)

plt.figure()
for index in range(3):
    img = xb[index][:3].permute(1,2,0)
    pred_mask = preds[index]

    plt.subplot(3, 3, index * 3 + 1)
    plt.title("Input Image")
    plt.imshow(img.cpu())
    plt.axis("off")

    plt.subplot(3, 3, index * 3 + 3)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask.cpu(), cmap="gray")
    plt.axis("off")

plt.savefig("sample_val_preds_naip.png", dpi=300, bbox_inches="tight")
plt.show()