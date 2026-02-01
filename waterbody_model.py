from torch.utils.data import Dataset, DataLoader, random_split
import os
from PIL import Image
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch._C import memory_format
from torchvision.models import resnet18, ResNet18_Weights
import json

class WaterBodyDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform, mask_transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.transform = transform
        self.mask_transform = mask_transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image_name = self.images[idx]

        img_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.transform(image)
        mask = self.mask_transform(mask)
        
        return image, mask

transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
mask_transform = transforms.Compose([transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST), transforms.PILToTensor(), transforms.Lambda(lambda x: (x > 0).long().squeeze(0))])

dataset = WaterBodyDataset("/waterbody-dataset/images", "/waterbody-dataset/masks", transform, mask_transform)
train_size = int(0.8 * len(dataset)) + 1
val_size = int(0.2 * len(dataset))

train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)

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
    def __init__(self):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.pool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        #self.layer4 = base.layer4

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        #x4 = self.layer4(x1)

        return x0, x1, x2, x3


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()

        self.encoder = ResNetEncoder()

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

model = UNet().to(device).to(memory_format=torch.channels_last)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 20
train_losses = []
test_losses = []
best_test = float("inf")
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

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, memory_format=torch.channels_last, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                preds = model(xb)
                loss = criterion(preds, yb)
            test_loss += loss.item() * xb.size(0)
    test_loss /= len(val_loader.dataset)
    test_losses.append(test_loss)
    if test_loss < best_test:
        best_test = test_loss
        torch.save(model.state_dict(), "best_unet_waterbody.pth")

    print(f"Epoch {epoch+1}:, Train Loss: {train_loss}, Test Loss: {test_loss}")

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("train_test_curve_waterbody.png", dpi=300, bbox_inches="tight")
plt.show()

model = UNet().to(device).to(memory_format=torch.channels_last)
state_dict = torch.load("best_unet_waterbody.pth", map_location=device)
model.load_state_dict(state_dict)

model.eval()
xb, yb = next(iter(val_loader))
xb = xb.to(device, memory_format=torch.channels_last)
yb = yb.to(device)

with torch.no_grad():
    with torch.amp.autocast("cuda"):
        logits = model(xb)
        preds = logits.argmax(dim=1)

plt.figure()
for index in range(3):
    img = xb[index].permute(1,2,0)
    true_mask = yb[index]
    pred_mask = preds[index]

    plt.subplot(3, 3, index * 3 + 1)
    plt.title("Input Image")
    plt.imshow(img.cpu())
    plt.axis("off")

    plt.subplot(3, 3, index * 3 + 2)
    plt.title("Ground Truth Mask")
    plt.imshow(true_mask.cpu(), cmap="gray")
    plt.axis("off")

    plt.subplot(3, 3, index * 3 + 3)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask.cpu(), cmap="gray")
    plt.axis("off")

plt.savefig("sample_val_preds_waterbody.png", dpi=300, bbox_inches="tight")
plt.show()