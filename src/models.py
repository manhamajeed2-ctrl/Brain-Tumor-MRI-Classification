"""Model definitions: Custom CNN and ResNet-from-scratch."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    """Lightweight custom CNN designed for 224x224 MRI images.
    
    Architecture:
    - 5 convolutional blocks (Conv -> BN -> ReLU -> MaxPool)
    - AdaptiveAvgPool
    - Fully connected head with Dropout
    """

    def __init__(self, num_classes: int = 4, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 5: 14 -> 7
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """Basic residual block (ResNet-18 style)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetFromScratch(nn.Module):
    """Simplified ResNet-18 inspired network trained from scratch.
    
    Suitable for medical imaging when pretrained weights are unavailable.
    """

    def __init__(self, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.in_channels = 64

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def get_model(name: str, num_classes: int = 4, pretrained: bool = False):
    """Factory function.
    
    name: 'custom_cnn' | 'resnet_scratch' | 'mobilenet_v2' | 'resnet50' | 'resnet18' | 'efficientnet_b0'
    pretrained only works when torchvision is installed.
    """
    name = name.lower()
    if name == "custom_cnn":
        return CustomCNN(num_classes=num_classes)
    elif name == "resnet_scratch":
        return ResNetFromScratch(num_classes=num_classes)
    elif name in ("mobilenet_v2", "resnet50", "resnet18", "efficientnet_b0"):
        try:
            import torchvision.models as models
            if name == "mobilenet_v2":
                weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
                model = models.mobilenet_v2(weights=weights)
                model.classifier[1] = nn.Linear(model.last_channel, num_classes)
            elif name == "resnet50":
                weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
                model = models.resnet50(weights=weights)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif name == "resnet18":
                weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                model = models.resnet18(weights=weights)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif name == "efficientnet_b0":
                weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
                model = models.efficientnet_b0(weights=weights)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            return model
        except Exception as e:
            raise RuntimeError(
                f"Transfer learning model '{name}' requires torchvision. "
                f"Install it with: pip install torchvision. Original error: {e}"
            )
    else:
        raise ValueError(f"Unknown model name: {name}")
