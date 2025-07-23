import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from config import Config


class PyramidPooling(nn.Module):
    """PyramidPooling"""

    def __init__(self, in_channels, pool_sizes, norm_layer=nn.BatchNorm2d):
        super(PyramidPooling, self).__init__()
        self.pool_sizes = pool_sizes

        self.stages = nn.ModuleList([
            self._make_stage(in_channels, size, norm_layer)
            for size in pool_sizes
        ])

    def _make_stage(self, in_channels, pool_size, norm_layer):
        """create single stage"""
        prior = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        conv = nn.Conv2d(in_channels, in_channels // len(self.pool_sizes),
                         kernel_size=1, bias=False)
        bn = norm_layer(in_channels // len(self.pool_sizes))
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(stage(feats), size=(h, w), mode='bilinear',
                                align_corners=False) for stage in self.stages] + [feats]
        return torch.cat(priors, 1)


class PSPNet(nn.Module):
    def __init__(self, num_classes=2, backbone='resnet50', pool_sizes=(1, 2, 3, 6), in_channels=3):
        super(PSPNet, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        resnet = models.resnet50(pretrained=True)
        self.backbone_channels = 2048

        # support in_channels
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation = (2, 2)
                m.padding = (2, 2)
                m.stride = (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.psp = PyramidPooling(self.backbone_channels, pool_sizes)
        self.final = nn.Sequential(
            nn.Conv2d(self.backbone_channels * 2, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )
        self.auxiliary = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.size()
        x = self.conv1(x)  # Define conv1
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_aux = self.layer3(x)
        x = self.layer4(x_aux)
        x = self.psp(x)
        x = self.final(x)
        x = F.interpolate(x, size=input_size[2:], mode='bilinear', align_corners=False)
        if self.training:
            aux = self.auxiliary(x_aux)
            aux = F.interpolate(aux, size=input_size[2:], mode='bilinear', align_corners=False)
            return x, aux
        else:
            return x


class PSPNetLoss(nn.Module):
    """Loss function for PSPNet"""

    def __init__(self, aux_weight=0.4):
        super(PSPNetLoss, self).__init__()
        self.aux_weight = aux_weight
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        if isinstance(outputs, tuple):
            main_out, aux_out = outputs
            main_loss = self.criterion(main_out, targets)
            aux_loss = self.criterion(aux_out, targets)
            return main_loss + self.aux_weight * aux_loss
        else:
            return self.criterion(outputs, targets)


def create_model(config):
    """Create model"""
    model = PSPNet(
        num_classes=config.NUM_CLASSES,
        backbone=config.BACKBONE,
        pool_sizes=config.PSP_SIZE
    )
    return model


def count_parameters(model):
    """Calculate number of parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    config = Config()
    model = create_model(config)
    print(f"parameter numbers: {count_parameters(model):,}")

    # Test forward propagation
    dummy_input = torch.randn(2, 3, 256, 256)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        print(f"shape: {output.shape}")