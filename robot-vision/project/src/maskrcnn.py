import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

# Load dataset - change for FLC/NABirds/CUB/RPC

# num_classes = 2 # FLC
# num_classes = 400 # NABirds
# num_classes = 200 # CUB
# num_classes = 50000~ # RPC


# Load ResNet-50 with FPN as the backbone, pretrained on ImageNet
backbone = resnet_fpn_backbone('resnet50', pretrained=True)

# Create the Mask R-CNN model with the specified backbone
model = MaskRCNN(backbone, num_classes=num_classes)