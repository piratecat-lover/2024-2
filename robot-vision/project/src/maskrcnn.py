import os
import torch
import torchvision
from torchvision.io import read_image
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from PIL import Image
import pandas as pd

# Load dataset for CUB_200_2011

# Caltech UCSD Birds 2011
base_dir = './project/data/CUB_200_2011'
num_classes = 200

class CUBDataset(Dataset):
    def __init__(self, base_dir, metadata, is_train=True, transform=None):
        self.base_dir = base_dir
        self.metadata = metadata[metadata["is_train"] == (1 if is_train else 0)]
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        try:
            data = self.metadata.iloc[idx]
            img_path = os.path.join(self.base_dir, 'images', data['filepath'])
            image = Image.open(img_path).convert("RGB")
            
            x, y, width, height = data['x'], data['y'], data['width'], data['height']
            boxes = torch.tensor([[x, y, x + width, y + height]], dtype=torch.float32)

            labels = torch.tensor([data['class_id']-1], dtype=torch.int64)

            # Create a dummy mask filled with zeros, same height and width as the image
            mask = torch.zeros((1, int(image.height), int(image.width)), dtype=torch.uint8)
            target = {"boxes": boxes, "labels": labels, "masks": mask}

            if self.transform:
                image = self.transform(image)

            return image, target
        
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None  # return None if the image fails to load

# Collate function for DataLoader
def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    return tuple(zip(*batch))

# Set up main function to hold everything
def main():
    # Load metadata
    images = pd.read_csv(os.path.join(base_dir, 'images.txt'), sep=" ", names=["img_id", "filepath"])
    bboxes = pd.read_csv(os.path.join(base_dir, 'bounding_boxes.txt'), sep=" ", names=["img_id", "x", "y", "width", "height"])
    labels = pd.read_csv(os.path.join(base_dir, 'image_class_labels.txt'), sep=" ", names=["img_id", "class_id"])
    split = pd.read_csv(os.path.join(base_dir, 'train_test_split.txt'), sep=" ", names=["img_id", "is_train"])
    metadata = images.merge(bboxes, on="img_id").merge(labels, on="img_id").merge(split, on="img_id")

    # Set up the dataset and dataloaders
    train_dataset = CUBDataset(base_dir, metadata, is_train=True, transform=ToTensor())
    test_dataset = CUBDataset(base_dir, metadata, is_train=False, transform=ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=8, collate_fn=collate_fn)

    # Define Mask R-CNN with ResNet-50 and FPN backbone
    num_classes = 200  # For CUB dataset
    backbone = resnet_fpn_backbone(backbone_name = 'resnet50', weights="IMAGENET1K_V1")
    model = MaskRCNN(backbone, num_classes=num_classes)

    # Training setup
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training loop
    num_epochs = 5
    model.train()
    for epoch in range(num_epochs):
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1}, Loss: {losses.item()}")

    # Save the model
    torch.save(model.state_dict(), "mask_rcnn_cub.pth")

    # Load the model for inference/testing
    model.eval()
    
    # Evaluate on test set
    all_outputs = []
    with torch.no_grad():
        for images, targets in test_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            all_outputs.extend(outputs)

    # Print sample output
    for i, output in enumerate(all_outputs[:3]):  # View some sample outputs
        print(f"Sample {i+1} output:", output)

if __name__ == "__main__":
    main()