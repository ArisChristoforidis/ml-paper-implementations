import PIL
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from model import UNet

BATCH_SIZE = 32
PIN_MEMORY = True
EPOCHS = 100
LR = 1E-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# From here: https://stackoverflow.com/questions/71674595/runtimeerror-only-batches-of-spatial-targets-supported-3d-tensors-but-got-tar
VOC_COLORMAP = [
    [0, 0, 0],  # Background
    [128, 0, 0],  # Aeroplane
    [0, 128, 0],  # Bicycle
    [128, 128, 0],  # Bird
    [0, 0, 128],  # Boat
    [128, 0, 128],  # Bottle
    [0, 128, 128],  # Bus
    [128, 128, 128],  # Car
    [64, 0, 0],  # Cat
    [192, 0, 0],  # Chair
    [64, 128, 0],  # Cow
    [192, 128, 0],  # Diningtable
    [64, 0, 128],  # Dog
    [192, 0, 128],  # Horse
    [64, 128, 128],  # Motorbike
    [192, 128, 128],  # Person
    [0, 64, 0],  # Pottedplant
    [128, 64, 0],  # Sheep
    [0, 192, 0],  # Sofa
    [128, 192, 0],  # Train
    [0, 64, 128],  # tvmonitor
    [224, 224, 192],  # tvmonitor
]

color2idx = lambda x: VOC_COLORMAP.index(x)

def create_mask(image: torch.Tensor):
    """
    Creates a mask given an argmax tensor (from the model)
    Args:
        image (torch.Tensor): The model output.

    Returns:
        PIL.Image: The mask image.
    """
    H, W = image.shape
    out = PIL.Image.new(mode='RGB', size=[W, H])
    for h in range(H):
        for w in range(W):

            out.putpixel((w,h), tuple(VOC_COLORMAP[image[h, w]]))
    return out

class EncodeOutput:
    def __call__(self, image):
        W, H = image.size
        palette = torch.tensor(image.getpalette()).reshape(256, 3)
        image = transforms.functional.pil_to_tensor(image)
        unique_values = torch.unique(image).tolist()
        mapping = {v: color2idx(palette[v].tolist()) for v in unique_values}
        image.apply_(mapping.get)
        return image

def main():

    transforms_input = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
    transforms_target = transforms.Compose([transforms.Resize((256, 256)), EncodeOutput()])
    train_data = datasets.VOCSegmentation('voc_data', transform=transforms_input, target_transform=transforms_target, image_set='train', download=True)
    test_data = datasets.VOCSegmentation('voc_data', transform=transforms_input, target_transform=transforms_target, image_set='val', download=True)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY)
    

    model = UNet(3, 22, restore_dims=True)
    model = model.to(DEVICE)

    optimizer = Adam(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        mean_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE, dtype=torch.long)
            _, loss = model(inputs, targets)

            mean_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"unet_{epoch}.pt")
        mean_loss /= len(train_loader)
        print(f"Epoch {epoch + 1} train loss: {mean_loss:.3f}")

if __name__ == "__main__":
    main()