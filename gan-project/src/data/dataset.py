from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_mnist_dataloader(batch_size=32, image_path='./', download=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    
    mnist_dataset = datasets.MNIST(root=image_path, train=True, transform=transform, download=download)
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader