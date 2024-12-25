from torch.utils.data import DataLoader
from torchvision import transforms, datasets  

# Каждый поток отвечает за загрузку одного батча. Если num_workers=4, то 4 потока загружают 4 батча данных параллельно, но модель обрабатывает их последовательно
# Изображение в MNIST имеет размер 28x28=784 пикселей
mnist_transforms = transforms.Compose([transforms.ToTensor(),                            # Изображение в тензор
                                       transforms.Normalize(mean=0.5, std=0.5)])         # Нормализация пикселей тензора
                                                                      
data = datasets.MNIST(root='./dataset', train=True, download=True, transform=mnist_transforms)

mnist_dataloader = DataLoader(data, batch_size=128, shuffle=True, num_workers=4, persistent_workers=True)    