from torch.utils.data import DataLoader
from torchvision import transforms, datasets  

# Преобразования будут применяться по очереди ко всем изображениям по мере необходимости. Так как у нас 4 потока (num_workers=4), параллельно преобразуем 4 изображения
# Изображение в MNIST имеет размер 28x28=784 пикселей
mnist_transforms = transforms.Compose([transforms.ToTensor(),                               # Изображение в тензор
                                       transforms.Normalize(mean=0.5, std=0.5)])             # Нормализация пикселей тензора
                                                                                                                                                                                                                                                    
data = datasets.MNIST(root='./dataset', train=True, download=True, transform=mnist_transforms)

mnist_dataloader = DataLoader(data, batch_size=128, shuffle=True, num_workers=4, persistent_workers=True)            # Обьект который позволяет итерироваться по data. Элемент data это кортеж(тензор(вектор), метка(0-9))