from torch.utils.data import DataLoader
from torchvision import transforms, datasets  

mnist_transforms = transforms.Compose([transforms.ToTensor(),                             
                                       transforms.Normalize(mean=[0.1307], std=[0.3081])]) 
             
# Каждый поток отвечает за загрузку одного батча. Если num_workers=4, то 4 потока загружают 4 батча данных параллельно, но модель обрабатывает их последовательно                                                                                      
data = datasets.MNIST(root='./dataset', train=True, download=True, transform=mnist_transforms)

# Загружаем обучающие и тестовые данные
train_data = datasets.MNIST(root='./dataset', train=True, download=True, transform=mnist_transforms)
test_data = datasets.MNIST(root='./dataset', train=False, download=True, transform=mnist_transforms)

# Даталоадеры для обучающих и тестовых данных
train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4, persistent_workers=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=4, persistent_workers=True)

