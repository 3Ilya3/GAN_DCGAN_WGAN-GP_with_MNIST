import torch.nn as nn
import torch.nn.functional as F

class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()

        # Сверточные слои с ReLU активацией и BatchNorm
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 1x28x28 -> 32x28x28
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32x28x28 -> 64x28x28
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64x28x28 -> 128x28x28
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        
        # Max Pooling для уменьшения размера изображения
        self.pool = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14

        # Полносвязные слои
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 14 * 14, 1024),  # 128x7x7 -> 1024
            nn.ReLU(True)
        )
        
        self.output = nn.Sequential(
            nn.Linear(1024, 10)  # 1024 -> 10 (для 10 классов MNIST)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.pool(x)  # Применяем MaxPool после сверточных слоев
        
        x = x.view(-1, 128 * 14 * 14)  # Преобразуем в одномерный вектор
        
        x = self.fc1(x)
        x = self.output(x)
        
        #return F.log_softmax(x, dim=1)  # Логарифм вероятностей для многоклассовой классификации

        # F.cross_entropy автоматически применяет log_softmax
        return x

