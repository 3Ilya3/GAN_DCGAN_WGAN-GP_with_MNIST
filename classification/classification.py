import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import model as m

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Classification():
    def __init__(self, learning_rate, model_path=None):
        self.learning_rate = learning_rate
        self.model = m.CNN_MNIST().to(device)
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.current_epoch = 0

        if model_path is not None:
            self.load_model(model_path)
        
        self.writer = SummaryWriter(log_dir='logs')

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_optimizer_state_dict': self.model_optimizer.state_dict(),
            'current_epoch': self.current_epoch
        }, path)
        print(f'Model saved as {path} at epoch {self.current_epoch}')

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model_optimizer.load_state_dict(checkpoint['model_optimizer_state_dict'])
        self.current_epoch = checkpoint['current_epoch'] + 1
        print(f'Model loaded from {path}, starting from epoch {self.current_epoch}')

    def train(self, dataloader, num_epochs):
      
        current_epoch = self.current_epoch
        total_epoch = num_epochs + current_epoch

        for epoch in range(current_epoch, total_epoch + 1):
          
            num_batches = 0
            last_batch = None
            last_predicted = None
          
            with tqdm(dataloader, desc=f"Epoch {epoch}/{total_epoch}", unit="batch") as pbar:
           
                for batch, targets in pbar: 
                    batch, targets = batch.to(device), targets.to(device)
                    num_batches += 1
                    # Шаг обучения
                    #loss = self.model_step(batch, targets)

                    self.model_optimizer.zero_grad()

                    outputs = self.model(batch)

                    # Логиты преобразуются в вероятности автоматически, модель не возвращает вероятности
                    loss = F.cross_entropy(outputs, targets)
            
                    loss.backward()
                    self.model_optimizer.step()

                    # Логирование потерь в TensorBoard
                    self.writer.add_scalar('Loss', loss.item(), epoch * len(dataloader) + num_batches)
                    
                    pbar.set_postfix(loss=loss.item())

                    last_batch = batch
                    _, last_predicted = torch.max(outputs, 1)

            if epoch % 20 == 0:
                self.current_epoch = epoch
                self.visualize_images_with_labels(epoch, last_batch, last_predicted, f'images_train_class/image_of_epoch_{epoch}.png')

            if epoch % 50 == 0:
                self.current_epoch = epoch
                self.save_model(f'models_class/model_{epoch}')
              
        self.current_epoch = total_epoch
        self.writer.close()

    def evaluate(self, dataloader):
        correct_predictions = 0
        total_samples = 0

        # Переводим модель в режим оценки (выключаем Dropout, BatchNorm и т.д.)
        self.model.eval()

        with torch.no_grad():  
            with tqdm(dataloader, desc="Evaluating", unit="batch") as pbar:
                for batch_idx, (batch, targets) in enumerate(pbar):
                    batch, targets = batch.to(device), targets.to(device)

                    # Получаем логиты из модели
                    outputs = self.model(batch)
                    
                    '''
                    Модель возвращает логиты, а не вероятности. Чем больше логит, тем больше вероятность. Поэтому здесь не преобразуем в вероятности.
                    max возвращает тензор (максимальное значение, индекс). Нам интересен индекс, он равен метке (классу).
                    ''' 

                    # Извлекаем индексы классов с максимальными логитами
                    _, predicted = torch.max(outputs, 1)

                    if batch_idx % 4 == 0:
                        self.visualize_images_with_labels(batch_idx, batch, predicted, f'images_evalute_class/image_of_batch_{batch_idx}.png')

                    # Считаем количество правильных предсказаний
                    correct_predictions += (predicted == targets).sum().item()

                    # Общее количество примеров
                    total_samples += targets.size(0)

                    pbar.set_postfix(accuracy=100 * correct_predictions / total_samples)

        
        accuracy = 100 * correct_predictions / total_samples

        print(f"Accuracy: {accuracy:.2f}%")

    def count_predicted_classes(self, dataloader):
    # Создаем словарь для подсчета предсказанных классов
        class_counts = {i: 0 for i in range(10)}

        # Переводим модель в режим оценки (выключаем Dropout, BatchNorm и т.д.)
        self.model.eval()

        with torch.no_grad():  # Отключаем градиенты для оценки
            with tqdm(dataloader, desc="Counting predictions", unit="batch") as pbar:
                for batch_idx, batch in enumerate(pbar):  
                    batch = batch.to(device)

                    # Получаем логиты из модели
                    outputs = self.model(batch)

                    # Извлекаем индексы классов с максимальными логитами (классы)
                    _, predicted = torch.max(outputs, 1)

                    # Подсчитываем, сколько раз был предсказан каждый класс
                    for class_id in predicted:
                        class_counts[class_id.item()] += 1

                    pbar.set_postfix(class_counts=class_counts)

        # Выводим статистику по предсказанным классам
        print("Predicted class counts:")
        for class_id, count in class_counts.items():
            print(f"Class {class_id}: {count} predictions")



    def visualize_images_with_labels(self, epoch, batch, predicted, path):
        nrow, ncol = 3, 8  
        fig = plt.figure(figsize=(ncol * 2, nrow * 2))  
        fig.suptitle(f'Epoch {epoch}', fontsize=30)

        for i in range(nrow):
            for j in range(ncol):
                img_idx = i * ncol + j
                img = batch[img_idx].cpu().detach()

                ax = plt.subplot(nrow, ncol, img_idx + 1)  
                ax.imshow(img, cmap='gray')  
                ax.axis('off')
                ax.set_title(f'Predicted: {predicted[img_idx].item()}')  

        plt.subplots_adjust(wspace=0.5, hspace=0.5)  
        plt.savefig(path)
        plt.close(fig)  



    