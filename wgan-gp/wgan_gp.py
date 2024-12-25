import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import wgan_gp_networks as m

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class WGAN_GP():

  def __init__(self, learning_rate, lambda_gp=10, model_path=None):
    self.learning_rate = learning_rate
    self.lambda_gp = lambda_gp  # Коэффициент градиентного штрафа
    self.generator = m.Generator().to(device)
    self.critic = m.Critic().to(device)
    '''
    После каждой эпохи мы генерируем 100 изображений (случайных шумовых векторов длиной 100 из стандартного нормального распределения. Тензор, состоящий из 100 тензоров).
    Сохраняем выходные изображения в списке test_progression, чтобы отслеживать и визуализировать прогресс генератора на протяжении обучения. 
    '''
    self.test_noises = torch.randn(100,m.nz, 1, 1, device=device)
    self.test_progression = []    

    # Оптимизаторы
    self.g_optimizer = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    self.c_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate, betas=(0.5, 0.9))

    self.current_epoch = 0

    if model_path is not None:
        self.load_model(model_path)

    self.writer = SummaryWriter(log_dir='logs')                             

  def forward(self, z):
    """
    Передача тензора (вектор шума) генератору.
    Генерация изображения.
    """
    return self.generator(z)

  def generator_step(self, x):
    
    z = torch.randn(x.shape[0], m.nz, 1, 1, device=device)

    generated_imgs = self.forward(z)

    fake_score = self.critic(generated_imgs)

    # Потеря генератора: стремимся увеличить оценку критика для поддельных данных
    g_loss = -torch.mean(fake_score)

    # Обратное распространение и обновление весов
    self.g_optimizer.zero_grad()
    g_loss.backward()
    self.g_optimizer.step()

    return g_loss.item()

  """
  Критик максимизирует разницу. а генератор минимизирует
  """
  def critic_step(self, x):

    real_score = self.critic(x)
    
    z = torch.randn(x.shape[0], m.nz, 1, 1, device=device)
    fake_imgs = self.forward(z)
    fake_score = self.critic(fake_imgs.detach())

    # Вычисление градиентного штрафа
    gp = self.gradient_penalty(x, fake_imgs)

    # Потеря критика: максимизация разницы между реальными и сгенерированными данными
    #c_loss = -(torch.mean(real_score) - torch.mean(fake_score)) + gp
    c_loss = torch.mean(fake_score) - torch.mean(real_score) + gp

    self.c_optimizer.zero_grad()
    c_loss.backward()
    self.c_optimizer.step()

    return c_loss.item()

  
  """
  Зачем нужна интерполяция?
  Интерполяция в градиентном штрафе (Gradient Penalty) используется для обеспечения 
  выполнения условия 1-Липшицевости критика на ВСЕМ ПРОМЕЖУТКЕ между реальными и сгенерированными данными.
  1-Липшицевость должна выполняться на всём промежутке между реальными и сгенерированными данными.
  В оригинальном WGAN Липшицевость достигается с помощью отсечения весов критика.
  """

  def gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=device) # Случайные коэффициенты из [0, 1]
        # Интерполяция создаёт промежуточные точки между реальными и сгенерированными данными
        interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True) # Интерполяция данных

        interpolates_score = self.critic(interpolates)

        # Вычисление градиентов критика по интерполированным данным
        gradients = torch.autograd.grad(
            outputs=interpolates_score,
            inputs=interpolates,
            grad_outputs=torch.ones_like(interpolates_score, device=device),
            create_graph=True,
            retain_graph=True
        )[0]

        # Вычисление нормы L2 для градиентов
        gradients = gradients.view(batch_size, -1) # Преобразуем в плоский вектор
        gradient_norm = gradients.norm(2, dim=1) # L2-норма

        # Градиентный штраф
        gp = self.lambda_gp * ((gradient_norm - 1) ** 2).mean()  # Отклонение нормы от 1
        return gp
  
  def train(self, dataloader, num_epochs, critic_steps=5):
        
        current_epoch = self.current_epoch
        total_epoch = num_epochs + current_epoch

        for epoch in range(current_epoch, total_epoch + 1):
            
            num_batches = 0
            
            with tqdm(dataloader, desc=f"Epoch {epoch}/{total_epoch}", unit="batch") as pbar:
                for batch, _ in pbar: 
                    batch = batch.to(device)

                    num_batches += 1
                
                    # Обучение критика несколько раз
                    for _ in range(critic_steps):
                        c_loss = self.critic_step(batch)

                    # Обучение генератора
                    g_loss = self.generator_step(batch)

                    # Логирование потерь в TensorBoard
                    self.writer.add_scalar('Loss/Critic', c_loss, epoch * len(dataloader) + num_batches)
                    self.writer.add_scalar('Loss/Generator', g_loss, epoch * len(dataloader) + num_batches)

                    pbar.set_postfix(G_loss=g_loss, C_loss=c_loss)

            # Сохранение прогресса
            epoch_test_images = self.forward(self.test_noises)
            self.test_progression.clear()
            self.test_progression.append(epoch_test_images.detach().cpu().numpy())

            if epoch % 20 == 0:
                self.current_epoch = epoch
                self.visualize_images(epoch)
            
            if epoch % 50 == 0:
                self.current_epoch = epoch
                self.save_model(f'models/wgan_model_{epoch}')
          
        self.current_epoch = total_epoch
        self.writer.close()

  def visualize_images(self, epoch):
        nrow, ncol = 3, 8
        fig = plt.figure(figsize=((ncol + 1) * 2, (nrow + 1) * 2))
        fig.suptitle(f'Epoch {epoch}', fontsize=30)
        gs = gridspec.GridSpec(nrow, ncol,
                               wspace=0.0, hspace=0.0,
                               top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                               left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
        for i in range(nrow):
            for j in range(ncol):
                img_idx = i * ncol + j
                img = np.reshape(self.test_progression[-1][img_idx], (28, 28))
                ax = plt.subplot(gs[i, j])
                ax.imshow(img, cmap='gray')
                ax.axis('off')
        plt.savefig(f'images/wgan_image_epoch_{epoch}.png')
        plt.close(fig)

  def save_model(self, path):
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_g_state_dict': self.g_optimizer.state_dict(),
            'optimizer_c_state_dict': self.c_optimizer.state_dict(),
            'current_epoch': self.current_epoch
        }, path)
        print(f'Model saved as {path} at epoch {self.current_epoch}')

  def load_model(self, path):
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.c_optimizer.load_state_dict(checkpoint['optimizer_c_state_dict'])
        self.current_epoch = checkpoint['current_epoch'] + 1
        print(f'Model loaded from {path}, starting from epoch {self.current_epoch}')
