import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Отключает всё, кроме ошибок

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
#from data import train_loader, test_loader
import classification as cl
from gan import GAN
from dcgan import DCGAN
from wgan_gp import WGAN_GP 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Генерация датасетов из моделей
def generate_datasets(models, num_samples=1000, noise_dim=100, device=device):
    datasets = {}
    for model_name, model in models.items():
        model.generator.eval()  # Перевод генератора в режим оценки
        with torch.no_grad():
            noise_gan = torch.randn(num_samples, noise_dim, device=device)
            noise_dcgan = torch.randn(num_samples, noise_dim, 1, 1, device=device)
            if isinstance(model, GAN):
                generated_data = model.generator(noise_gan).detach().cpu()
                generated_data = generated_data.view(num_samples, 1, 28, 28)
            elif isinstance(model, DCGAN):
                generated_data = model.generator(noise_dcgan).detach().cpu()
            elif isinstance(model, WGAN_GP):
                generated_data = model.generator(noise_dcgan).detach().cpu()
            datasets[model_name] = generated_data
    return datasets

# Класс для адаптации сгенерированных данных
class GeneratedDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x


def main():
    lr = 0.001
    model = cl.Classification(lr, model_path="models/model_100")
    #model.train(train_loader, num_epochs=1000)
    #model.evaluate(test_loader)
    gan_instance = GAN(model_path="model_GAN_300")
    dcgan_instance = DCGAN(model_path="model_DCGAN_500")
    wgan_gp_instance = WGAN_GP(model_path="model_WGAN_GP_400")

    models = {
        'gan': gan_instance,
        'dcgan': dcgan_instance,
        'wgan-gp': wgan_gp_instance
    }

    # Генерация данных
    num_samples = 10000
    generated_datasets = generate_datasets(models, num_samples=num_samples, noise_dim=100, device=device))

    # Подготовка DataLoader для классификатора
    mnist_transforms = transforms.Compose([transforms.Normalize(mean=[0.1307], std=[0.3081])])
    dataloaders = {}
    for model_name, data in generated_datasets.items():
        dataset = GeneratedDataset(data, transform=mnist_transforms)
        dataloaders[model_name] = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    for model_name, dataloader in dataloaders.items():
        print(f"Counting predictions for dataset generated by {model_name}")
        model.count_predicted_classes(dataloader)  

if __name__ == "__main__":
    main()