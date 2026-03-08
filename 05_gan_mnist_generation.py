"""
MNIST Veri Seti ile GAN (Generative Adversarial Network) Görüntü Üretimi:
PyTorch ile el yazısı rakamlardan oluşan MNIST veri setini kullanarak bir GAN mimarisinin nasıl kurulacağını, Discriminator (Ayırt Edici) ve Generator (Üretici) ağlarının birbirleriyle nasıl rekabet ederek 
eğitileceğini ve modelin sıfırdan nasıl yeni görüntüler üreteceğini gösterir.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# %% Veri Seti Hazırlama ve Ön İşleme 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Cihaz ayarı (GPU desteği kontrolü)

# Hiperparametreler
batch_size = 128
image_size = 28*28 # MNIST görüntülerinin düzleştirilmiş boyutu (784)

# Görüntülerin -1 ile 1 arasına normalize edilmesi (Tanh aktivasyonu ile uyum için)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

# MNIST veri setini yükleme ve DataLoader kurulumu
dataset = datasets.MNIST(root= "./data", train=True, transform=transform, download=True)
dataLoader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

# %% Discriminator (Ayırt Edici) Model Mimarisi
class Discriminator(nn.Module): # Giriş görüntülerini 'Gerçek' (1) veya 'Sahte' (0) olarak sınıflandırır.
    def __init__(self):
        super(Discriminator,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size,1024),
            nn.LeakyReLU(0.2), # Gradyan kaybını önlemek için LeakyReLU
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1), 
            nn.Sigmoid() # Olasılık değeri üretir (0-1)
        )
    def forward(self,img):
        return self.model(img.view(-1,image_size))

# %% Generator (Üretici) Model Mimarisi
class Generator(nn.Module): #Rastgele gürültüden (z_dim) başlayarak gerçekçi görüntüler üretmeye çalışır.
    def __init__(self,z_dim):
        super(Generator,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim,256),
            nn.ReLU(),
            nn.Linear(256,512), 
            nn.ReLU(),
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Linear(1024, image_size), 
            nn.Tanh() # Çıktıyı -1 ile 1 arasına getirir
        )
    def forward(self,x):
        return self.model(x).view(-1,1,28,28) # Çıktıyı görüntü formatına sokar

# %% Model Başlatma ve Eğitim Yapılandırması
# Hiperparametreler
learning_rate = 0.0002
z_dim = 100 # Latent space (gizli alan) boyutu
epochs = 20 

# Modellerin ve Optimizasyon algoritmalarının (Adam) tanımlanması
generator = Generator(z_dim).to(device)
discriminator = Discriminator().to(device)
criterion = nn.BCELoss() # Binary Cross Entropy Loss

g_optimizer = optim.Adam(generator.parameters(),lr=learning_rate,betas=(0.5,0.999)) 
d_optimizer = optim.Adam(discriminator.parameters(),lr=learning_rate,betas=(0.5,0.999))

# %% Eğitim Döngüsü
for epoch in range(epochs):
    for i, (real_imgs, _) in enumerate(dataLoader):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)
        # Etiketlerin hazırlanması
        real_labels = torch.ones(batch_size,1).to(device)
        fake_labels = torch.zeros(batch_size,1).to(device)

        # --- Discriminator Eğitimi ---
        # 1. Gerçek görüntülerle eğit
        real_loss = criterion(discriminator(real_imgs),real_labels)
        # 2. Sahte görüntülerle eğit
        z = torch.randn(batch_size, z_dim).to(device)
        fake_imgs = generator(z)
        fake_loss = criterion(discriminator(fake_imgs.detach()),fake_labels)
        
        d_loss = real_loss + fake_loss
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # --- Generator Eğitimi ---
        # Discriminator'ı sahte görüntülerin gerçek olduğuna inandırmaya çalışır
        g_loss = criterion(discriminator(fake_imgs),real_labels)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
    print(f"Epoch {epoch+1}/{epochs} d_loss: {d_loss.item():.3f}, g_loss: {g_loss.item():.3f}")

# %% Model Testi ve Görselleştirme
# Rastgele gürültüden yeni görüntüler üretme ve 4x4 ızgarada gösterme
with torch.no_grad():
    z=torch.randn(16,z_dim).to(device)
    sample_imgs = generator(z).cpu()
    grid= np.transpose(utils.make_grid(sample_imgs,nrow=4,normalize=True),(1,2,0))
    plt.imshow(grid)
    plt.show()
