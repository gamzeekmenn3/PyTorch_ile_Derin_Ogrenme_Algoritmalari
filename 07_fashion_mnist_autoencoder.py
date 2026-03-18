"""
FashionMNIST Autoencoder: 
Görüntü verilerini düşük boyutlu bir "latent space" içine sıkıştırmayı ve ardından bu sıkıştırılmış veriden orijinal görüntüyü minimum kayıpla geri üretmeyi amaçlar.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

# %% Veri Seti Hazırlığı
transform = transforms.Compose([transforms.ToTensor()]) # Görüntüleri tensöre çeviriyoruz. ToTensor() işlemi pikselleri otomatik [0, 1] arasına normalize eder.

train_dataset = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)

# Hiperparametreler
batch_size = 128
epochs = 50
learning_rate = 1e-3

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %% Model Mimarisi (Autoencoder)
class AutoEncoder(nn.Module):
    """
    Encoder: 784 pikseli (28x28) önce 256'ya, sonra 64 birime (latent space) sıkıştırır.
    Decoder: 64 birimlik özeti tekrar 784 piksele genişletir.
    """
    def __init__(self):
        # Encoder: Veriyi temsil eden en önemli özellikleri çıkarır (Sıkıştırma)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU()
        )
        # Decoder: Sıkıştırılmış veriden orijinal görüntüyü yeniden inşa eder
        self.decoder = nn.Sequential(
            nn.Linear(64,256),
            nn.ReLU(),
            nn.Linear(256,28*28),
            nn.Sigmoid(), # Sigmoid fonksiyonu çıktıyı [0-1] aralığında tutmak için kullanılır.
            nn.Unflatten(1,(1,28,28)) # Vektörü tekrar 28x28 görüntü formuna sokar
        )
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# %% Callback: Early Stopping
class EarlyStopping: # Kayıpta (loss) iyileşme durduğunda eğitimi erken sonlandırarak aşırı öğrenmeyi önler.
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def __call__(self, loss):
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0 # İyileşme var, sayacı sıfırla
        else:
            self.counter += 1 # İyileşme yok, sayacı artır
        
        return self.counter >= self.patience

# %% Eğitim Hazırlığı ve Fonksiyonu
model = AutoEncoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
early_stopping = EarlyStopping(patience=5, min_delta=0.001)

def training(model, train_loader, optimizer, criterion, early_stopping, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.5f}")

        if early_stopping(avg_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break

training(model,train_loader,optimizer,criterion,early_stopping,epochs)

# %% Değerlendirme Metriği: SSIM - Yapısal Benzerlik Endeksi (SSIM) hesaplar. 1.0 mükemmel benzerlik, 0.0 alakasız görüntüler demektir.
def compute_ssim(img1,img2, sigma = 1.5): 
    C1 = (0.01*255)**2
    C2 = (0.03*255)**2

    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    mu1,mu2 = gaussian_filter(img1,sigma), gaussian_filter(img2,sigma)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2

    sigma1_sq = gaussian_filter(img1**2, sigma) - mu1_sq
    sigma2_sq = gaussian_filter(img2**2, sigma) - mu2_sq
    sigma12 = gaussian_filter(img1*img2, sigma) - mu1_mu2

    ssim_map =  ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

# %% Görselleştirme ve Test
def evaluate_and_plot(model, test_loader, n_images=10):
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            inputs, _ = batch
            outputs = model(inputs)
            break
        
    inputs = inputs.numpy()
    outputs = outputs.numpy()

    fig, axes = plt.subplots(2, n_images, figsize = (n_images,3))
    ssim_scores = []

    for i in range(n_images):
        img1 = np.squeeze(inputs[i])
        img2 = np.squeeze(outputs[i])

        ssim_score = compute_ssim(img1,img2)
        ssim_scores.append(ssim_score)

        axes[0,i].imshow(img1, cmap= "gray")
        axes[0,i].axis("off")
        axes[1,i].imshow(img2, cmap= "gray")
        axes[1,i].axis("off")

    axes[0,0].set_title("Original")
    axes[1,0].set_title("Decoded image")
    plt.show()

    avg_ssim = np.mean(ssim_scores)
    print(f"Average SSIM: {avg_ssim}")

evaluate_and_plot(model,test_loader,n_images=10)
