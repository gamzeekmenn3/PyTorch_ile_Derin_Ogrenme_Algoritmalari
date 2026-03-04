"""
CIFAR-10 Nesne Tanıma ve Sınıflandırma:
Bu proje, PyTorch kütüphanesi kullanılarak CIFAR-10 veri setindeki 10 farklı nesne kategorisini sınıflandırmak için Evrişimli Sinir Ağı (CNN) mimarisi içerir.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# %% 1. Veri Yükleme VE Ön İşleme
def get_data_loaders(batch_size=64): # CIFAR10 verilerini indirir, normalize eder ve DataLoader objelerini döndürür.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(((0.5,0.5,0.5)), ((0.5,0.5,0.5))) #rgb kanallarını normalize et
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# %% 2. Veri Görselleştirme
def imshow(img):
    # Normalize edilmiş tensoru görselleştirmek için geri dönüştürür.
    img = img / 2 + 0.5 # Unnormalize: (x * std) + mean
    np_img = img.numpy() # Tensor'u numpy array'e çevir
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
plt.show()

def get_sample_images(train_loader): # Eğitim veri setinden bir batch örnek görüntü döndürür.
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    return images, labels

def visualize(n): # Eğitim veri setinden n adet örnek görüntüyü ekrana basar.
    train_loader, test_loader = get_data_loaders()
    images, labels = get_sample_images(train_loader)
    plt.figure()
  
    for i in range(n):
        plt.subplot(1, n, i+1)
        imshow(images[i])
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
    plt.show()

# %% 3. CNN Model Mimarisi
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
      
        # 1. Evrişim Bloğu: Giriş 3x32x32 -> Çıkış 32x32x32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # Max Pooling: Boyutu yarıya indirir -> 32x16x16
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2. Evrişim Bloğu: Giriş 32x16x16 -> Çıkış 64x16x16
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Max Pooling sonrası -> 64x8x8

        # Regülasyon: Overfitting'i (aşırı öğrenme) engellemek için %20 dropout
        self.dropout = nn.Dropout(0.2) 

        # Tam Bağlantılı Katmanlar (Fully Connected)
        # 64 kanal * 8x8 boyut = 4096 giriş özelliği
        self.fc1 = nn.Linear(64 * 8 * 8, 128) 
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Katmanların sırayla çalıştırılması
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        # Flatten: Çok boyutlu matrisi vektöre çevirme
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Cihaz seçimi (GPU varsa kullan)

# Loss fonksiyonu ve optimizer tanımı
define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(), 
    optim.SGD(model.parameters(), lr=0.001, momentum=0.9))

# %% 4. EĞİTİM VE TEST FONKSİYONLARI
def train_model(model, train_loader, criterion, optimizer, epochs = 5):
    # Bu fonksiyon modeli verilen eğitim verisi üzerinde eğitir. Her epoch sonunda ortalama loss değerini hesaplar ve eğitim sürecini grafikle gösterir.
    model.train()
    train_losses = [] 
    for epoch in range(epochs): 
        total_loss = 0 
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  
            outputs = model(images)
            loss = criterion(outputs, labels) 
            loss.backward() 
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.5f}')

    # Kayıp Grafiği Çizimi
    plt.figure
    plt.plot(range(1, epochs+1), train_losses, marker='o', linestyle='-', label = 'Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()

def test_model(model, test_loader, dataset_type='Test'):
    # Bu fonksiyon modeli test veya eğitim verisi üzerinde değerlendirir. Doğru tahmin sayısını hesaplayarak doğruluk (accuracy) oranını ekrana yazdırır.
    model.eval()
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader: 
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) 
            _, predicted = torch.max(outputs.data, 1) 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'{dataset_type} accuracy: {100 * correct / total}%')

# %% 5. ANA PROGRAM AKIŞI
# Bu bölüm programın çalıştırıldığı ana kısımdır. Veri yüklenir, model oluşturulur, eğitim gerçekleştirilir ve ardından model test edilir.
if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()
    visualize(10)

    model = CNN().to(device)
    criterion, optimizer = define_loss_and_optimizer(model)
    train_model(model, train_loader, criterion, optimizer, epochs=10)

    test_model(model, test_loader, dataset_type='Test')
    train_model(model, train_loader, dataset_type='Train')
