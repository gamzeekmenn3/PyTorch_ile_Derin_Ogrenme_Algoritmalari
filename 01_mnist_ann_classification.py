"""
PyTorch ile MNIST El Yazısı Rakam Sınıflandırması:
Bu proje, PyTorch kütüphanesi kullanılarak MNIST veri setindeki el yazısı rakamları sınıflandırmak için bir Yapay Sinir Ağı (ANN) mimarisi içerir.
"""
# %% [1] Kütüphanelerin Yüklenmesi
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Donanım Kontrolü (GPU desteği varsa kullanılır, yoksa CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [2] Veri Hazırlığı (Data Loading)
def get_data_loaders(batch_size = 64): # MNIST veri setini indirir, normalize eder ve batch'ler halinde döner.
    
    # Görüntüleri Tensör'e çeviriyoruz ve -1 ile 1 arasında normalize ediyoruz
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # Eğitim ve Test setlerinin indirilmesi ve hazırlanması
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # DataLoader ile veri setlerini yükle ve batch'lere ayır
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def visualize_samples(loader, n):
    # Veri setinden rastgele örnek görüntüleri görselleştirir.
    images, labels = next(iter(loader))
   
    fig, axes = plt.subplots(1, n, figsize=(10, 5))
    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].set_title(f'Label: {labels[i].item()}')
        axes[i].axis('off')
    plt.show()

# %% [3] Model Mimarisi (ANN)
class NeuralNetwork(nn.Module):
    # 3 Katmanlı Tam Bağlı (Fully Connected) Yapay Sinir Ağı: Giriş: 28x28 (784) -> Gizli 1: 128 -> Gizli 2: 64 -> Çıkış: 10 (0-9 rakamları)
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten() # 2D görüntüyü (28,28) -> 1D vektöre (784) dönüştürür
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# %% [4] Eğitim Fonksiyonu
def train_model(model, train_loader, criterion, optimizer, epochs = 10):
    model.train()
    train_losses = []

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device) # Verileri kullanılan cihaza (GPU/CPU) gönder
           
           # Gradyanları sıfırla, ileri besleme yap, hatayı hesapla ve geri yayılım yaparak ağırlıkları güncelle
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.3f}")

    # loss graph
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, marker='o', linestyle='-', label = "Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()

# %% [5] Test ve Değerlendirme
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device) 
            predictions = model(images)
            _, predicted_labels = torch.max(predictions, 1) 
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:3f}%") 

# %% [6] Hiperparametre Yapılandırması (Yeni Bölüm)
def define_loss_and_optimizer(model, learning_rate=0.001):
    # Kayıp fonksiyonu ve optimizer'ı tanımlar.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return criterion, optimizer

# %% [7] Main Bloğu
if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()
    visualize_samples(train_loader, 5)
    model = NeuralNetwork().to(device)
    criterion, optimizer = define_loss_and_optimizer(model, learning_rate=0.001)
    losses = train_model(model, train_loader, criterion, optimizer)
    test_model(model, test_loader)
