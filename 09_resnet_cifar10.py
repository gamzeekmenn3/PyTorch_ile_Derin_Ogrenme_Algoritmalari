"""
ResNet ile CIFAR-10 Sınıflandırma: Custom ResNet ve Transfer Learning Uygulaması
Bu çalışma iki ana yaklaşımı içerir:
1. ResNet mimarisini temel bloklarından (Residual Blocks) itibaren sıfırdan inşa etmek.
2. PyTorch'un önceden eğitilmiş ResNet18 modelini kullanarak Transfer Learning yapmak.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm

# Cihaz Yapılandırması (GPU varsa kullanılır)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Veri Seti Hazırlığı
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) 
    ])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# %% Residual (Artık) Blok Tanımı
class ResidualBlock(nn.Module):
# ResNet'in temel yapı taşı. 'Skip Connection' sayesinde derin ağlarda kaybolan gradyan (vanishing gradient) problemini çözer.
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        # İlk Evrişim: Boyut küçültme stride parametresi ile yapılır
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=3, stride = stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) # Batch normalization
        self.relu = nn.ReLU()
        # İkinci Evrişim
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x # Girişi sakla (Short-cut bağlantısı için)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Giriş verisini (identity) çıktıya ekle: F(x) + x
        out += identity
        out = self.relu(out)

        return out
    
# %% Özel ResNet Mimarisi Oluşturma
class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet Katmanları (Layer 1-4)
        self.layer1 = self._make_layer(64, 64, blocks=2)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # Global Ortalama Havuzlama ve Tam Bağlı Katman
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        # Eğer giriş ve çıkış kanalları uyuşmuyorsa veya stride > 1 ise boyutları eşitle
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]

        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# %% Model Seçimi (Custom vs Transfer Learning)
use_custom_model = True
if use_custom_model:
    model = CustomResNet(num_classes=10).to(device)
else:
    model = models.resnet18(pretrained = True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Linear(256, 10))
    model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %% Eğitim Döngüsü
num_epochs = 1
for epoch in tqdm(range(num_epochs)):
    model.train()
    running_loss = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# %% Değerlendirme (Evaluation)
model.eval()
correct=0
total=0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model (images)
        _,predicted= torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Test accuracy: {100* correct / total}%")
