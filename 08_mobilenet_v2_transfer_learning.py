"""
Oxford Flowers 102 Veri Seti ile Çiçek Sınıflandırma Projesi:
Bu proje, MobileNetV2 mimarisini kullanarak Transfer Learning yöntemiyle 102 farklı çiçek türünü yüksek doğrulukla sınıflandırmayı amaçlar.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# %% Veri Yükleme ve Veri Artırma (Data Augmentation)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Eğitim verisi için dönüşümler: Modelin genelleme yeteneğini artırmak için veri artırma uygulanır.
transform_train = transforms.Compose([
    transforms.Resize((224,224)), # MobileNet için standart giriş boyutu
    transforms.RandomHorizontalFlip(), # Görüntüleri rastgele yatay çevirme
    transforms.RandomRotation(10), # Rastgele 10 dereceye kadar döndürme
    transforms.ColorJitter(brightness=0.2, contrast=0.2,saturation=0.2,hue=0.1), # renk varyasyonları
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # Piksel değerlerini normalize etme
])
# Test verisi için dönüşümler: Sadece boyutlandırma ve normalizasyon uygulanır.
transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
# Oxford Flowers 102 veri setini indirme ve yükleme
train_dataset = datasets.Flowers102(root="./data",split="train",transform=transform_train,download=True)
test_dataset = datasets.Flowers102(root="./data",split="val", transform=transform_test, download=True)

# Veri setinden rastgele 5 örnek seçerek görselleştirme
indices = torch.randint(len(train_dataset),(5,))
samples = [train_dataset[i] for i in indices]

# görselleştirme
fig, axes = plt.subplots(1,5, figsize = (15,5))
for i, (image, label) in enumerate(samples):
    image = image.numpy().transpose((1,2,0)) # Tensör formatından (C,H,W) görüntü formatına (H,W,C) dönüşüm
    image = (image*0.5) + 0.5 # Normalizasyonu geri alarak görselleştirme
    axes[i].imshow(image)
    axes[i].set_title(f"Label: {label}")
    axes[i].axis("off")
plt.show()

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# %% Transfer Learning Tanımlama ve Model Eğitimi
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT) # Önceden eğitilmiş (Pre-trained) MobileNetV2 modelini yükleme

# Modelin son katmanını (Classifier) hedef veri setindeki sınıf sayısına (102) göre güncelleme
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs,102)
model = model.to(device)

# Kayıp fonksiyonu (CrossEntropy) ve Optimizer (Adam) tanımlama
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[1].parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) # Öğrenme oranını belirli periyotlarla azaltan Scheduler yapısı

# Model Eğitim Döngüsü (Training Loop)
epochs = 3
for epoch in tqdm(range(epochs)):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images,labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step() # Her epoch sonunda öğrenme oranını güncelleme
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}") # Epoch spnunda ortalama kaybı yazdırıyoruz

# modeli kaydetme
torch.save(model.state_dict(), "mobilenet_flowers102.pth")

# %% Test ve Değerlendirme
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs,1) # En yüksek olasılıklı sınıfı seçme
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Hata Matrisi (Confusion Matrix) Görselleştirme
cm = confusion_matrix(all_labels,all_preds)
plt.figure(figsize = (12,12))
sns.heatmap(cm, annot= False, cmap= "Blues")
plt.xlabel("Predicted")
plt.ylabel("Real")
plt.title("Confusion Matrix")
plt.show()

print(classification_report(all_labels,all_preds))
