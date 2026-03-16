"""
Radyal Tabanlı Fonksiyon Ağları (RBFN) ile Iris Sınıflandırma:
Bu proje, PyTorch kullanarak özel bir RBF katmanı oluşturmayı ve Iris veri seti üzerinde sınıflandırma yapmayı amaçlar.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %% Veri Hazırlama Süreci
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, header=None)
# Özelliklerin (X) ve hedef değişkenin (y) ayrıştırılması
X = df.iloc[:,:-1].values # İlk 4 sütun: Çanak/Taç yaprak ölçümleri
y, _ = pd.factorize(df.iloc[:,-1]) # Son sütun: Tür isimlerini sayısal sınıflara çevirir (0, 1, 2)

# Veriyi standardize etme
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Veri setini Eğitim ve Test olarak ikiye ayırma (%30 Test)
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.3,random_state=42)

# Verileri PyTorch üzerinde işlem yapabilmek için Tensör formatına dönüştürme
def to_tensor(data,target):
    return torch.tensor(data, dtype=torch.float32),torch.tensor(target,dtype=torch.long)

X_train, y_train = to_tensor(X_train,y_train)
X_test, y_test = to_tensor(X_test, y_test)

# %% RBFN Modeli ve Matematiksel Fonksiyonların Tanımlanması
# RBF Çekirdek Fonksiyonu (Gaussian Kernel): Giriş verisi ile merkezler arasındaki benzerliği hesaplar.
def rbf_kernel(X,centers,beta):
    return torch.exp(-beta*torch.cdist(X,centers)**2)

class RBFN(nn.Module):
    def __init__(self,input_dim,num_centers,output_dim):
        super(RBFN, self).__init__()
        # RBF Merkezleri: Modelin öğreneceği 'temsili' noktalar
        self.centers = nn.Parameter(torch.randn(num_centers,input_dim))
        # Beta: Çan eğrisinin (gauss) genişliğini kontrol eden parametre
        self.beta = nn.Parameter(torch.ones(1)* 2.0)
        # Çıkış Katmanı: RBF çıktısını hedef sınıf sayısına eşleyen doğrusal katman
        self.linear = nn.Linear(num_centers,output_dim)

    def forward(self,x):
        # 1. Adım: Veriyi RBF çekirdeğinden geçirerek yeni bir özellik uzayına taşı
        phi = rbf_kernel(x,self.centers,self.beta)
        # 2. Adım: Yeni özellikleri doğrusal katman ile sınıflandır
        return self.linear(phi)

# %% Modelin Kurulumu ve Eğitimi
num_centers = 10
model = RBFN(input_dim=4,num_centers=num_centers,output_dim=3)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

# Eğitim Süreci
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad() # Gradyanları sıfırla
    outputs = model(X_train) # İleri yayılım (Tahmin)
    loss = criterion(outputs, y_train) # Kayıp (Loss) hesapla
    loss.backward() # Geri yayılım (Gradyan hesaplama)
    optimizer.step() # Parametreleri güncelle (Öğrenme)

    if(epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

# %% 4. Test ve Model Performans Değerlendirmesi
with torch.no_grad(): # Test aşamasında gradyan hesaplamaya gerek yoktur
    y_pred = model(X_test)
    # En yüksek olasılıklı sınıfı belirle ve doğruluk oranını hesapla
    accuracy = (torch.argmax(y_pred,axis=1) == y_test).float().mean().item()
    print(f"Accuracy: {accuracy}")
