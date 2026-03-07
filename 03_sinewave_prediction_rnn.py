"""
RNN (Recurrent Neural Networks) ile Zaman Serisi Tahmini: 
Sinüs Dalgası Örneği: Bu script, PyTorch kullanarak temel bir RNN modelinin nasıl kurulacağını, zaman serisi verilerinin (Sinüs dalgası) nasıl hazırlanacağını ve eğitilen modelin test verileri üzerinde nasıl 
tahmin yapacağını gösterir.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# %% Veri Seti Oluşturma ve Hazırlama
def generate_data(seq_length = 50, num_samples = 1000):
    """
    Sinüs dalgası verisi üretir ve RNN girişine uygun (sequence, target) çiftlerine dönüştürür.
      - seq_length (int): Modelin bir tahminde bulunmak için geriye dönük bakacağı adım sayısı.
      - num_samples (int): Toplam veri noktası sayısı.
    """
    X = np.linspace(0,100,num_samples) # 0-100 arasında zaman ekseni oluştur
    y = np.sin(X) # Zaman eksenine karşılık gelen sinüs değerleri
    sequence = [] # Model giriş dizilerini saklar
    targets = [] # Modelin tahmin etmesi gereken hedef değerleri saklar

    # Kayan Pencere (Sliding Window) yöntemi ile veri hazırlama
    for i in range(len(X) - seq_length):
        sequence.append(y[i:i+seq_length]) # Giriş: t anına kadar olan dizi
        targets.append(y[i+seq_length]) # Hedef: t+1 anındaki gerçek değer
   
    # Kaynak verinin görselleştirilmesi
    plt.figure(figsize=(8,4))
    plt.plot(X,y, label="sin(x)", color="b",linewidth=2)
    plt.title("Sinüs Dalga Grafiği")
    plt.xlabel("Zaman(radyan)")
    plt.ylabel("Genlik")
    plt.legend()
    plt.grid(True)
    plt.show()

    return np.array(sequence), np.array(targets)

sequence, targets = generate_data()

# %% RNN Model Mimarisi Tanımlama
class RNN(nn.Module):
    # Çoktan-Bire (Many-to-One) RNN Mimarisi. Ardışık veriyi işleyerek son gizli durum (hidden state) üzerinden tek bir regresyon çıktısı üretir.
    def __init__(self, input_size, hidden_size, output_size,num_layers):
        super(RNN, self).__init__()
        # RNN Katmanı: Giriş verisini (Batch, Sequence, Feature) formatında işler
        self.rnn = nn.RNN(input_size, hidden_size,num_layers,batch_first=True)
        # Tam Bağlantılı (FC) Katman: RNN çıktısını tahmin değerine eşler
        self.fc = nn.Linear(hidden_size, output_size) 

    def forward(self, x):
        out,_ = self.rnn(x) # RNN üzerinden ileri besleme
        out = self.fc(out[:,-1,:]) # Sadece son zaman adımındaki çıktıyı alarak doğrusal katmana aktar
        return out
model = RNN(1, 16, 1,1)

# %% Hiperparametreler ve Eğitim Yapılandırması
# Model Yapılandırması
seq_length = 50 # Girdi dizisi uzunluğu
input_size = 1 # Her zaman adımındaki özellik sayısı (tek boyutlu sinüs değeri)
hidden_size = 16 # Gizli katmandaki hücre sayısı
output_size = 1 # Tahmin edilecek değer sayısı
num_layers = 1 # Üst üste binen RNN katman sayısı

# Eğitim Parametreleri
epochs = 20 # Tüm veri setinin kaç kez taranacağı
batch_size = 32 # Her iterasyonda işlenecek örnek sayısı
learning_rate = 0.001 # Optimizasyon adım büyüklüğü

# Veri Hazırlığı
X, y = generate_data(seq_length)

# Verileri PyTorch tensörlerine dönüştürme ve boyut ekleme [Batch, Seq, Feature]
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

# PyTorch Veri Yükleyici (DataLoader) kurulumu
dataset = torch.utils.data.TensorDataset(X,y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# %% Model Başlatma ve Eğitim Döngüsü
model = RNN(input_size, hidden_size, output_size,num_layers)
criterion = nn.MSELoss() # Kayıp fonksiyonu - mean squared error
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad() # Gradyanları sıfırla
        pred_y = model(batch_X) # İleri besleme: Tahmin üretme
        loss = criterion(pred_y, batch_y)# Tahmin ile gerçek değer arasındaki farkı hesapla ve loss hesapla
        loss.backward() # Geri yayılım: Gradyanları hesapla
        optimizer.step() # Ağırlıkları güncelle

    print(f'Epoch: [{epoch+1}/{epochs}], Loss: {loss.item():.4f}') # Her epoch sonunda mevcut kaybı yazdır

# %% Model Testi ve Değerlendirme 
X_test = np.linspace(100,110,seq_length).reshape(1,-1)
y_test = np.sin(X_test)
X_test2 = np.linspace(120,130,seq_length).reshape(1,-1) 
y_test2 = np.sin(X_test2)

# Numpy verilerini tensör formatına dönüştürme
X_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1) 
X_test2 = torch.tensor(y_test2, dtype=torch.float32).unsqueeze(-1) 

# Model üzerinden tahmin (Inference) gerçekleştirme
model.eval()
prediction1 = model(X_test).detach().numpy()
prediction2 = model(X_test2).detach().numpy()

# Sonuçların görsel olarak karşılaştırılması
plt.figure()
plt.plot(np.linspace(0,100,len(y)), y, marker='o', label='Training dataset')
plt.plot(X_test.numpy().flatten(), marker="o", label="Test 1")
plt.plot(X_test2.numpy().flatten(), marker="o", label="Test 2")
plt.plot(np.arange(seq_length,seq_length+1), prediction1.flatten(), "ro",label="Prediction 1")
plt.plot(np.arange(seq_length,seq_length+1), prediction2.flatten(), "go",label="Prediction 2")
plt.legend()
plt.show()
