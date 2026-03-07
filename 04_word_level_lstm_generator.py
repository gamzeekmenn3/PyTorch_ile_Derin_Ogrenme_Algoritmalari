"""
LSTM (Long Short-Term Memory) ile Kelime Bazlı Metin Üretimi:
Bu script, PyTorch kullanarak gömme (embedding) ve LSTM katmanlarından oluşan  bir model mimarisi ile basit bir metin üretim sistemi sunar. Proje süreci; 
    veri ön işleme, hiperparametre optimizasyonu (Grid Search) ve model eğitimi aşamalarını kapsamaktadır.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from itertools import product

# %% 1. Veri Yükleme ve Ön İşleme (Data Preprocessing)
# Modelin eğitileceği örnek ürün yorumları metni
text = """Bu ürün beklentimi fazlasıyla karşıladı.Malzeme kalitesi gerçekten çok iyi.Kargo hızlı ve sorunsuz bir şekilde elime ulaştı. 
Fiyatına göre performansı harika. Kesinlikle tavsiye ederim ve öneririm!"""

# Metin temizleme: Noktalama işaretlerini kaldır, küçük harfe çevir ve tokenize et
words = text.replace("."," ").replace("!"," ").lower().split()

# Kelime haznesi (Vocabulary) oluşturma ve frekans tabanlı indeksleme
word_counts = Counter(words)
vocab = sorted(word_counts, key = word_counts.get, reverse=True)
word_to_ix = {word: i for i, word in enumerate(vocab)} # Kelime -> İndeks
ix_to_word = {i: word for i, word in enumerate(vocab)} # İndeks -> Kelime

# Eğitim verisi hazırlama: (Girdi Kelimesi, Hedef Kelime) çiftleri oluşturma
data = [(words[i], words[i+1]) for i in range(len(words)-1)]

# %% 2. LSTM Model Mimarisi
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # Kelimeleri yoğun vektör temsiline (dense vector) dönüştüren katman
        self.lstm = nn.LSTM(embedding_dim, hidden_dim) # Uzun vadeli bağımlılıkları öğrenen LSTM katmanı
        self.fc = nn.Linear(hidden_dim, vocab_size) # Gizli durumu (hidden state) kelime haznesi boyutuna eşleyen katman

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x.view(1,1,-1))
        output = self.fc(lstm_out.view(1, -1))
        return output

model= LSTM(len(vocab), embedding_dim=8, hidden_dim=32)

def prepare_sequence(seq, to_ix):
    return torch.tensor([to_ix[w] for w in seq], dtype=torch.long) # Kelime listesini PyTorch tensörüne dönüştürür.

# %% 3. Hiperparametre Optimizasyonu (Grid Search)
# Denenecek parametre kombinasyonları
embedding_sizes = [8, 16]
hidden_sizes = [32, 64]
learning_rates = [0.01, 0.005]

best_loss = float('inf')
best_params = {}
print("--- Hiperparametre Optimizasyonu Başlatıldı ---")

for emb_size, hidden_size, lr in product(embedding_sizes, hidden_sizes, learning_rates):
    print(f"Deneme: embedding_dim={emb_size}, hidden_dim={hidden_size}, learning_rate={lr}")
    model = LSTM(len(vocab), emb_size, hidden_size)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 50
    total_loss = 0
    for epoch in range(epochs):
        epoch_loss = 0
        for word, next_word in data:
            model.zero_grad()
            input_tensor = prepare_sequence([word], word_to_ix)
            target_tensor = prepare_sequence([next_word], word_to_ix)
            output = model(input_tensor)
            loss = loss_function(output, target_tensor)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() 
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.5f}")
        total_loss = epoch_loss

        # En iyi performansı gösteren parametreleri güncelle
        if total_loss < best_loss:
            best_loss = total_loss
            best_params = {
                'embedding_dim': emb_size,
                'hidden_dim': hidden_size,
                'learning_rate': lr
            }
        print()
print("En iyi parametreler:", best_params)

# %% 4. Final Model Eğitimi (Final Training)
# Optimizasyon sonucunda bulunan en iyi parametrelerle modeli yeniden oluştur
final_model = LSTM(len(vocab), best_params['embedding_dim'], best_params['hidden_dim'])
optimizer = optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])
loss_function = nn.CrossEntropyLoss()

print("--- Final Model Eğitimi Başlıyor ---")
epochs=100
for epoch in range(epochs):
    epoch_loss = 0
    for word, next_word in data:
        final_model.zero_grad()
        input_tensor = prepare_sequence([word], word_to_ix)
        target_tensor = prepare_sequence([next_word], word_to_ix)
        output = final_model(input_tensor)
        loss = loss_function(output, target_tensor)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {epoch_loss:.5f}")

# %% 5. Çıkarım (Inference) ve Metin Üretimi
def predict_sequence(start_word,num_words):
    # Eğitilen modeli kullanarak ardışık kelime tahmini yapar.
    current_word = start_word
    output_sequence = [current_word]
    
    for _ in range(num_words):
        with torch.no_grad(): 
            input_tensor = prepare_sequence([current_word], word_to_ix)
            output = final_model(input_tensor)
            predicted_ix = torch.argmax(output).item()
            predicted_word = ix_to_word[predicted_ix]
            output_sequence.append(predicted_word)
            current_word = predicted_word
    return output_sequence

start_word = "ürün"
num_predictions = 1
predicted_sequence = predict_sequence(start_word, num_predictions)
print(" ".join(predicted_sequence))
