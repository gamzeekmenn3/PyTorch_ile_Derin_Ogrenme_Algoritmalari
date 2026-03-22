"""
Sentiment Analysis with Transformer Architecture (PyTorch):
- Bu proje, Doğal Dil İşleme (NLP) tekniklerini kullanarak metin tabanlı bir duygu analizi (sınıflandırma) gerçekleştirmeyi amaçlar.
- Transformer mimarisinin Self-Attention mekanizmasını kullanarak pozitif ve negatif yorumlar arasındaki anlamsal farkları öğrenir.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import string
from collections import Counter

# %% Veri Tanımlama ve Ön İşleme(Preprocessing)
positive_sentences = [
"this is amazing", "I really like this","this product is great","absolutely fantastic experience","I enjoyed every moment","highly recommend this","this is wonderful",
"very satisfied with the result","this exceeded my expectations","I love this so much","brilliant work","top quality product","very impressive","superb performance",
"I am very happy with this","this works perfectly","excellent choice","great value for money","this is outstanding","I would buy this again","this is very useful",
"perfect in every way","I appreciate this","this is awesome","I had a great time","this is beautiful","very well done","I am impressed","this is the best",
"so खुश with this", "this made my day","absolutely loved it","great experience overall","this is top notch","super happy with it","this is brilliant","very good quality",
"I like it a lot","this is perfect","this works great","I am delighted","this is fantastic","great job","I am satisfied","very nice","this is cool","great design",
"I feel good about this","this is lovely","amazing quality","this is very good","extremely pleased","very comfortable","this is excellent","happy with my purchase",
"this is impressive","good experience","really nice product","this is enjoyable","top performance","very reliable","I really enjoyed this","this is well made",
"it looks great","this is superb","very effective","I am pleased","this is extraordinary","very smooth experience","this is exceptional","great features",
"this is outstanding work","nice quality","this is satisfying","very fast and efficient","this is perfect for me","I love the design","this is incredible",
"very useful product","this is top quality","very good experience","this is wonderful indeed","great support","this is remarkable","very enjoyable","this is well done",
"great functionality","I am very pleased","this is high quality","very practical","this is amazing indeed","this is very impressive","this is just perfect","really satisfied",
"this is fantastic work","this is very enjoyable","great usability","this is awesome product","very elegant","this is perfect choice"]

negative_sentences = [
"I do not like this","this is terrible","very bad experience","I hate this","this is disappointing","not worth the money","this is awful","I am unhappy with this",
"very poor quality","this is not good","I regret buying this","this is horrible","very frustrating","this did not work","I am disappointed","this is bad","not recommended",
"this is useless","very annoying","I dislike this","this is unacceptable","this is very poor","bad experience overall","this is not worth it","I am not satisfied",
"this is disappointing product","very low quality","this is a waste","I don't recommend this","this is broken","very खराब","this is terrible quality","very slow and bad",
"this is not impressive","I expected better","this is frustrating","very bad design","this is not useful","this is disappointing indeed","this is a bad choice",
"I am upset","this is very annoying","not happy at all","this is poor","this is not good at all","very bad performance","this is unsatisfactory","I do not enjoy this",
"this is awful experience","very unreliable","this is terrible work","I am very disappointed","this is bad quality","this is not okay","very негативный", "this is the worst",
"this is not good product","I don't like it","this is frustrating experience","this is very bad","this is low quality","this is a problem","I am not impressed",
"this is horrible product","this is a failure","this is very disappointing","this is not effective","this is messy","this is broken product","very disappointing result",
"this is not reliable","I am unhappy","this is poorly made","this is bad experience","this is not nice","this is low performance","this is not good quality",
"this is irritating","this is not acceptable","this is very weak","this is not useful at all","this is terrible experience","I am dissatisfied","this is not working",
"this is a bad product","this is low standard","this is not impressive at all","this is disappointing quality","this is a waste of time","this is not efficient",
"this is poor design","this is not enjoyable","this is bad choice","this is not good enough","this is very poor experience","this is unreliable","this is worst experience",
"this is not good at all"]

def preprocess(text): # Metni küçük harfe çevirir ve noktalama işaretlerini temizler.
    text = text.lower()
    text = text.translate(str.maketrans("","", string.punctuation))
    return text

# Veri birleştirme ve etiketleme (1: Pozitif, 0: Negatif)
data = positive_sentences + negative_sentences
labels = [1] * len(positive_sentences) + [0] * len(negative_sentences)
data = [preprocess(sentence) for sentence in data]

# Kelime Dağarcığı (Vocabulary) Oluşturma
all_words = " ".join(data).split()
word_counts = Counter(all_words)
vocab = {word: idx + 1 for idx, (word, _) in enumerate(word_counts.items())}
vocab["<PAD>"] = 0  # Dolgu (Padding) için özel token

# Metinleri Sayısal Tensörlere Dönüştürme
max_len = 15
def sentence_to_tensor(sentence, vocab, max_len=15):
    tokens = sentence.split()
    indices = [vocab.get(word, 0) for word in tokens]
    indices = indices[:max_len]
    indices += [0] * (max_len - len(indices))
    return torch.tensor(indices)

X = torch.stack([sentence_to_tensor(sentence, vocab, max_len) for sentence in data])
y = torch.tensor(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% Transformer Model Mimarisi
class TransformerClass(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, num_classes):
        super(TransformerClass, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, embedding_dim))
        self.transformer = nn.Transformer(d_model=embedding_dim, # Embedding vektör boyutu
                                          nhead=num_heads, # Dikkat mekanizması başlık sayısı
                                          num_encoder_layers=num_layers, # Encoder katman derinliği
                                          dim_feedforward=hidden_dim, # Feed-forward ağının gizli katman boyutu
                                          batch_first=True # Input formatının (batch, seq, feature) olduğunu belirtir
                                          )

        # Sınıflandırma Katmanları
        self.fc = nn.Linear(embedding_dim * max_len, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid() # İkili sınıflandırma (0-1 arası olasılık)

    def forward(self,x):
        embedded = self.embedding(x) + self.positional_encoding
        output = self.transformer(embedded,embedded)
        output = output.view(output.size(0),-1)
        output = torch.relu(self.fc(output))
        output = self.out(output)
        output = self.sigmoid(output)
        return output

# %% Model Eğitimi
vocab_size = len(vocab)
embedding_dim = 32
num_heads = 4
num_layers = 4
hidden_dim = 64
num_classes = 1

model = TransformerClass(vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, num_classes)

criterion = nn.BCELoss() # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr = 0.0005)

num_epochs = 30
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(X_train.long()).squeeze()
    loss = criterion(output, y_train.float())
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs} Loss: {loss}")

# %% Model Değerlendirme (Test)
model.eval()
with torch.no_grad():
    y_pred = model(X_test.long()).squeeze()
    y_pred = (y_pred > 0.5).float()
    
    y_pred_training = model(X_test.long()).squeeze()
    y_pred_training = (y_pred_training > 0.5).float()

accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy}")

accuracy_train = accuracy_score(y_test, y_pred_training)
print(f"Train accuracy: {accuracy_train}")
