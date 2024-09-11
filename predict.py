import re
import string
import joblib
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Emoji kaldırma fonksiyonu
def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F700-\U0001F77F"
                               u"\U0001F780-\U0001F7FF"
                               u"\U0001F800-\U0001F8FF"
                               u"\U0001F900-\U0001F9FF"
                               u"\U0001FA00-\U0001FA6F"
                               u"\U0001FA70-\U0001FAFF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r' ', text)

# Noktalama işaretlerini kaldırma fonksiyonu
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

# Stopwords kaldırma fonksiyonu
def remove_stopwords_arabic(text):
    stop_words = set(stopwords.words('arabic'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Fazla boşlukları temizleme fonksiyonu
def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

# Arapça kök bulma (Stemming) fonksiyonu
stemmer = ISRIStemmer()

def stem_arabic_text(text):
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

# Metni ön işleme fonksiyonu
def preprocess_text(text):
    text = remove_emoji(text)                   # Emoji temizleme
    text = remove_punctuation(text)             # Noktalama işaretlerini kaldırma
    text = remove_stopwords_arabic(text)        # Stopwords kaldırma
    text = remove_extra_spaces(text)            # Fazla boşlukları kaldırma
    text = stem_arabic_text(text)               # Arapça kök bulma (stemming)
    return text

# Kaydedilen modeli ve TF-IDF vektörizer'ı yükleme
model_path = '/Users/akadraydn/Desktop/sentiment-analysis-with-ai/models/stacking_model.pkl'
vectorizer_path = '/Users/akadraydn/Desktop/sentiment-analysis-with-ai/models/tfidf_vectorizer.pkl'

print("Model ve TF-IDF vektörizer yükleniyor...")
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Kullanıcının girdiği metin
text_input = input("Lütfen analiz etmek istediğiniz metni girin: ")

# Metni ön işleme
print("Metin işleniyor...")
processed_text = preprocess_text(text_input)

# Metni TF-IDF vektörüne dönüştürme
print("Metin vektörleştiriliyor...")
text_vector = vectorizer.transform([processed_text])

# Model ile sınıflandırma
print("Tahmin yapılıyor...")
predicted_class = model.predict(text_vector)[0]

# Sonucu yazdırma
print(f"Girilen metin '{predicted_class}' sınıfına aittir.")
