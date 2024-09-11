import os
import re
import string
import pandas as pd
import joblib
from tqdm import tqdm  # Eğitim sırasında ilerleme göstergesi için
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

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

# Klasör ve dosya yolları
base_dir = '/Users/akadraydn/Desktop/sentiment-analysis-with-ai/dataset'
categories = ['Culture', 'Finance', 'Medical', 'Politics', 'Religion', 'Sports', 'Tech']

# Veri setini yükleme ve işleme fonksiyonu
def load_and_preprocess_data(base_dir, categories):
    data = []
    
    for category in categories:
        category_dir = os.path.join(base_dir, category)
        for filename in tqdm(os.listdir(category_dir), desc=f"Processing {category}"):
            file_path = os.path.join(category_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                preprocessed_text = preprocess_text(text)
                data.append((preprocessed_text, category))
    
    return pd.DataFrame(data, columns=['text', 'label'])

# Veriyi yükleme
print("Veri seti yükleniyor ve işleniyor...")
df = load_and_preprocess_data(base_dir, categories)

# Veriyi eğitim ve test setlerine ayırma
print("Veri eğitim ve test setlerine ayrılıyor...")
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# TF-IDF vektörizasyonu
print("TF-IDF vektörizasyonu yapılıyor...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Temel modeller
estimators = [
    ('svc', SVC(probability=True)),  # SVC modeli probabilistik çıktılar için
    ('sgd', SGDClassifier()),
    ('lr', LogisticRegression(max_iter=1000)),
    ('ridge', RidgeClassifier())
]

# Stacking modeli - meta model olarak Logistic Regression kullanıyoruz
stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000)
)

# Modeli eğitme
print("Model eğitiliyor...")
stacking_model.fit(X_train_tfidf, y_train)

# Test seti üzerinde tahmin yapma
print("Test seti üzerinde tahminler yapılıyor...")
y_pred = stacking_model.predict(X_test_tfidf)

# Sonuçları değerlendirme
print("Sonuçlar değerlendiriliyor...")
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(report)

# Vektörizer'ı kaydet
vectorizer_save_path = '/Users/akadraydn/Desktop/sentiment-analysis-with-ai/models/tfidf_vectorizer.pkl'
joblib.dump(vectorizer, vectorizer_save_path)
print(f"TF-IDF vektörizer {vectorizer_save_path} yoluna kaydedildi.")

# Modeli kaydetme
model_save_path = '/Users/akadraydn/Desktop/sentiment-analysis-with-ai/models/stacking_model.pkl'
print(f"Model {model_save_path} yoluna kaydediliyor...")
joblib.dump(stacking_model, model_save_path)
print("Model başarıyla kaydedildi.")