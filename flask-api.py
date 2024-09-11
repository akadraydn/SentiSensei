from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # CORS hatalarını önlemek için ekledik

# Model ve vektörizer dosyalarını yükle
model = joblib.load('/Users/akadraydn/Desktop/sentiment-analysis-with-ai/models/stacking_model.pkl')
vectorizer = joblib.load('/Users/akadraydn/Desktop/sentiment-analysis-with-ai/models/tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']

    # Veriyi TF-IDF ile dönüştür
    transformed_text = vectorizer.transform([text])

    # Model tahmini yap
    prediction = model.predict(transformed_text)

    # Tahmini JSON formatında döndür
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True, port=5002)
