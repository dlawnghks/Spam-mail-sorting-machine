import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 모델과 벡터라이저 로드
tfidf = joblib.load('tfidf_model.pkl')
knn = joblib.load('knn_model.pkl')
nb_model = joblib.load('nb_model.pkl')

@app.route('/predict', methods=['POST'])
def predict_email():
    data = request.json
    title = data.get('title', '')
    content = data.get('content', '')
    test_text = title + " " + content

    # TF-IDF 변환
    test_tfidf = tfidf.transform([test_text])

    # K-NN 모델 예측
    knn_pred = knn.predict(test_tfidf)[0]
    nb_pred = nb_model.predict(test_tfidf)[0]

    # 예측 결과 결정
    if knn_pred == nb_pred:
        result = "스팸" if knn_pred == 1 else "햄"
    else:
        result = "스팸 의심 메일"

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
