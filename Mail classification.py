import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
from sklearn.model_selection import train_test_split

# 데이터와 모델을 저장할 전역 변수
tfidf = None
knn = None
nb_model = None

def combined_data():
    google_message = pd.read_csv("C:/Project/Rabble/datasets/messages.csv")
    spam_message = pd.read_csv("C:/Project/Rabble/datasets/spam_virus.csv")
    google_message['메일종류'] = '햄'
    spam_message['메일종류'] = '스팸'
    final_google_df = google_message[['메일종류', '메일제목']]
    final_spam_df = spam_message[['메일종류', '메일제목']]

    ham_count = final_google_df.shape[0]
    final_spam_df = final_spam_df.sample(n=ham_count, random_state=42)
    combined_df = pd.concat([final_google_df, final_spam_df], axis=0, ignore_index=True)
    return combined_df

def train_model():
    global tfidf, knn, nb_model

    combined_data_df = combined_data()
    X = combined_data_df['메일제목']
    y = combined_data_df['메일종류'].apply(lambda x: 1 if x == '스팸' else 0)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    tfidf = TfidfVectorizer(max_features=3000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)

    # K-NN 모델 학습
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_tfidf, y_train)
    joblib.dump(knn, 'knn_model.pkl')

    # 나이브 베이즈 모델 학습
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    joblib.dump(nb_model, 'nb_model.pkl')

    # TF-IDF 모델 저장
    joblib.dump(tfidf, 'tfidf_model.pkl')

    print("모델이 학습되고 저장되었습니다.")
    return knn, nb_model

def predict_email():
    # 저장된 모델 로드
    tfidf = joblib.load('tfidf_model.pkl')
    knn = joblib.load('knn_model.pkl')
    nb_model = joblib.load('nb_model.pkl')

    # 사용자로부터 이메일 제목과 내용을 입력받기
    title = input("이메일 제목을 입력하세요: ")
    content = input("이메일 내용을 입력하세요: ")
    test_text = title + " " + content
    test_tfidf = tfidf.transform([test_text])

    # K-NN 예측
    knn_pred = knn.predict(test_tfidf)[0]
    nb_pred = nb_model.predict(test_tfidf)[0]

    # 결과 비교 및 출력
    if knn_pred == nb_pred:
        # 두 모델의 결과가 동일할 경우, 하나의 결과만 출력
        result = "스팸" if knn_pred == 1 else "햄"
        print(f"\n이 이메일은: {result}")
    else:
        # 두 모델의 결과가 다를 경우, "스팸 의심 메일"로 출력
        print("\n이 이메일은: 스팸 의심 메일")

# 모델 학습
train_model()

# 사용자 입력을 통한 판별
predict_email()
