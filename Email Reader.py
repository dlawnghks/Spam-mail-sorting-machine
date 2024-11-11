import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
from sklearn.model_selection import train_test_split

# 데이터와 모델을 저장할 전역 변수
tfidf = None
knn = None

def combined_data():
    # 학습 데이터 로드
    google_message = pd.read_csv("C:/Project/Rabble/datasets/messages.csv")
    spam_message = pd.read_csv("C:/Project/Rabble/datasets/spam_virus.csv")

    # 메일 종류 추가
    google_message['메일종류'] = '햄'
    spam_message['메일종류'] = '스팸'

    # 필요한 열 선택
    final_google_df = google_message[['메일종류', '메일제목']]
    final_spam_df = spam_message[['메일종류', '메일제목']]

    # Ham 데이터 수
    ham_count = final_google_df.shape[0]

    # Spam 데이터 언더샘플링
    final_spam_df = final_spam_df.sample(n=ham_count, random_state=42)

    # 데이터 결합
    combined_df = pd.concat([final_google_df, final_spam_df], axis=0, ignore_index=True)

    return combined_df

def load_and_prepare_test_data(title, content):
    # 테스트 데이터 불러오기 (구분자로 '|' 사용)
    #test_data = pd.read_csv("C:/Project/Rabble/datasets/test_mail.csv", sep='|')


    # 제목과 내용을 결합하여 새로운 열 생성
    #test_data['제목_내용'] = test_data['메일제목'] + " " + test_data['메일내용']

    test_data = title + " " + content

    return test_data  # 이메일 제목과 내용

def train_model():
    global tfidf, knn

    # 학습 데이터 준비
    combined_data_df = combined_data()

    # 텍스트와 레이블 설정
    X = combined_data_df['메일제목']
    y = combined_data_df['메일종류'].apply(lambda x: 1 if x == '스팸' else 0)

    # 훈련 세트와 검증 세트로 분리
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # TF-IDF 벡터화
    tfidf = TfidfVectorizer(max_features=3000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)

    # K-NN 분류 모델 학습
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_tfidf, y_train)

    # 모델 저장
    joblib.dump(tfidf, 'tfidf_model.pkl')
    joblib.dump(knn, 'knn_model.pkl')
    print("모델이 학습되고 저장되었습니다.")

    # 검증 데이터에 대한 예측 및 정확도 평가
    y_val_pred = knn.predict(X_val_tfidf)
    accuracy = accuracy_score(y_val, y_val_pred)
    conf_matrix = confusion_matrix(y_val, y_val_pred)

    print("검증 데이터 정확도:", accuracy)
    print("혼동 행렬:\n", conf_matrix)

    return knn

def load_test_data_and_predict():
    # 저장된 모델 로드
    tfidf = joblib.load('tfidf_model.pkl')
    knn = joblib.load('knn_model.pkl')

    # 테스트 데이터 로드 및 준비
    X_test = load_and_prepare_test_data()

    # TF-IDF 벡터화 (테스트 데이터)
    X_test_tfidf = tfidf.transform(X_test)

    # 예측
    y_pred = knn.predict(X_test_tfidf)

    # 예측 결과 출력
    for prediction in y_pred:
        result = "스팸" if prediction == 1 else "햄"
        print(f"이 이메일은: {result}")

# 모델 학습
train_model()


model = joblib.load("tfidf_model.pkl")

# 테스트 데이터로 예측
load_test_data_and_predict()