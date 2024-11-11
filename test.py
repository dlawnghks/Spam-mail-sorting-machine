import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터와 모델을 저장할 전역 변수
tfidf = None
knn = None

def combined_data():
    # 데이터 불러오기
    google_message = pd.read_csv("C:/Project/Rabble/datasets/messages.csv")
    spam_message = pd.read_csv("C:/Project/Rabble/datasets/spam_virus.csv")

    # '수신일자 '수신시간'을 합쳐 '날짜' 열 생성
    spam_message['날짜'] = pd.to_datetime(spam_message['수신일자'] + ' ' + spam_message['수신시간'], format='%Y-%m-%d %H:%M')
    spam_message = spam_message.drop(['수신일자', '수신시간'], axis=1)

    # 수신시간을 저장할 새로운 열 생성
    google_message['수신시간'] = None

    # 날짜 데이터 형식 변경
    if google_message['날짜'].dtype == 'object':
        google_message['날짜'] = pd.to_datetime(google_message['날짜'])
    if spam_message['날짜'].dtype == 'object':
        spam_message['날짜'] = pd.to_datetime(spam_message['날짜'])

    # 메일 종류 추가
    google_message['메일종류'] = '햄'
    spam_message['메일종류'] = '스팸'  # 스팸 메일에도 '메일종류' 열 추가

    # 필요한 열 선택 (메일내용 열 제거)
    final_google_df = google_message[['날짜', '메일종류', '메일제목']]  # 메일내용 제거
    final_spam_df = spam_message[['날짜', '메일종류', '메일제목']]  # 메일내용 제거

    # Ham 데이터 수
    ham_count = final_google_df.shape[0]

    # Spam 데이터 언더샘플링
    final_spam_df = final_spam_df.sample(n=ham_count, random_state=42)

    # 데이터 결합
    combined_df = pd.concat([final_google_df, final_spam_df], axis=0, ignore_index=True)

    return combined_df

# 데이터 준비
data = combined_data()

# 텍스트와 레이블 설정 (메일내용 제거)
X = data['메일제목']  # 메일 제목을 특징으로 사용
y = data['메일종류'].apply(lambda x: 1 if x == '스팸' else 0)  # 스팸: 1, 햄: 0으로 레이블 인코딩

# TF-IDF 벡터화
tfidf = TfidfVectorizer(max_features=3000)  # 주요 특징 3000개까지 제한
X_tfidf = tfidf.fit_transform(X)

# 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# K-NN 분류 모델 학습
k = 5  # K값 설정
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# 예측
y_pred = knn.predict(X_test)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print("정확도:", accuracy)

# 이메일 분류 함수
def classify_email(new_email_subject):
    global tfidf, knn  # 전역 변수로 선언
    new_email_tfidf = tfidf.transform([new_email_subject])  # 단일 이메일 제목을 리스트로 감싸서 변환
    prediction = knn.predict(new_email_tfidf)  # 예측
    return "스팸" if prediction[0] == 1 else "햄"  # 스팸이면 1, 햄이면 0

# 사용 예시
new_email_subject = "여기에 새로 온 이메일의 제목이 들어갑니다."
result = classify_email(new_email_subject)
print("이 이메일은:", result)
