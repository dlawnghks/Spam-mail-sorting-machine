import pandas as pd
import imaplib
import email
from email.header import decode_header
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 데이터와 모델을 저장할 전역 변수
tfidf = None
knn = None

def combined_data():
    # 데이터 불러오기
    google_message = pd.read_csv("C:/Project/Rabble/datasets/messages.csv")
    spam_message = pd.read_csv("C:/Project/Rabble/datasets/spam_virus.csv")

    google_message['메일종류'] = '햄'
    spam_message['메일종류'] = '스팸'

    # '메일내용'이 없는 경우 빈 문자열로 설정
    if '메일내용' not in spam_message.columns:
        spam_message['메일내용'] = ''

    # 필요한 열만 선택
    final_google_df = google_message[['메일종류', '메일제목', '메일내용']]
    final_spam_df = spam_message[['메일종류', '메일제목', '메일내용']]

    # 스팸과 햄 메일의 개수를 동일하게 조정
    ham_count = final_google_df.shape[0]
    final_spam_df = final_spam_df.sample(n=ham_count, random_state=42)

    combined_df = pd.concat([final_google_df, final_spam_df], axis=0, ignore_index=True)

    return combined_df

# 데이터 준비
data = combined_data()

# 텍스트와 레이블 설정
X = data['메일내용']
y = data['메일종류'].apply(lambda x: 1 if x == '스팸' else 0)

# TF-IDF 벡터화
tfidf = TfidfVectorizer(max_features=3000)
X_tfidf = tfidf.fit_transform(X)

# 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# K-NN 분류 모델 학습
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# 정확도 평가
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("정확도:", accuracy)

# 이메일 분류 함수
def classify_email(new_email_content):
    global tfidf, knn
    new_email_tfidf = tfidf.transform([new_email_content])
    prediction = knn.predict(new_email_tfidf)
    return "스팸" if prediction[0] == 1 else "햄"

# 이메일 서버에 연결
def connect_to_email(username, password):
    mail = imaplib.IMAP4_SSL('imap.gmail.com')  # Gmail IMAP 서버
    mail.login(username, password)
    return mail

# 새로운 이메일 가져오기
def fetch_new_emails(mail):
    mail.select('inbox')  # 인박스 선택
    result, data = mail.search(None, 'UNSEEN')  # 읽지 않은 메일 검색
    email_ids = data[0].split()

    emails = []
    for e_id in email_ids:
        result, msg_data = mail.fetch(e_id, '(RFC822)')
        msg = email.message_from_bytes(msg_data[0][1])
        subject, encoding = decode_header(msg['Subject'])[0]
        if isinstance(subject, bytes):
            subject = subject.decode(encoding if encoding else 'utf-8')
        emails.append({
            'subject': subject,
            'from': msg['From'],
            'body': get_email_body(msg)
        })
    return emails

# 이메일 본문 가져오기
def get_email_body(msg):
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                return part.get_payload(decode=True).decode()  # 본문 디코드
    else:
        return msg.get_payload(decode=True).decode()

# 이메일을 분류하는 메인 루프
def main():
    username = os.getenv("EMAIL_USERNAME")  # .env에서 이메일 가져오기
    password = os.getenv("EMAIL_PASSWORD")  # .env에서 비밀번호 가져오기
    mail = connect_to_email(username, password)

    while True:
        new_emails = fetch_new_emails(mail)
        for email in new_emails:
            result = classify_email(email['body'])  # 이메일 분류
            print(f"제목: {email['subject']}, 보낸사람: {email['from']}, 분류: {result}")

        # 잠시 대기 (10초 후에 다시 확인)
        import time
        time.sleep(10)

if __name__ == "__main__":
    main()
