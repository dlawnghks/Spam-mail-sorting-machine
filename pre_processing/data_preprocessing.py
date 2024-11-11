import pandas as pd
import re

def combined_data():
    # 데이터 불러오기
    google_message = pd.read_csv("C:/Project/Rabble/datasets/messages.csv")
    spam_message = pd.read_csv("C:/Project/Rabble/datasets/spam_virus.csv")

    selected_columns = google_message.iloc[:, [0, 1, 2]]


    # '수신일자 '수신시간'을 합쳐 '날짜' 열 생성
    spam_message['날짜'] = pd.to_datetime(spam_message['수신일자'] + ' ' + spam_message['수신시간'], format='%Y-%m-%d %H:%M')
    spam_message = spam_message.drop(['수신일자', '수신시간'], axis=1)

    # 수신시간을 저장할 새로운 열 생성
    google_message['수신시간'] = None

    # 수신시간 추출 함수 정의 (메일내용에서 '날짜:' 패턴을 찾아 시간 추출)
    def extract_time(mail_content):
        # 예시 패턴: '날짜: 2024. 10. 15. 오전 12:00'
        match = re.search(r'날짜:\s*\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.\s*(오전|오후)\s*(\d{1,2}):(\d{2})', mail_content)
        if match:
            period, hour, minute = match.groups()
            hour = int(hour)
            if period == '오후' and hour != 12:
                hour += 12
            elif period == '오전' and hour == 12:
                hour = 0
            return f"{hour:02d}:{minute}"
        return None

    # '내용' 열을 사용하여 '수신시간' 추출
    google_message['수신시간'] = google_message['메일내용'].apply(extract_time)

    if google_message['날짜'].dtype == 'object':
        google_message['날짜'] = pd.to_datetime(google_message['날짜'])

    if spam_message['날짜'].dtype == 'object':
        spam_message['날짜'] = pd.to_datetime(spam_message['날짜'])

    google_message['메일종류'] = '햄'

    # 필요한 열 선택
    final_google_df = google_message[['날짜', '메일종류', '메일제목']]
    final_spam_df = spam_message[['날짜','메일종류', '메일제목']]

    # 데이터 결합
    combined_df = pd.concat([final_google_df, final_spam_df], axis=0, ignore_index=True)

    print(combined_df.head())

    return combined_df
