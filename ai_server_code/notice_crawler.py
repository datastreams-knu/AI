import os
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_upstage import UpstageEmbeddings
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pinecone import Pinecone
from langchain_upstage import ChatUpstage
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document
from langchain.vectorstores import FAISS
import re
from datetime import datetime
import pytz
from langchain.schema.runnable import Runnable
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain.schema.runnable import RunnableSequence, RunnableMap
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from konlpy.tag import Okt
from collections import defaultdict
import Levenshtein
import numpy as np
from IPython.display import display, HTML
import numpy as np
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt
from difflib import SequenceMatcher
import pandas as pd
from pymongo import MongoClien

# Pinecone API 키와 인덱스 이름 선언
pinecone_api_key = 'cd22a6ee-0b74-4e9d-af1b-a1e83917d39e'  # 여기에 Pinecone API 키를 입력
index_name = 'prof'

# Upstage API 키 선언
upstage_api_key = 'up_pGRnryI1JnrxChGycZmswEZm934Tf'  # 여기에 Upstage API 키를 입력

# Pinecone API 설정 및 초기화
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)

# mongodb 연결, client로
client = MongoClient("mongodb://localhost:27017/")

db = client["test_database"]
collection = db["test_collection"]

# 데이터 삽입
# 하나의 문서 삽입

def get_korean_time():
    return datetime.now(pytz.timezone('Asia/Seoul'))

# URL에서 제목, 날짜, 내용(본문 텍스트와 이미지 URL) 추출하는 공지사항 함수
def extract_text_and_date_from_url(urls):
    all_data = []

    def fetch_text_and_date(url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # 제목 추출
            title_element = soup.find('span', class_='bo_v_tit')
            title = title_element.get_text(strip=True) if title_element else "Unknown Title"

            # 본문 텍스트와 이미지 URL을 분리하여 저장
            text_content = "Unknown Content"  # 텍스트 초기화
            image_content = []  # 이미지 URL을 담는 리스트 초기화

            # 본문 내용 추출
            paragraphs = soup.find('div', id='bo_v_con')
            if paragraphs:
                # 텍스트 추출
                text_content = "\n".join([para.get_text(strip=True) for para in paragraphs.find_all('p')])

                # 이미지 URL 추출
                for img in paragraphs.find_all('img'):
                    img_src = img.get('src')
                    if img_src:
                        image_content.append(img_src)

            # 날짜 추출
            date_element = soup.select_one("strong.if_date")  # 수정된 선택자
            date = date_element.get_text(strip=True) if date_element else "Unknown Date"

            # 제목이 Unknown Title이 아닐 때만 데이터 추가
            if title != "Unknown Title":
                return title, text_content, image_content, date, url  # 문서 제목, 본문 텍스트, 이미지 URL 리스트, 날짜, URL 반환
            else:
                return None, None, None, None, None  # 제목이 Unknown일 경우 None 반환
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return None, None, None, None, url

    with ThreadPoolExecutor() as executor:
        results = executor.map(fetch_text_and_date, urls)

    # 유효한 데이터만 추가
    all_data = [(title, text_content, image_content, date, url) for title, text_content, image_content, date, url in results if title is not None]
    return all_data


# 교수 정보 추출 함수 1, 2 ,3
# input : url 리스트
# ouput : list of (title, text_content, image_content, date, prof_url)
def extract_professor_info_from_urls(urls):
    all_data = []

    def fetch_professor_info(url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            # 교수 정보가 담긴 요소들 선택
            professor_elements = soup.find("div", id="dr").find_all("li")

            for professor in professor_elements:
                # 이미지 URL 추출
                image_element = professor.find("div", class_="dr_img").find("img")
                image_content = image_element["src"] if image_element else "Unknown Image URL"

                # 이름 추출
                name_element = professor.find("div", class_="dr_txt").find("h3")
                title = name_element.get_text(strip=True) if name_element else "Unknown Name"

                # 연락처와 이메일 추출 후 하나의 텍스트로 결합
                contact_info = professor.find("div", class_="dr_txt").find_all("dd")
                contact_number = contact_info[0].get_text(strip=True) if len(contact_info) > 0 else "Unknown Contact Number"
                email = contact_info[1].get_text(strip=True) if len(contact_info) > 1 else "Unknown Email"
                text_content = f"{title}, {contact_number}, {email}"

                # 날짜와 URL 설정
                date = "작성일24-01-01 00:00"

                prof_url_element = professor.find("a")
                prof_url = prof_url_element["href"] if prof_url_element else "Unknown URL"

                # 각 교수의 정보를 all_data에 추가
                all_data.append((title, text_content, image_content, date, prof_url))

        except Exception as e:
            print(f"Error processing {url}: {e}")

    # ThreadPoolExecutor를 이용하여 병렬 크롤링
    with ThreadPoolExecutor() as executor:
        executor.map(fetch_professor_info, urls)

    return all_data

def extract_professor_info_from_urls_2(urls):
    all_data = []

    def fetch_professor_info(url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            # 교수 정보가 담긴 요소들 선택
            professor_elements = soup.find("div", id="Student").find_all("li")

            for professor in professor_elements:
                # 이미지 URL 추출
                image_element = professor.find("div", class_="img").find("img")
                image_content = image_element["src"] if image_element else "Unknown Image URL"

                # 이름 추출
                name_element = professor.find("div", class_="cnt").find("div", class_="name")
                title = name_element.get_text(strip=True) if name_element else "Unknown Name"

                # 연락처와 이메일 추출
                contact_place = professor.find("div", class_="dep").get_text(strip=True) if professor.find("div", class_="dep") else "Unknown Contact Place"
                email_element = professor.find("dl", class_="email").find("dd").find("a")
                email = email_element.get_text(strip=True) if email_element else "Unknown Email"

                # 텍스트 내용 조합
                text_content = f"성함(이름):{title}, 연구실(장소):{contact_place}, 이메일:{email}"

                # 날짜와 URL 설정
                date = "작성일24-01-01 00:00"
                prof_url = url

                # 각 교수의 정보를 all_data에 추가
                all_data.append((title, text_content, image_content, date, prof_url))

        except Exception as e:
            print(f"Error processing {url}: {e}")

    # ThreadPoolExecutor를 이용하여 병렬 크롤링
    with ThreadPoolExecutor() as executor:
        executor.map(fetch_professor_info, urls)

    return all_data

def extract_professor_info_from_urls_3(urls):
    all_data = []

    def fetch_professor_info(url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            # 교수 정보가 담긴 요소들 선택
            professor_elements = soup.find("div", id="Student").find_all("li")

            for professor in professor_elements:
                # 이미지 URL 추출
                image_element = professor.find("div", class_="img").find("img")
                image_content = image_element["src"] if image_element else "Unknown Image URL"

                # 이름 추출
                name_element = professor.find("div", class_="cnt").find("h1")
                title = name_element.get_text(strip=True) if name_element else "Unknown Name"

                # 연락처 추출
                contact_number_element = professor.find("span", class_="period")
                contact_number = contact_number_element.get_text(strip=True) if contact_number_element else "Unknown Contact Number"

                # 연구실 위치 추출
                contact_info = professor.find_all("dl", class_="dep")
                contact_place = contact_info[0].find("dd").get_text(strip=True) if len(contact_info) > 0 else "Unknown Contact Place"

                # 이메일 추출
                email = contact_info[1].find("dd").find("a").get_text(strip=True) if len(contact_info) > 1 else "Unknown Email"

                # 텍스트 내용 조합
                text_content = f"성함(이름):{title}, 연락처(전화번호):{contact_number}, 사무실(장소):{contact_place}, 이메일:{email}"

                # 날짜와 URL 설정
                date = "작성일24-01-01 00:00"
                prof_url = url

                # 각 교수의 정보를 all_data에 추가
                all_data.append((title, text_content, image_content, date, prof_url))

        except Exception as e:
            print(f"Error processing {url}: {e}")

    # ThreadPoolExecutor를 이용하여 병렬 크롤링
    with ThreadPoolExecutor() as executor:
        executor.map(fetch_professor_info, urls)

    return all_data


# 최신 wr_id 추출 함수
def get_latest_wr_id():
    url = "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_1"
    response = requests.get(url)
    if response.status_code == 200:
        match = re.search(r'wr_id=(\d+)', response.text)
        if match:
            return int(match.group(1))
    return None

# text -> chunk splitter.
class CharacterTextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i:i + self.chunk_size])
        return chunks


####################################################################
#                   사용자 질문 명사화 함수                         #
####################################################################
def transformed_query(user_question):
    # 중복된 단어를 제거한 명사를 담을 리스트
    query_nouns = []

    # 1. 숫자와 특정 단어가 결합된 패턴 추출 (예: '2024학년도', '1월' 등)
    pattern = r'\d+(?:학년도|년|월|일|학기|시|분|초)?'
    number_matches = re.findall(pattern, user_question)
    query_nouns += number_matches
    # 추출된 단어를 content에서 제거
    for match in number_matches:
        user_question = user_question.replace(match, '')

    # 2. 영어와 한글이 붙어 있는 패턴 추출 (예: 'SW전공' 등)
    eng_kor_pattern = r'\b[a-zA-Z]+[가-힣]+\b'
    eng_kor_matches = re.findall(eng_kor_pattern, user_question)
    query_nouns += eng_kor_matches
    # 추출된 단어를 content에서 제거
    for match in eng_kor_matches:
        user_question = user_question.replace(match, '')

    # 3. 영어 단어 단독으로 추출
    english_words = re.findall(r'\b[a-zA-Z]+\b', user_question)
    query_nouns += english_words
    # 추출된 단어를 content에서 제거
    for match in english_words:
        user_question = user_question.replace(match, '')
    # 4. "튜터"라는 단어가 포함되어 있으면 "TUTOR" 추가
    if '튜터' in user_question:
        query_nouns.append('TUTOR')
        user_question = user_question.replace('튜터', '')  # '튜터' 제거
    if '탑싯' in user_question:
        query_nouns.append('TOPCIT')
        user_question = user_question.replace('탑싯','')
    if '시험' in user_question:
        query_nouns.append('시험')
    if '하계' in user_question:
        query_nouns.append('여름')
        query_nouns.append('하계')
    if '동계' in user_question:
        query_nouns.append('겨울')
        query_nouns.append('동계')
    if '겨울' in user_question:
        query_nouns.append('겨울')
        query_nouns.append('동계')
    if '여름' in user_question:
        query_nouns.append('여름')
        query_nouns.append('하계')
    if '성인지' in user_question:
        query_nouns.append('성인지')
    if '첨성인' in user_question:
        query_nouns.append('첨성인')
    if '글솦' in user_question:
        query_nouns.append('글솝')
    if '수꾸' in user_question:
        query_nouns.append('수강꾸러미')
        
    # 5. Okt 형태소 분석기를 이용한 추가 명사 추출
    okt = Okt()
    additional_nouns = [noun for noun in okt.nouns(user_question) if len(noun) >= 1]
    query_nouns += additional_nouns
    
    # "공지", "사항", "공지사항"을 query_nouns에서 제거
    remove_noticement = ['공지', '사항', '공지사항','필독','첨부파일']
    query_nouns = [noun for noun in query_nouns if noun not in remove_noticement]
    
    # 6. "수강" 단어와 관련된 키워드 결합 추가
    if '수강' in user_question:
        related_keywords = ['변경', '신청', '정정', '취소','꾸러미']
        for keyword in related_keywords:
            if keyword in user_question:
                # '수강'과 결합하여 새로운 키워드 추가
                combined_keyword = '수강' + keyword
                query_nouns.append(combined_keyword)
                if ('수강' in query_nouns):
                  query_nouns.remove('수강')
                for keyword in related_keywords:
                  if keyword in query_nouns:
                    query_nouns.remove(keyword)
                    
    # 최종 명사 리스트에서 중복된 단어 제거
    query_nouns = list(set(query_nouns))
    return query_nouns


'''
# 본격적인 크롤링 시작
'''

# 스크래핑할 URL 목록 생성
now_number = get_latest_wr_id()
urls = []
for number in range(now_number, 27726, -1):     #2024-08-07 수강신청 안내시작..28148
    urls.append("https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_1&wr_id=" + str(number))

# 교수진 페이지 URL 목록 리스트
prof_urls = [
    "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub2_1&lang=kor",
    "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub2_2&lang=kor",
    "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub2_5&lang=kor"
]

document_data = extract_text_and_date_from_url(urls) # URL에서 문서 내용 추출 결과 리스트
prof_data = extract_professor_info_from_urls(prof_urls[0])
prof_data_2 = extract_professor_info_from_urls_2(prof_urls[1])
prof_data_3 = extract_professor_info_from_urls_3(prof_urls[2])
combined_prof_data = prof_data + prof_data_2 + prof_data_3 # url에서 교수 정보 추출 결과 리스트

# chunck 크기따라 text 분리 객체 생성.
# chunk_size=1000, chunk_overlap=150
text_splitter = CharacterTextSplitter() # 텍스트 분리 및 URL과 날짜 매핑

index_of_docs = 0   # 데이터베이스에 인덱스로 접근하기 위한 변수, html의 wrid로 하면 될듯?
for title, doc, image, date, url in document_data:
    if isinstance(doc, str) and doc.strip():  # doc가 문자열인지 확인하고 비어있지 않은지 확인
        texts = text_splitter.split_text(doc)
        titles = [title] * len(texts)  # 제목을 분리된 텍스트와 동일한 길이로 추가
        doc_urls = [url] * len(texts)
        doc_dates = [date] * len(texts)  # 분리된 각 텍스트에 동일한 날짜 적용

        # 이미지 URL도 저장
        if image:  # 이미지 URL이 비어 있지 않은 경우
            image_url = [image] * len(texts)  # 동일한 길이로 이미지 URL 추가
        else:  # 이미지 URL이 비어 있는 경우
            image_url = ["No content"] * len(texts)  # "No content" 추가

    elif image:  # doc가 비어 있고 이미지가 있는 경우
        # 텍스트는 "No content"로 추가
        texts = "No content"
        titles = title
        doc_urls = url
        doc_dates = date
        image_url = image  # 이미지 URL 추가

    else:  # doc와 image가 모두 비어 있는 경우
        texts = "No content"
        image_url = "No content"  # 이미지도 "No content"로 추가
        titles = title
        doc_urls = url
        doc_dates = date
    
    #데이터베이스에 저장
    data = {
        "_id" : index_of_docs,
        "text" : texts,
        "title" : titles,
        "transformed_title" : transformed_query(title),
        "image_url" : image_url,
        "doc_urls" : doc_urls,
        "doc_date" : date
    }
    collection.insert_one(data)
    index_of_docs += 1


# prof_data는 extract_professor_info_from_urls 함수의 반환값
for title, doc, image, date, url in combined_prof_data :
    if isinstance(doc, str) and doc.strip():  # 교수 정보가 문자열로 있고 비어있지 않을 때
        texts = text_splitter.split_text(doc)
        titles = [title] * len(texts)  # 교수 이름을 분리된 텍스트와 동일한 길이로 추가
        doc_urls = [url] * len(texts)
        doc_dates = [date] * len(texts)  # 분리된 각 텍스트에 동일한 날짜 적용

        # 이미지 URL도 저장
        if image:  # 이미지 URL이 비어 있지 않은 경우
            image_urls = [image] * len(texts)  # 동일한 길이로 이미지 URL 추가
        else:
            image_urls = ["No content"] * len(texts)  # "No content" 추가

    elif image:  # doc가 비어 있고 이미지가 있는 경우
        texts = "No content"
        titles = title
        doc_urls = url
        doc_dates = date
        image_urls = image  # 이미지 URL 추가

    else:  # doc와 image가 모두 비어 있는 경우
        texts = "No content"
        image_urls = "No content"  # 이미지도 "No content"로 추가
        titles = title
        doc_urls = url
        doc_dates = date
    
    data = {
        "_id" : index_of_docs,
        "text" : texts,
        "title" : titles,
        "transformed_title" : transformed_query(title),
        "image_url" : image_url,
        "doc_urls" : doc_urls,
        "doc_dates" : date
    }
    collection.insert_one(data)
    index_of_docs += 1


'''
transformed 된 user query BM25 유사도 계산 모듈
데이터 베이스에 있는 자료에서 찾는 코드로 변경할 필요가 있음.
'''
tokenized_titles = [doc["transformed_title"] for doc in collection.find({}, {"transformed_title": 1, "_id": 0})]
# 기존과 동일한 파라미터를 사용하고 있는지 확인
bm25_titles = BM25Okapi(tokenized_titles, k1=1.5, b=0.75)  # 기존 파라미터 확인

# Dense Retrieval (Upstage 임베딩)
embeddings = UpstageEmbeddings(
  api_key = upstage_api_key,
  model="solar-embedding-1-large"
) # Upstage API 키 사용

dense_doc_vectors = np.array(embeddings.embed_documents(texts))  # 문서 임베딩
# Pinecone에 문서 임베딩 저장 (문서 텍스트와 URL, 날짜를 메타데이터에 포함)
for i, embedding in enumerate(dense_doc_vectors):
    metadata = {
        "title": titles[i],
        "text": texts[i],
        "url": doc_urls[i],  # URL 메타데이터
        "date": doc_dates[i]  # 날짜 메타데이터 추가
    }
    index.upsert([(str(i), embedding.tolist(), metadata)])  # 문서 ID, 임베딩 벡터, 메타데이터 추가


