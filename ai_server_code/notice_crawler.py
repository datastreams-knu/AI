# %pip install -U langchain-community
# %pip install beautifulsoup4 requests scikit-learn pinecone-client numpy langchain-upstage faiss-cpu
# %pip install langchain
# %pip install nltk
# !pip install spacy
# !python -m spacy download ko_core_news_sm
# # 필요한 시스템 패키지 설치
# !apt-get install -y python3-dev
# !apt-get install -y libmecab-dev
# !apt-get install -y mecab mecab-ko mecab-ko-dic
# %pip install rank_bm25
# # konlpy 설치
# !pip install konlpy
# !pip install python-Levenshtein
# !pip install sentence-transformers
# !pip install pymongo

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
from rank_bm25 import BM25Okapi
from difflib import SequenceMatcher
from pymongo import MongoClient

# Pinecone API 키와 인덱스 이름 선언
pinecone_api_key = 'cd22a6ee-0b74-4e9d-af1b-a1e83917d39e'
index_name = 'db1'

# Upstage API 키 선언
upstage_api_key = 'up_pGRnryI1JnrxChGycZmswEZm934Tf'


# Pinecone API 설정 및 초기화
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)
def get_korean_time():
    return datetime.now(pytz.timezone('Asia/Seoul'))

# mongodb 연결, client로
client = MongoClient("mongodb://localhost:27017/")

db = client["knu_chatbot"]
collection = db["notice_collection"]

qqqqqqqqqqqq
# 단어 명사화 함수.
def transformed_query(content):

    # 중복된 단어를 제거한 명사를 담을 리스트
    query_nouns = []

    # 1. 숫자와 특정 단어가 결합된 패턴 추출 (예: '2024학년도', '1월' 등)
    pattern = r'\d+(?:학년도|년|월|일|학기|시|분|초|기|개)?'
    number_matches = re.findall(pattern, content)
    query_nouns += number_matches
    # 추출된 단어를 content에서 제거
    for match in number_matches:
        content = content.replace(match, '')

    # 2. 영어와 한글이 붙어 있는 패턴 추출 (예: 'SW전공' 등)
    eng_kor_pattern = r'\b[a-zA-Z]+[가-힣]+\b'
    eng_kor_matches = re.findall(eng_kor_pattern, content)
    query_nouns += eng_kor_matches
    # 추출된 단어를 content에서 제거
    for match in eng_kor_matches:
        content = content.replace(match, '')

    # 3. 영어 단어 단독으로 추출
    english_words = re.findall(r'\b[a-zA-Z]+\b', content)
    query_nouns += english_words
    # 추출된 단어를 content에서 제거
    for match in english_words:
        content = content.replace(match, '')
    # 4. "튜터"라는 단어가 포함되어 있으면 "TUTOR" 추가
    if '튜터' in content:
        query_nouns.append('TUTOR')
        content = content.replace('튜터', '')  # '튜터' 제거
    if '탑싯' in content:
        query_nouns.append('TOPCIT')
        content=content.replace('탑싯','')
    if '시험' in content:
        query_nouns.append('시험')
    if '하계' in content:
        query_nouns.append('여름')
        query_nouns.append('하계')
    if '동계' in content:
        query_nouns.append('겨울')
        query_nouns.append('동계')
    if '겨울' in content:
        query_nouns.append('겨울')
        query_nouns.append('동계')
    if '여름' in content:
        query_nouns.append('여름')
        query_nouns.append('하계')
    if '성인지' in content:
        query_nouns.append('성인지')
    if '첨성인' in content:
        query_nouns.append('첨성인')
    if '글솦' in content:
        query_nouns.append('글솝')
    if '수꾸' in content:
        query_nouns.append('수강꾸러미')
    if '장학금' in content:
        query_nouns.append('장학생')
        query_nouns.append('장학')
    if '장학생' in content:
        query_nouns.append('장학금')
        query_nouns.append('장학')
    if '대해' in content:
        content=content.replace('대해','')
    # 비슷한 의미 모두 추가 (세미나)
    related_keywords = ['세미나', '행사', '특강', '강연']
    if any(keyword in content for keyword in related_keywords):
        for keyword in related_keywords:
            query_nouns.append(keyword)
    # "공지", "사항", "공지사항"을 query_nouns에서 '공지사항'이라고 고정하고 나머지 부분 삭제
    keywords=['공지','사항','공지사항']
    if any(keyword in content for keyword in keywords):
      # 키워드 제거
      for keyword in keywords:
          content = content.replace(keyword, '')
          query_nouns.append('공지사항')  # 'query_noun'에 추가
    # 5. Okt 형태소 분석기를 이용한 추가 명사 추출
    okt = Okt()
    additional_nouns = [noun for noun in okt.nouns(content) if len(noun) > 1]
    query_nouns += additional_nouns


    # 6. "수강" 단어와 관련된 키워드 결합 추가
    if '수강' in content:
        related_keywords = ['변경', '신청', '정정', '취소','꾸러미']
        for keyword in related_keywords:
            if keyword in content:
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

####공지사항 크롤링하는 코드 ################
def get_latest_wr_id():
    url = "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_1"
    response = requests.get(url)
    if response.status_code == 200:
        match = re.search(r'wr_id=(\d+)', response.text)
        if match:
            return int(match.group(1))
    return None


# 스크래핑할 URL 목록 생성
now_number = get_latest_wr_id()
urls = []
for number in range(now_number, 27726, -1):     #2024-08-07 수강신청 안내시작..28148
    urls.append("https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_1&wr_id=" + str(number))

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
                # paragraphs 내부에서 'p', 'div', 'li' 태그 텍스트 추출
                text_content = "\n".join([element.get_text(strip=True) for element in paragraphs.find_all(['p', 'div', 'li'])])
                #print(text_content)
                if text_content.strip() == "":
                    text_content = ""
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

#### 크롤링한 공지사항 정보 document_data에 저장
document_data = extract_text_and_date_from_url(urls)
################################################################################################

# 텍스트 분리기 초기화
class CharacterTextSplitter:
    def __init__(self, chunk_size=1100, chunk_overlap=150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i:i + self.chunk_size])
        return chunks

text_splitter = CharacterTextSplitter(chunk_size=1100, chunk_overlap=150)

################################################################################################

# 텍스트 분리 및 URL과 날짜 매핑
texts = []
image_url=[]
titles = []
doc_urls = []
doc_dates = []

for title, doc, image, date, url in document_data:
    if isinstance(doc, str) and doc.strip():  # doc가 문자열인지 확인하고 비어있지 않은지 확인
        split_texts = text_splitter.split_text(doc)
        texts.extend(split_texts)
        titles.extend([title] * len(split_texts))  # 제목을 분리된 텍스트와 동일한 길이로 추가
        doc_urls.extend([url] * len(split_texts))
        doc_dates.extend([date] * len(split_texts))  # 분리된 각 텍스트에 동일한 날짜 적용
        
        # 이미지 URL도 저장
        if image:  # 이미지 URL이 비어 있지 않은 경우
            image_url.extend([image] * len(split_texts))  # 동일한 길이로 이미지 URL 추가
        else:  # 이미지 URL이 비어 있는 경우
            image_url.extend(["No content"] * len(split_texts))  # "No content" 추가
            image = "No content"

    elif image:  # doc가 비어 있고 이미지가 있는 경우
        # 텍스트는 "No content"로 추가
        texts.append("No content")
        titles.append(title)
        doc_urls.append(url)
        doc_dates.append(date)
        image_url.append(image)  # 이미지 URL 추가

    else:  # doc와 image가 모두 비어 있는 경우
        texts.append("No content")
        image_url.append("No content")  # 이미지도 "No content"로 추가
        titles.append(title)
        doc_urls.append(url)
        doc_dates.append(date)
        image = "No content"
    
    temp_data = {
        "title" : title,
        "image_url" : image
    	}
    if not collection.find_one(temp_data):
        collection.insert_one(temp_data)
        print("Document inserted.")
    else:
        print("Duplicate document. Skipping insertion.")



######################################################################################################
########################### 지금까지 공지사항 정보 ###################################################
######################################################################################################


######   정교수진의 정보 받아오는 코드 ##########

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
        results = executor.map(fetch_professor_info, urls)

    return all_data


######   초빙교수진의 정보 받아오는 코드 ##########

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

######   직원의 정보 받아오는 코드 ##########

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

                # 담당 업무 추출
                role = contact_info[2].find("dd").get_text(strip=True) if len(contact_info) > 2 else "Unknown Role"

                # 텍스트 내용 조합
                text_content = f"성함(이름):{title}, 연락처(전화번호):{contact_number}, 사무실(장소):{contact_place}, 이메일:{email}, 담당업무:{role}"

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


# 교수진 페이지 URL 목록
urls2 = [
    "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub2_1&lang=kor",
]
urls3 = [
    "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub2_2&lang=kor",
]
urls4 = [
    "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub2_5&lang=kor",
]

prof_data = extract_professor_info_from_urls(urls2)
prof_data_2 = extract_professor_info_from_urls_2(urls3)
prof_data_3 = extract_professor_info_from_urls_3(urls4)

combined_prof_data = prof_data + prof_data_2 + prof_data_3

# 교수 정보 크롤링 데이터 분리 및 저장
professor_texts = []
professor_image_urls = []
professor_titles = []
professor_doc_urls = []
professor_doc_dates = []

# prof_data는 extract_professor_info_from_urls 함수의 반환값
for title, doc, image, date, url in combined_prof_data :
    if isinstance(doc, str) and doc.strip():  # 교수 정보가 문자열로 있고 비어있지 않을 때
        split_texts = text_splitter.split_text(doc)
        professor_texts.extend(split_texts)
        professor_titles.extend([title] * len(split_texts))  # 교수 이름을 분리된 텍스트와 동일한 길이로 추가
        professor_doc_urls.extend([url] * len(split_texts))
        professor_doc_dates.extend([date] * len(split_texts))  # 분리된 각 텍스트에 동일한 날짜 적용
	
        # 이미지 URL도 저장
        if image:  # 이미지 URL이 비어 있지 않은 경우
            professor_image_urls.extend([image] * len(split_texts))  # 동일한 길이로 이미지 URL 추가
        else:
            professor_image_urls.extend(["No content"] * len(split_texts))  # "No content" 추가
            image = "No content"
            

    elif image:  # doc가 비어 있고 이미지가 있는 경우
        professor_texts.append("No content")
        professor_titles.append(title)
        professor_doc_urls.append(url)
        professor_doc_dates.append(date)
        professor_image_urls.append(image)  # 이미지 URL 추가

    else:  # doc와 image가 모두 비어 있는 경우
        professor_texts.append("No content")
        professor_image_urls.append("No content")  # 이미지도 "No content"로 추가
        professor_titles.append(title)
        professor_doc_urls.append(url)
        professor_doc_dates.append(date)
        image = "No content"
    
    temp_data = {
        "title" : title,
        "image_url" : image
    }
    if not collection.find_one(temp_data):
        collection.insert_one(temp_data)
        print("Document inserted.")
    else:
        print("Duplicate document. Skipping insertion.")
    
# 교수 정보 데이터를 기존 데이터와 합치기
texts.extend(professor_texts)
image_url.extend(professor_image_urls)
titles.extend(professor_titles)
doc_urls.extend(professor_doc_urls)
doc_dates.extend(professor_doc_dates)


######################################################################################################
###########################교수 및 직원정보임 위는  ##################################################
######################################################################################################



####### 취업정보를 받아오는 코드 #######

def get_latest_wr_id_1():
    url = "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_3_b"
    response = requests.get(url)
    if response.status_code == 200:
        # re.findall을 사용하여 모든 wr_id 값을 찾아 리스트로 반환
        match = re.findall(r'wr_id=(\d+)', response.text)
        if match:
        # wr_ids 리스트에 있는 모든 wr_id 값을 출력
          max_wr_id = max(int(wr_id) for wr_id in match)
          return max_wr_id
    return None


now_company_number=get_latest_wr_id_1()

company_urls=[]
for number in range(now_company_number,1149,-1):
  company_urls.append("https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_3_b&wr_id="+str(number))


def extract_company_from_url(urls):
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
                # paragraphs 내부에서 'p', 'div', 'li' 태그 텍스트 추출
                text_content = "\n".join([element.get_text(strip=True) for element in paragraphs.find_all(['p', 'div', 'li'])])
                #print(text_content)
                if text_content.strip() == "":
                    text_content = ""
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

company_data= extract_company_from_url(company_urls)

for title, doc, image, date, url in company_data:
    if isinstance(doc, str) and doc.strip():  # doc가 문자열인지 확인하고 비어있지 않은지 확인
        split_texts = text_splitter.split_text(doc)
        texts.extend(split_texts)
        titles.extend([title] * len(split_texts))  # 제목을 분리된 텍스트와 동일한 길이로 추가
        doc_urls.extend([url] * len(split_texts))
        doc_dates.extend([date] * len(split_texts))  # 분리된 각 텍스트에 동일한 날짜 적용
        
        # 이미지 URL도 저장
        if image:  # 이미지 URL이 비어 있지 않은 경우
            image_url.extend([image] * len(split_texts))  # 동일한 길이로 이미지 URL 추가
        else:  # 이미지 URL이 비어 있는 경우
            image_url.extend(["No content"] * len(split_texts))  # "No content" 추가
            image = "No content"
            
    elif image:  # doc가 비어 있고 이미지가 있는 경우
        # 텍스트는 "No content"로 추가
        texts.append("No content")
        titles.append(title)
        doc_urls.append(url)
        doc_dates.append(date)
        image_url.append(image)  # 이미지 URL 추가

    else:  # doc와 image가 모두 비어 있는 경우
        texts.append("No content")
        image_url.append("No content")  # 이미지도 "No content"로 추가
        image = "No content"
        titles.append(title)
        doc_urls.append(url)
        doc_dates.append(date)

    temp_data = {
        "title" : title,
        "image_url" : image
    }
    if not collection.find_one(temp_data):
        collection.insert_one(temp_data)
        print("Document inserted.")
    else:
        print("Duplicate document. Skipping insertion.")

######################################################################################################
###########################취업정보임 위는  ##########################################################
######################################################################################################



def get_latest_wr_id_2():
    url = "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_4"
    response = requests.get(url)
    if response.status_code == 200:
        # re.findall을 사용하여 모든 wr_id 값을 찾아 리스트로 반환
        match = re.findall(r'wr_id=(\d+)', response.text)
        if match:
        # wr_ids 리스트에 있는 모든 wr_id 값을 출력
          max_wr_id = max(int(wr_id) for wr_id in match)
          return max_wr_id
    return None


now_seminar_number=get_latest_wr_id_2()

seminar_urls=[]
for number in range(now_seminar_number,246,-1):
  seminar_urls.append("https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_4&wr_id="+str(number))


def extract_seminar_from_url(urls):
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
                # paragraphs 내부에서 'p', 'div', 'li' 태그 텍스트 추출
                text_content = "\n".join([element.get_text(strip=True) for element in paragraphs.find_all(['p', 'div', 'li'])])
                #print(text_content)
                if text_content.strip() == "":
                    text_content = ""
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

seminar_data= extract_seminar_from_url(seminar_urls)



for title, doc, image, date, url in seminar_data:
    if isinstance(doc, str) and doc.strip():  # doc가 문자열인지 확인하고 비어있지 않은지 확인
        split_texts = text_splitter.split_text(doc)
        texts.extend(split_texts)
        titles.extend([title] * len(split_texts))  # 제목을 분리된 텍스트와 동일한 길이로 추가
        doc_urls.extend([url] * len(split_texts))
        doc_dates.extend([date] * len(split_texts))  # 분리된 각 텍스트에 동일한 날짜 적용

        # 이미지 URL도 저장
        if image:  # 이미지 URL이 비어 있지 않은 경우
            image_url.extend([image] * len(split_texts))  # 동일한 길이로 이미지 URL 추가
        else:  # 이미지 URL이 비어 있는 경우
            image_url.extend(["No content"] * len(split_texts))  # "No content" 추가
            image = "No content"
    elif image:  # doc가 비어 있고 이미지가 있는 경우
        # 텍스트는 "No content"로 추가
        texts.append("No content")
        titles.append(title)
        doc_urls.append(url)
        doc_dates.append(date)
        image_url.append(image)  # 이미지 URL 추가

    else:  # doc와 image가 모두 비어 있는 경우
        texts.append("No content")
        image_url.append("No content")  # 이미지도 "No content"로 추가
        titles.append(title)
        doc_urls.append(url)
        doc_dates.append(date)
        image = "No content"
    
    temp_data = {
        "title" : title,
        "image_url" : image
    }
    if not collection.find_one(temp_data):
        collection.insert_one(temp_data)
        print("Document inserted.")
    else:
        print("Duplicate document. Skipping insertion.")

######################################################################################################
############################세미나임 위는  ##########################################################
######################################################################################################

# 밑에 코드는 초기에 한 번만 돌림. 실제 서버 돌릴 때는 사용 X
# Dense Retrieval (Upstage 임베딩)
embeddings = UpstageEmbeddings(
  api_key=upstage_api_key,
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
