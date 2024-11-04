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


# Pinecone API 키와 인덱스 이름 선언
pinecone_api_key = '423c8b35-3b1a-46fb-aebb-e20c9a36dba1'  # 여기에 Pinecone API 키를 입력
index_name = 'test2'

# Upstage API 키 선언
upstage_api_key = 'up_yKqThHL17ZcjIGzeOxYkYCaTVqyLb'  # 여기에 Upstage API 키를 입력

# Pinecone API 설정 및 초기화
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)


# URL에서 제목, 날짜, 내용(본문 텍스트나 이미지 URL) 추출하는 함수
def extract_text_and_date_from_url(urls):
    all_data = []

    def fetch_text_and_date(url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # 제목 추출
            title_element = soup.find('span', class_='bo_v_tit')
            title = title_element.get_text(strip=True) if title_element else "Unknown Title"

            # 본문 내용 또는 이미지 URL 추출
            paragraphs = soup.find('div', id='bo_v_con')
            content = []  # 텍스트와 이미지 URL을 모두 담는 리스트

            if paragraphs:
                for element in paragraphs.find_all(['p', 'img']):
                    if element.name == 'p':  # 본문 텍스트 요소
                        content.append(element.get_text(strip=True))
                    elif element.name == 'img':  # 이미지 요소
                        img_src = element.get('src')
                        if img_src:
                            content.append(img_src)

            # 날짜 추출
            date_element = soup.select_one("strong.if_date")  # 수정된 선택자
            date = date_element.get_text(strip=True) if date_element else "Unknown Date"

            # 제목이 Unknown Title이 아닐 때만 데이터 추가
            if title != "Unknown Title":
                return title, content, date, url  # 문서 제목, 내용(리스트 형태), 날짜, URL 반환
            else:
                return None, None, None, None  # 제목이 Unknown일 경우 None 반환
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return None, None, None, url

    with ThreadPoolExecutor() as executor:
        results = executor.map(fetch_text_and_date, urls)

    # 유효한 데이터만 추가
    all_data = [(title, content, date, url) for title, content, date, url in results if title is not None]
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

# 스크래핑할 URL 목록 생성
now_number = get_latest_wr_id()
urls = []
for number in range(now_number, now_number-120, -1):     #27726
    urls.append("https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_1&wr_id=" + str(number))

# URL에서 문서와 날짜 추출
document_data = extract_text_and_date_from_url(urls)


# 텍스트 분리기 초기화
class CharacterTextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i:i + self.chunk_size])
        return chunks

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


import base64

# 수집된 데이터 출력 예시
def display_images(all_data):
    html_content = ""
    for title, content, date, url in all_data:
        html_content += f"<h2>{title}</h2>"
        html_content += f"<p>Date: {date}</p>"
        html_content += f"<p>URL: {url}</p>"
        
        for item in content:
            if item.startswith("data:image"):  # Base64 이미지 데이터
                # 이미지 태그로 삽입
                html_content += f'<img src="{item}" alt="Embedded Image" style="max-width: 500px; max-height: 500px;"><br>'
            else:  # 일반 텍스트 내용
                html_content += f"<p>{item}</p>"
        html_content += "<hr>"

    # HTML로 표시
    from IPython.display import display, HTML
    display(HTML(html_content))

# Example usage:
# display_images(all_data)  # all_data를 매개변수로 전달하여 이미지와 텍스트 표시


print(display_images(document_data))