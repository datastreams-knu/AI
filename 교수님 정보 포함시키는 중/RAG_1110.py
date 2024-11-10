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
# Pinecone API 키와 인덱스 이름 선언
pinecone_api_key = 'cd22a6ee-0b74-4e9d-af1b-a1e83917d39e'  # 여기에 Pinecone API 키를 입력
index_name = 'prof'

# Upstage API 키 선언
upstage_api_key = 'up_pGRnryI1JnrxChGycZmswEZm934Tf'  # 여기에 Upstage API 키를 입력

# Pinecone API 설정 및 초기화
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)
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
            print(date)

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



def extract_professor_info_from_urls(urls):
    all_data = []

    def fetch_professor_info(url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            # 교수 정보가 담긴 요소들 선택
            professor_elements = soup.find_all("div", id="dr")

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
                date = "작성일24-11-07 14:11"

                # 교수 정보를 반환
                return title, text_content, image_content, date, url

        except Exception as e:
            print(f"Error processing {url}: {e}")
            return None  # 오류 발생 시 None 반환

    # ThreadPoolExecutor를 이용하여 병렬 크롤링
    with ThreadPoolExecutor() as executor:
        results = executor.map(fetch_professor_info, urls)

    # 유효한 데이터만 추가
    all_data = [(title, text_content, image_content, date, url) 
                for result in results if result is not None
                for title, text_content, image_content, date, url in [result]]

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
for number in range(now_number, now_number-1, -1):     #2024-08-07 수강신청 안내시작..28148
    urls.append("https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_1&wr_id=" + str(number))

# 교수진 페이지 URL 목록
urls2 = [
    "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub2_1&lang=kor",
    "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub2_2&lang=kor",
    "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub2_3&lang=kor",
    "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub2_6&lang=kor",
    "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub2_4&lang=kor",
    "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub2_5&lang=kor",
]

# URL에서 문서와 날짜 추출
document_data = extract_text_and_date_from_url(urls)
prof_data = extract_professor_info_from_urls(urls2)

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

# 교수 정보 크롤링 데이터 분리 및 저장
professor_texts = []
professor_image_urls = []
professor_titles = []
professor_doc_urls = []
professor_doc_dates = []

for title, doc, image, date, url in prof_data:
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

# 교수 정보 데이터를 기존 데이터와 합치기
texts.extend(professor_texts)
image_url.extend(professor_image_urls)
titles.extend(professor_titles)
doc_urls.extend(professor_doc_urls)
doc_dates.extend(professor_doc_dates)

########################################################################################################

def transformed_query(content):
    # 중복된 단어를 제거한 명사를 담을 리스트
    query_nouns = []

    # 1. 숫자와 특정 단어가 결합된 패턴 추출 (예: '2024학년도', '1월' 등)
    pattern = r'\d+(?:학년도|년|월|일|학기|시|분|초)?'
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
    if '동계' in content:
        query_nouns.append('겨울')
    if '겨울' in content:
        query_nouns.append('동계')
    if '여름' in content:
        query_nouns.append('하계')
    if '성인지' in content:
        query_nouns.append('성인지')
    if '첨성인' in content:
        query_nouns.append('첨성인')
    if '글솦' in content:
        query_nouns.append('글솝')
    # 5. Okt 형태소 분석기를 이용한 추가 명사 추출
    okt = Okt()
    additional_nouns = [noun for noun in okt.nouns(content) if len(noun) >= 1]
    query_nouns += additional_nouns
    # "공지", "사항", "공지사항"을 query_nouns에서 제거
    remove_noticement = ['공지', '사항', '공지사항']
    query_nouns = [noun for noun in query_nouns if noun not in remove_noticement]
    # 6. "수강" 단어와 관련된 키워드 결합 추가
    if '수강' in content:
        related_keywords = ['변경', '신청', '정정', '취소','꾸러미']
        for keyword in related_keywords:
            if keyword in content:
                # '수강'과 결합하여 새로운 키워드 추가
                combined_keyword = '수강' + keyword
                query_nouns.append(combined_keyword)
    # 최종 명사 리스트에서 중복된 단어 제거
    query_nouns = list(set(query_nouns))
    return query_nouns

# BM25 유사도 계산
tokenized_titles = [transformed_query(title) for title in titles]# 제목마다 명사만 추출하여 토큰화

# 기존과 동일한 파라미터를 사용하고 있는지 확인
bm25_titles = BM25Okapi(tokenized_titles, k1=1.5, b=0.75)  # 기존 파라미터 확인





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





def best_docs(user_question):
      # 사용자 질문
      okt = Okt()
      query_noun=transformed_query(user_question)
     # print(f"=================\n\n question: {user_question} 추출된 명사: {query_noun}")

      title_question_similarities = bm25_titles.get_scores(query_noun)  # 제목과 사용자 질문 간의 유사도
      title_question_similarities/=25

      # 사용자 질문에서 추출한 명사와 각 문서 제목에 대한 유사도를 조정하는 함수
      def adjust_similarity_scores(query_noun, titles, similarities):
          # 각 제목에 대해 명사가 포함되어 있는지 확인 후 유사도 조정
          for idx, title in enumerate(titles):
              # 제목에 포함된 query_noun 요소의 개수를 센다
              matching_nouns = [noun for noun in query_noun if noun in title]

              # 하나 이상의 명사가 포함된 경우 유사도 0.1씩 가산
              if matching_nouns:
                  similarities[idx] += 0.21 * len(matching_nouns)
                  for noun in matching_nouns:
                    if re.search(r'\d', noun):  # 숫자가 포함된 단어 확인
                        similarities[idx] += 0.21  # 숫자가 포함된 단어에 대해서는 유사도를 0.5 증가
              # query_noun에 "대학원"이 없고 제목에 "대학원"이 포함된 경우 유사도를 0.1 감소
              if "대학원" not in query_noun and "대학원" in title:
                  similarities[idx] -= 1
              if "파일럿" not in query_noun and "파일럿" in title:
                  similarities[idx]-=1
              if ("현장" and "실습" and "현장실습" )not in query_noun and "대체" in query_noun:
                  similarities[idx]-=1
              # 본문 내용이 "No content"인 경우 유사도 0.5 추가 (조정값은 필요에 따라 변경 가능)
              if texts[idx] == "No content":
                  similarities[idx] *=2.5  # 본문이 "No content"인 경우 유사도를 높임
          return similarities


      top_15_titles_idx = np.argsort(title_question_similarities)[-20:][::-1]
      # tooop=top_15_titles_idx[:3]
      # print("처음 정렬된 BM25 문서:")
      # for idx in tooop:  # top_20_titles_idx에서 각 인덱스를 가져옴
      #     print(f"  제목: {titles[idx]}")
      #     print(f"  유사도: {title_question_similarities[idx]}")
      #     print(f" URL: {doc_urls[idx]}")
      #     print("-" * 50)
      # 조정된 유사도 계산
      adjusted_similarities = adjust_similarity_scores(query_noun, titles, title_question_similarities)
      # 유사도 기준 상위 15개 문서 선택
      top_20_titles_idx = np.argsort(title_question_similarities)[-20:][::-1]

       # 결과 출력
      # print("최종 정렬된 BM25 문서:")
      # for idx in top_20_titles_idx:  # top_20_titles_idx에서 각 인덱스를 가져옴
      #     print(f"  제목: {titles[idx]}")
      #     print(f"  유사도: {title_question_similarities[idx]}")
      #     print(f" URL: {doc_urls[idx]}")
      #     print("-" * 50)

      Bm25_best_docs = [(titles[i], doc_dates[i], texts[i], doc_urls[i],image_url[i]) for i in top_20_titles_idx]

      ####################################################################################################

      # 1. Dense Retrieval - Text 임베딩 기반 20개 문서 추출
      query_dense_vector = np.array(embeddings.embed_query(user_question))  # 사용자 질문 임베딩

      # Pinecone에서 텍스트에 대한 가장 유사한 벡터 20개 추출
      pinecone_results_text = index.query(vector=query_dense_vector.tolist(), top_k=20, include_values=True, include_metadata=True)
      pinecone_similarities_text = [res['score'] for res in pinecone_results_text['matches']]
      pinecone_docs_text = [(res['metadata'].get('title', 'No Title'),
                            res['metadata'].get('date', 'No Date'),
                            res['metadata'].get('text', ''),
                            res['metadata'].get('url', 'No URL')) for res in pinecone_results_text['matches']]

      # 2. Dense Retrieval - Title 임베딩 기반 20개 문서 추출
      dense_noun=transformed_query(user_question)
      query_title_dense_vector = np.array(embeddings.embed_query(dense_noun))  # 사용자 질문에 대한 제목 임베딩

      #################################################3#################################################3
      #####################################################################################################3

      #1. 본문 임베딩 기반 문서 추출 결과 출력
      # print("본문 기반 유사도:")
      # for idx, doc in enumerate(pinecone_docs_text):
      #     title, date, text, url = doc
      #     score = pinecone_similarities_text[idx]
      #     print(f"제목: {title}, 유사도: {score}, {len(text)} 날짜: {date}, URL: {url}")


      ################################################3#################################################3
      ####################################################################################################3


      #####파인콘으로 구한  문서 추출 방식 결합하기.
      combine_dense_docs = []

      # 1. 본문 기반 문서를 combine_dense_docs에 먼저 추가
      for idx, text_doc in enumerate(pinecone_docs_text):
          text_similarity = pinecone_similarities_text[idx]*3
          combine_dense_docs.append((text_similarity, text_doc))  # (유사도, (제목, 날짜, 본문, URL))

      ####query_noun에 포함된 키워드로 유사도를 보정
      # 유사도 기준으로 내림차순 정렬
      combine_dense_docs.sort(key=lambda x: x[0], reverse=True)

      # 결과 출력
      # print("\n통합된 파인콘문서 유사도:")
      # for score, doc in combine_dense_docs:
      #     title, date, text, url = doc
      #     print(f"제목: {title}\n유사도: {score} {url}")
      #     print('---------------------------------')


      #################################################3#################################################3
      #####################################################################################################3

      # Step 1: combine_dense_docs에 제목, 본문, 날짜, URL을 미리 저장

      # combine_dense_doc는 (유사도, 제목, 본문 내용, 날짜, URL) 형식으로 데이터를 저장합니다.
      combine_dense_doc = []

      # combine_dense_docs의 내부 구조에 맞게 두 단계로 분해
      for score, (title, date, text, url) in combine_dense_docs:
          combine_dense_doc.append((score, title, text, date, url))

      # Step 2: combine_dense_docs와 BM25 결과 합치기
      final_best_docs = []

      # combine_dense_docs와 BM25 결과를 합쳐서 처리
      for score, title, text, date, url in combine_dense_doc:
          matched = False
          for bm25_doc in Bm25_best_docs:
              if bm25_doc[0] == title:  # 제목이 일치하면 유사도를 합산
                  combined_similarity = score + adjusted_similarities[titles.index(bm25_doc[0])]
                  final_best_docs.append((combined_similarity, bm25_doc[0], bm25_doc[1], bm25_doc[2], bm25_doc[3], bm25_doc[4]))
                  matched = True
                  break
          if not matched:

              # 제목이 일치하지 않으면 combine_dense_docs에서만 유사도 사용
              final_best_docs.append((score,title, date, text, url, "No content"))


      # 제목이 일치하지 않는 BM25 문서도 추가
      for bm25_doc in Bm25_best_docs:
          matched = False
          for score, title, text, date, url in combine_dense_doc:
              if bm25_doc[0] == title and bm25_doc[1]==text:  # 제목이 일치하면 matched = True로 처리됨
                  matched = True
                  break
          if not matched:
              # 제목이 일치하지 않으면 BM25 문서만 final_best_docs에 추가
              combined_similarity = adjusted_similarities[titles.index(bm25_doc[0])]  # BM25 유사도 가져오기
              final_best_docs.append((combined_similarity, bm25_doc[0], bm25_doc[1], bm25_doc[2], bm25_doc[3], bm25_doc[4]))
      final_best_docs.sort(key=lambda x: x[0], reverse=True)
      final_best_docs=final_best_docs[:20]



      # print("\n\n\n\n필터링 전 최종문서 (유사도 큰 순):")
      # for idx, (scor, titl, dat, tex, ur, image_ur) in enumerate(final_best_docs):
      #     print(f"순위 {idx+1}: 제목: {titl}, 유사도: {scor},본문 {len(tex)} 날짜: {dat}, URL: {ur}")
      #     print("-" * 50)



      def cluster_documents_by_similarity(docs, threshold=0.7):
          clusters = []

          for doc in docs:
              title = doc[1]
              added_to_cluster = False

              # 기존 클러스터와 비교
              for cluster in clusters:
                  # 첫 번째 문서의 제목과 현재 문서의 제목을 비교해 유사도를 계산
                  cluster_title = cluster[0][1]
                  similarity = SequenceMatcher(None, cluster_title, title).ratio()

                  # 유사도가 threshold 이상이면 해당 클러스터에 추가
                  if similarity >= threshold:
                      cluster.append(doc)
                      added_to_cluster = True
                      break

              # 유사한 클러스터가 없으면 새로운 클러스터 생성
              if not added_to_cluster:
                  clusters.append([doc])

          return clusters

      def last_filter_keyword(DOCS,query_noun):
          # 필터링에 사용할 키워드 리스트
          Final_best=DOCS
          # 키워드가 포함된 경우 유사도를 조정하고, 유사도 기준으로 내림차순 정렬
          for idx, doc in enumerate(DOCS):
              score, title, date, text, url, image = doc
              # query_nouns에 없는 키워드가 본문에 포함되었는지 확인
              if any(keyword in text for keyword in ['계약학과', '대학원', '타대학원']) and not any(keyword in query_noun for keyword in ['계약학과', '대학원', '타대학원']):
                  score -= 0.1  # 유사도 점수를 0.1 낮추기
              # 조정된 유사도 점수를 사용하여 다시 리스트에 저장
              Final_best[idx] = (score, title, date, text,  url, image)
              #print(Final_best[idx])
          return Final_best


      def filter_and_sort_clusters(query_nouns, clusters):
          # 숫자가 포함되어 있는지 확인하는 정규 표현식 패턴
          number_pattern = r"\d"
          # query_noun에 숫자가 포함되어 있는지 확인
          # 특정 키워드도 함께 확인합니다.
          keywords = ["최근", "최신", "현재", "지금"]
          #print(f": {query_nouns}")
          if (any(keyword in word for word in query_nouns for keyword in keywords) or not any(re.search(number_pattern, word) for word in query_nouns)):
              # 숫자가 없으면 날짜 기준으로 클러스터 정렬
                clusters = sorted(clusters, key=lambda cluster: max(doc[0] for doc in cluster), reverse=True)
                best_cluster = clusters[0]
                best_cluster_similarity=best_cluster[0][0]
                check_date_cluster = sorted(best_cluster, key=lambda doc: doc[2], reverse=True)  # 날짜 기준 내림차순 정렬
                check_date_similarity =check_date_cluster[0][0]
                if (best_cluster_similarity-check_date_similarity>1):
                  #print("if인데 유사도")
                  best_cluter=last_filter_keyword(best_cluster,query_nouns)
                  sorted_cluster = sorted(best_cluster, key=lambda doc: doc[0], reverse=True)
                else:
                  #print("else인데 유사도 또는 날짜")
                  if (not any(re.search(number_pattern, word) for word in query_nouns)):
                    #print("아무것도 없는데.. 최신정보로?")
                    sorted_cluster = sorted(best_cluster, key=lambda doc: doc[2], reverse=True)
                  else:
                    #print("있었네..유사도로")
                    best_cluter=last_filter_keyword(best_cluster,query_nouns)
                    sorted_cluster = sorted(best_cluster, key=lambda doc: doc[0], reverse=True)
          else:
            #print("유사도위주로 판단")
            clusters = sorted(clusters, key=lambda cluster: max(doc[0] for doc in cluster), reverse=True)
            best_cluster = clusters[0]
            best_cluter=last_filter_keyword(best_cluster,query_nouns)
            sorted_cluster = sorted(best_cluster, key=lambda doc: doc[0], reverse=True)  # 유사도 기준 내림차순 정렬
          return sorted_cluster
      # Step 1: Adjust similarity scores based on the presence of query_noun


      # Step 2: Cluster documents by similarity
      clusters = cluster_documents_by_similarity(final_best_docs)

      query_nouns=transformed_query(user_question)

      # Step 3: Filter and sort clusters based on query_noun
      final_best_docs= filter_and_sort_clusters(query_nouns, clusters)


      # print("\n\n\n\n최종 상위 문서 (유사도 및 날짜 기준 정렬):")
      # for idx, (scor, titl, dat, tex, ur, image_ur) in enumerate(final_best_docs):
      #     print(f"순위 {idx+1}: 제목: {titl}, 유사도: {scor}, 날짜: {dat}, URL: {ur}")
      #     print("-" * 50)


      top_title =  final_best_docs[0][1]  # 최상위 문서의 제목 가져오기
      top_score = final_best_docs[0][0]
      # 기준 제목을 제외한 제목들을 추출
      candidate_titles = [doc[1] for doc in final_best_docs[1:]]

      # 제목 간 유사도 계산
      vectorizer = TfidfVectorizer().fit_transform([top_title] + candidate_titles)
      similarity_matrix = cosine_similarity(vectorizer)
      title_similarities = similarity_matrix[0][1:]  # 첫 번째 제목과의 유사도 리스트
      # 일정 유사도(threshold) 이상인 문서의 유사도 값을 합산
      threshold = 0.6
      adjusted_score = top_score  # 초기 상위 문서의 유사도 값

      for idx, sim_score in enumerate(title_similarities):
          if sim_score >= threshold:
              adjusted_score += final_best_docs[idx + 1][0]+0.5
              break# 유사도가 높은 문서의 유사도 값 합산

      # 첫 번째 문서의 유사도를 조정된 값으로 업데이트
      final_best_docs[0] = (adjusted_score, *final_best_docs[0][1:])
      # Step 3: 최상위 문서와 동일한 제목을 가진 문서 필터링 및 개수 카운트
      result_docs = [doc for doc in final_best_docs if doc[1] == top_title]
      top_title_count = len(result_docs)
      # Step 4: 결과 출력
      # print("\n\n\n\n최종 상위 문서 (유사도 및 날짜 기준 정렬):")
      # for idx, (scor, titl, dat, tex, ur, image_ur) in enumerate(result_docs):
      #     print(f"순위 {idx+1}: 제목: {titl[:10]}, 유사도: {scor}, 날짜: {dat}, URL: {ur}")
      #     print("-" * 50)

      return result_docs[:top_title_count]



prompt_template = """당신은 경북대학교 컴퓨터학부 공지사항을 전달하는 직원이고, 사용자의 질문에 대해 올바른 공지사항의 내용을 참조하여 정확하게 전달해야 할 의무가 있습니다.
현재 한국 시간: {current_time}

주어진 컨텍스트를 기반으로 다음 질문에 답변해주세요:

{context}

질문: {question}

답변 시 다음 사항을 고려해주세요:

1. 질문의 내용이 이벤트의 기간에 대한 것일 경우, 문서에 주어진 기한과 현재 한국 시간을 비교하여 해당 이벤트가 예정된 것인지, 진행 중인지, 또는 이미 종료되었는지에 대한 정보를 알려주세요.
  예를 들어, "2학기 수강신청 일정은 언제야?"라는 질문을 받았을 경우, 현재 시간은 11월이라고 가정하면 수강신청은 기간은 8월이었으므로 이미 종료된 이벤트입니다.
  따라서, "2학기 수강신청은 이미 종료되었습니다."라는 문구를 추가로 사용자에게 제공해주고, 2학기 수강신청 일정에 대한 정보를 사용자에게 제공해주어야 합니다.
2. 질문에서 핵심적인 키워드들을 골라 키워드들과 관련된 문서를 찾아서 해당 문서를 읽고 정확한 내용을 답변해주세요.
3. 질문에 포함된 핵심적인 키워드와 관련된 내용의 문서가 여러 개가 있을 경우, 질문의 내용에 구체적인 기간에 대한 정보가 없다면 (ex. 2024년 1학기, 2차 등) 가장 최근의 문서에 대한 정보를 우선적으로 제공하세요.
  예를 들어, Tutor 모집글이 1~7차까지 존재한다고 가정하였을 때, 질문 내에 구체적으로 3차 모집에 대한 정보를 물었다면 Tutor 3차 모집에 대한 문서를 제공해야 합니다.
  그러나, 단순히 Tutor 모집 정보에 대해 알려달라고 질문을 받았을 경우, 구체적인 기간에 대한 정보가 없으므로 가장 최신 문서인 7차 모집에 대한 문서를 제공하면 됩니다.
  다른 예시로, 3~8월은 1학기이고, 9월~2월은 2학기입니다. 따라서, 사용자의 질문 시간이 2024년 11월 10일이라고 가정하고, 질문에 포함된 핵심 키워드가 2024년 1학기에도 존재하고 2학기에도 존재하는 경우,
  질문 내에 구체적으로 1학기에 대한 정보를 묻는 것이 아니라면 가장 최근 문서인 2학기에 대한 정보를 알려주세요.
  (ex. 현재 한국 시간 : 2024년 11월 10일이라고 가정하였을 때, 질문이 "수강 정정에 대한 일정을 알려줘."라면, 2024년 2학기 수강 정정에 대한 문서를 검색합니다.
  그러나, "1학기 수강 정정에 대한 일정을 알려줘."와 같이 구체적인 기간이 있다면, 2024년 1학기 수강 정정에 대한 문서를 검색해야 합니다.)
4. 문서의 내용을 그대로 길게 전달하기보다는 질문에서 요구하는 내용에 해당하는 답변만을 제공함으로써 최대한 답변을 간결하고 일관된 방식으로 제공하세요.
5. 질문의 키워드와 일치하거나 비슷한 맥락의 문서를 발견하지 못한 경우, 잘못된 정보를 제공하지 말고 모른다고 답변하세요.
6. 수강 정정과 수강정정, 수강 변경과 수강변경, 수강 신청과 수강신청 등과 같이 띄어쓰기가 존재하지만 같은 의미를 가진 단어들은 동일한 키워드로 인식해주세요.
7. 답변은 친절하게 존댓말로 제공하세요.

답변:"""


# PromptTemplate 객체 생성
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["current_time", "context", "question"]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_answer_from_chain(best_docs, user_question):
    documents = []
    if doc[3]!="No content":
      doc_titles= [doc[1] for doc in best_docs]
      doc_dates=[doc[2] for doc in best_docs]
      doc_texts = [doc[3] for doc in best_docs]
      doc_urls = [doc[4] for doc in best_docs]  # URL을 별도로 저장

    documents = [
        Document(page_content=text, metadata={"title": title, "url": url, "doc_date": datetime.strptime(date, '작성일%y-%m-%d %H:%M')})
        for title, text, url, date in zip(doc_titles, doc_texts, doc_urls, doc_dates)
    ]
    # 키워드 기반 관련성 필터링 추가 (질문과 관련 없는 문서 제거)
    # 사용자 질문을 전처리하여 공백 제거 후 명사만 추출
    okt = Okt()
    query_nouns=transformed_query(user_question)
    # print(query_nouns)
    relevant_docs = [doc for doc in documents if any(keyword in doc.page_content for keyword in query_nouns)]
    #print(relevant_docs)
    if not relevant_docs:
      return None, None, None
    vector_store = FAISS.from_documents(relevant_docs, embeddings)
    retriever = vector_store.as_retriever()
    llm = ChatUpstage(api_key=upstage_api_key)
    # PromptTemplate 인스턴스 사용
    qa_chain = (
        {
            "current_time": lambda _: get_korean_time().strftime("%Y년 %m월 %d일 %H시 %M분"),
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return qa_chain, retriever, relevant_docs  # retriever를 반환


##### 유사도 제목 날짜 본문  url image_url순으로 저장됨
def get_ai_message(question):
    top_doc = best_docs(question)  # 가장 유사한 문서 가져오기
    top_docs = [list(doc) for doc in top_doc]
    # print(f"\ntitles: {top_docs[0][1]} similarity: {top_docs[0][0]}, text:{(len(top_docs[0][3]))} doc_dates: {top_docs[0][2]} URL: {top_docs[0][4]}")
    ### top_docs에 이미지 URL이 들어있다면?
    if len(top_docs[0])==6 and top_docs[0][5]!="No content" and top_docs[0][3]=="No content" and top_docs[0][0]>1.8:
           # image_display 초기화 및 여러 이미지 처리
         #print("첫번째 조건 만족")
         image_display = ""
         for img_url in top_docs[0][5]:  # 여러 이미지 URL에 대해 반복
             image_display += f"<img src='{img_url}' alt='관련 이미지' style='max-width: 500px; max-height: 500px;' /><br>"
         doc_references = top_docs[0][4]
         # content 초기화
         content = []
        # top_docs의 내용 확인
         if top_docs[0][3] == "No content":
             content = []  # No content일 경우 비우기
         else:
             content = top_docs[0][3]  # content에 top_docs[0][3] 내용 저장
         if content:
            html_output = f"{image_display}<p>{content}</p><hr>\n"
         else:
            html_output = f"{image_display}<p>>\n"
        # HTML 출력 및 반환할 내용 생성
         display(HTML(image_display))
         return  f"<p>항상 정확한 답변을 제공하지 못할 수 있습니다.아래의 URL들을 참고하여 정확하고 자세한 정보를 확인하세요.\n{doc_references}"

    else:
        qa_chain, retriever, relevant_docs = get_answer_from_chain(top_docs, question)  # 답변 생성 체인 생성

        image_display = ""
        seen_img_urls = set()  # 이미 출력된 이미지 URL을 추적하는 set
        if (top_docs[0][5] != "No content"):
            #print("Ok")
            for img_url in top_docs[0][5]:  # 여러 이미지 URL에 대해 반복
                if img_url not in seen_img_urls:  # img_url이 이미 출력되지 않은 경우
                    image_display += f"<img src='{img_url}' alt='관련 이미지' style='max-width: 500px; max-height: 500px;' /><br>"
                    seen_img_urls.add(img_url)  # img_url을 set에 추가하여 중복을 방지
        doc_references = top_docs[0][4]

        if not qa_chain or not relevant_docs:
          if (top_docs[0][5]!="No content") and top_docs[0][0]>1.8:
            display(HTML(image_display))
            url=doc_references
            return f"\n\n해당 질문에 대한 내용은 이미지 파일로 확인해주세요.\n 자세한 사항은 공지사항을 살펴봐주세요.\n\n{url}"
          else:
            url="https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_1"
          return f"\n\n해당 질문은 공지사항에 없는 내용입니다.\n 자세한 사항은 공지사항을 살펴봐주세요.\n\n{url}"
        if (top_docs[0][0]<1.8):
          url="https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_1"
          return f"\n\n해당 질문은 공지사항에 없는 내용입니다.\n 자세한 사항은 공지사항을 살펴봐주세요.\n\n{url}"
        existing_answer = qa_chain.invoke(question)# 초기 답변 생성 및 문자열로 할당
        answer_result=existing_answer
        display(HTML(image_display))
        # 상위 3개의 참조한 문서의 URL 포함 형식으로 반환
        doc_references = "\n".join([
            f"\n참고 문서 URL: {doc.metadata['url']}"
            for doc in relevant_docs[:1] if doc.metadata.get('url') != 'No URL'
        ])
        # AI의 최종 답변과 참조 URL을 함께 반환
        return f"{answer_result}\n\n------------------------------------------------\n항상 정확한 답변을 제공하지 못할 수 있습니다.\n아래의 URL들을 참고하여 정확하고 자세한 정보를 확인하세요.\n{doc_references}"