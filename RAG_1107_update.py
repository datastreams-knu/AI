# %pip install -U langchain-community
# %pip install beautifulsoup4 requests scikit-learn pinecone-client numpy langchain-upstage faiss-cpu
# %pip install langchain
# !pip install spacy
# !python -m spacy download ko_core_news_sm
# # 필요한 시스템 패키지 설치
# !apt-get install -y python3-dev
# !apt-get install -y libmecab-dev
# !apt-get install -y mecab mecab-ko mecab-ko-dic

# # konlpy 설치
# !pip install konlpy
# !pip install python-Levenshtein

######################코랩환경에서 설치할 pip 정리#########################################

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
from konlpy.tag import Okt
from collections import defaultdict
import Levenshtein
import numpy as np
from IPython.display import display, HTML
# Pinecone API 키와 인덱스 이름 선언
pinecone_api_key = 'cd22a6ee-0b74-4e9d-af1b-a1e83917d39e'  # 여기에 Pinecone API 키를 입력
index_name = 'new-test'

# Upstage API 키 선언
upstage_api_key = 'up_pGRnryI1JnrxChGycZmswEZm934Tf'  # 여기에 Upstage API 키를 입력

# Pinecone API 설정 및 초기화
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)




# URL에서 제목, 날짜, 내용(본문 텍스트와 이미지 URL) 추출하는 함수
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
for number in range(now_number, now_number - 10, -1):     #27726
    urls.append("https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_1&wr_id=" + str(number))

# URL에서 문서와 날짜 추출
document_data = extract_text_and_date_from_url(urls)


def get_korean_time():
    return datetime.now(pytz.timezone('Asia/Seoul'))





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





# 1. Sparse Retrieval (TF-IDF)

#texts에 대한 TF-IDF 벡터화
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(texts)

# 제목에 대한 TF-IDF 벡터화
title_vectorizer = TfidfVectorizer()
title_vectors = title_vectorizer.fit_transform(titles)



# 2. Dense Retrieval (Upstage 임베딩)
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





####### 쿼리 변환 작업 및 전처리#############
okt = Okt()
# 사용자 질문을 전처리하여 공백 제거 후 명사만 추출
def clean_noun(nouns):
    cleaned_nouns = []
    for noun in nouns:
        # 접미사 및 조사를 정규 표현식으로 제거
        # 예: 정정은 → 정정
        cleaned_noun = re.sub(r'은$|는$|이$|가$|을$|를$|의$|에$|에서$|와$|과$', '', noun)

        # 앞뒤의 접두사도 제거하고 남는 부분이 있을 경우만 추가
        if cleaned_noun:
            cleaned_nouns.append(cleaned_noun)
    return cleaned_nouns

# 조사, 접미사 등을 제거하는 함수
def clean_sentence(sentence):
    # 조사 및 접미사를 반복적으로 제거
    sentence = re.sub(r'([가-힣]+?)(은|는|이|가|을|를|에|에서|와|과)', r'\1', sentence)  # 명사 뒤 조사 제거
    # 필요한 경우, 특정 조사나 접미사를 추가로 제거
    sentence = re.sub(r'에$|를$|을$|의$', '', sentence)  # 문장 끝에 남아있는 조사 제거

    return sentence



def get_best_docs(users_question):
    try:

        # 불필요한 부분 제거
        user_question = clean_sentence(users_question)
        n=clean_noun(okt.nouns(users_question))
        nouns=' '.join(n)
        #print(f"변환된 질문:{user_question}, 키워드 추출 :{nouns}")
  ##############################################################
  #####    TEXT 기준으로 TF-IDF 유사도 체크 (NOUN / USER_QUESTION)
  ##############################################################

        # Sparse Retrieval: TF-IDF 벡터화  TEXT 기준임.
        query_tfidf_vector = vectorizer.transform([nouns])
        # TF-IDF 코사인 유사도 계산
        text_noun_cosine_similarities = cosine_similarity(query_tfidf_vector, doc_vectors).flatten()



        # Sparse Retrieval: TF-IDF 벡터화  TEXT 기준임.
        querys_tfidf_vector = vectorizer.transform([user_question])
        # TF-IDF 코사인 유사도 계산
        text_question_cosine_similarities = cosine_similarity(querys_tfidf_vector, doc_vectors).flatten()


  ##############################################################
  #####        Title 기준으로 TF-IDF 유사도 체크 (NOUN / USER_QUESTION)
  ##############################################################

        #### nouns 기준으로 실행
        # 제목에 대한 TF-IDF 벡터화
        query_title_vector = title_vectorizer.transform([nouns])
        # 제목과 사용자 질문 간의 유사도 계산
        title_noun_cosine_similarities = cosine_similarity(query_title_vector, title_vectors).flatten()



        #####   user_question 기준으로 실행
        # 제목에 대한 TF-IDF 벡터화
        query_titles_vector = title_vectorizer.transform([user_question])
        # 제목과 사용자 질문 간의 유사도 계산
        title_question_cosine_similarities = cosine_similarity(query_titles_vector, title_vectors).flatten()

  ##############################################################
  ##############################################################
# Dense Retrieval: Upstage 임베딩을 통한 유사도 계산
        query_dense_vector = np.array(embeddings.embed_query(user_question))


        # Pinecone에서 가장 유사한 벡터 찾기
        pinecone_results = index.query(vector=query_dense_vector.tolist(), top_k=10, include_values=True, include_metadata=True)
        pinecone_similarities = [res['score'] for res in pinecone_results['matches']]
        pinecone_docs = [(res['metadata'].get('title', 'No Title'), res['metadata'].get('date', 'No Date'),
                          res['metadata'].get('text', ''), res['metadata'].get('url', 'No URL')) for res in pinecone_results['matches']]

        # TF-IDF에서 상위 10개 문서의 유사도, texts기준 question
        top_text_question_tfidf_indices = np.argsort(text_question_cosine_similarities)[-10:][::-1]
        text_question_best_docs = [(titles[i], doc_dates[i], texts[i], doc_urls[i],image_url[i]) for i in top_text_question_tfidf_indices ]

        # TF-IDF에서 상위 10개 문서의 유사도, texts기준 noun
        top_text_noun_tfidf_indices = np.argsort(text_noun_cosine_similarities)[-10:][::-1]
        text_noun_best_docs = [(titles[i], doc_dates[i], texts[i], doc_urls[i],image_url[i]) for i in top_text_noun_tfidf_indices]

        # TF-IDF에서 상위 10개 문서의 유사도, title기준 question
        top_title_question_indices = np.argsort(title_question_cosine_similarities)[-10:][::-1]
        title_question_best_docs = [(titles[i], doc_dates[i], texts[i], doc_urls[i],image_url[i]) for i in top_title_question_indices ]

        # TF-IDF에서 상위 10개 문서의 유사도, title기준 noun
        top_title_noun_indices = np.argsort(title_noun_cosine_similarities)[-10:][::-1]
        title_noun_best_docs = [(titles[i], doc_dates[i], texts[i], doc_urls[i],image_url[i]) for i in top_title_noun_indices ]


# 필터링된 문서를 저장할 리스트와 초기화된 유사도 리스트
        filtered_docs = []
        filtered_similarities = []

        # 4가지 TF-IDF 방식에서 상위 10개 문서의 유사도와 문서 정보 결합
        all_tfidf_docs = text_question_best_docs + text_noun_best_docs + title_question_best_docs + title_noun_best_docs
        all_tfidf_similarities = text_question_cosine_similarities[top_text_question_tfidf_indices].tolist() + text_noun_cosine_similarities[top_text_noun_tfidf_indices].tolist() + title_question_cosine_similarities[top_title_question_indices].tolist() + title_noun_cosine_similarities[top_title_noun_indices].tolist()


        # 중복 제목 필터링 및 유사도 조정
        for i, doc in enumerate(all_tfidf_docs):
            # 제목에서 공백과 특수 문자 제거 후 소문자로 변환하여 비교
            title = re.sub(r'\s+|\W+', '', doc[0]).lower()
            current_text = re.sub(r'\s+|\W+', '', doc[2]).lower()
            similarity = all_tfidf_similarities[i]

            # 키워드 체크 ('대학원' 키워드 확인)
            keywords_normalized = [re.sub(r'\s+|\W+', '', keyword).lower() for keyword in n]
            if '대학원' not in keywords_normalized and '대학원' in title:
                similarity *= 0.1  # 유사도를 90% 감소

            # 문서가 이미 존재하는지 확인
            is_duplicate_content = False
            for j, existing_doc in enumerate(filtered_docs):
                existing_title = re.sub(r'\s+|\W+', '', existing_doc[0]).lower()
                existing_text = re.sub(r'\s+|\W+', '', existing_doc[2]).lower()

                if existing_title == title:
                    # 제목이 동일한 경우
                    s1 = filtered_similarities[j]
                    s2 = similarity
                    filtered_similarities[j] += s2  # 기존 유사도를 더함
                    similarity += s1
                    if existing_text == current_text:  # 내용도 동일한 경우
                        is_duplicate_content = True

            # 내용이 중복되지 않은 경우에만 문서 추가
            if not is_duplicate_content:
                filtered_docs.append(doc)
                filtered_similarities.append(similarity)

        # 필터링된 문서 상위 10개 출력
        top_filtered_indices = np.argsort(filtered_similarities)[-10:][::-1]

        # 상위 10개 문서에 대해 doc[4]가 "No content"가 아닌 경우 유사도를 2배로 조정 ( 본문 내용이 존재하지 않아서 생긴 유사도 낮게 측정된 문제를 해결함)
        for i in top_filtered_indices:
            if filtered_docs[i][4] != "No content":
                filtered_similarities[i] *= 2


        # 결합된 유사도 배열 생성
        combined_similarities = (np.array(filtered_similarities)).tolist() + (np.array(pinecone_similarities) * 3).tolist()
        # 최종적으로 가장 높은 유사도를 가진 문서들을 정렬하여 상위 10개 문서 선정
        final_top_indices = np.argsort(combined_similarities)[-10:][::-1]



        # 최종 문서 목록 생성 및 출력
        #print("\nCombined Top 10 Final Documents:")
        final_best_docs = []
        for i in final_top_indices:
            if i < len(filtered_similarities):
                # TF-IDF 기반 문서
                idx = i
                doc = filtered_docs[idx]
                similarity = filtered_similarities[idx]
                #print(f"Title: {doc[0]}, Date: {doc[1]}, URL: {doc[3]}, Similarity: {similarity}")
                final_best_docs.append(doc)
            else:
                # Pinecone 기반 문서
                pinecone_idx = i - len(filtered_similarities)
                doc = pinecone_docs[pinecone_idx]
                similarity = pinecone_similarities[pinecone_idx]
                #print(f"Title: {doc[0]}, Date: {doc[1]}, URL: {doc[3]}, Similarity: {similarity}")
                final_best_docs.append(doc)

        # for doc in final_best_docs :
        #   print(doc)
        # print("---------------------------------")

        #### 상위 10개 문서를 제목 키워드 일치로 필터링
        def rank_documents_based_on_keywords(best_docs, keywords):
            # 각 문서의 제목과 키워드의 일치를 기반으로 랭킹 조정
            keyword_scores = []

            for doc in best_docs:
                title = doc[0]  # 문서의 제목
                score = 0

                for keyword in keywords:
                    count = title.count(keyword)  # 키워드 등장 횟수
                    # 키워드의 등장 횟수에 따라 가중치를 부여
                    score += count * (count + 1)  # 예: 1회 등장 시 1점, 2회 등장 시 3점, 3회 등장 시 6점 등으로 가중치 증가

                keyword_scores.append(score)

            # 점수를 기준으로 문서 우선순위 조정
            ranked_docs = sorted(zip(best_docs, keyword_scores), key=lambda x: x[1], reverse=True)

            # 정렬된 문서 목록 반환
            return [doc for doc, score in ranked_docs]
        key=clean_noun(okt.nouns(user_question))
        final_best_docs = rank_documents_based_on_keywords(final_best_docs, key)

        # for doc in final_best_docs :
        #   print(doc)
        # print("---------------------------------")


        #### 첫 번째 문서가 질문의 의도에 제일 적합한 문서라고 판단하고 첫 번째 문서 제목을 기준으로 9개 문서 레반슈타인 거리로 유사도 체크해서 0.9 넘는 것만 최종 문서들로 채택함.
        #### 그리고 date순으로 정렬해 (3차 4차 5차 or 10월 9월 8월 )과 같이 가장 최신정보를 알 수 있게 함.
        def levenshtein_similarity(title1, title2):
            # 두 제목의 레벤슈타인 거리 계산
            distance = Levenshtein.distance(title1, title2)
            # 유사도 계산 (편집 거리 / 최대 길이로 나누어 정규화)
            return 1 - (distance / max(len(title1), len(title2)))

        def rank_documents_by_recent_date(final_best_docs, threshold=0.9):
              # 첫 번째 문서를 기준으로 유사한 제목을 가진 문서만 필터링
              filtered_docs = [final_best_docs[0]]  # 첫 번째 문서를 기준으로 시작
              base_title = final_best_docs[0][0]  # 기준 문서 제목

              for doc in final_best_docs[1:]:
                  title = doc[0]
                  similarity = levenshtein_similarity(base_title, title)
                  # 유사도가 임계값 이상인 경우만 필터링 목록에 추가
                  if similarity > threshold:
                      filtered_docs.append(doc)

              # 필터링된 문서 목록을 날짜 기준으로 내림차순 정렬
              filtered_docs.sort(key=lambda x: x[1], reverse=True)
              return filtered_docs

        # 기존의 final_best_docs에서 문서 선택 후 레벤슈타인 거리 기반으로 정렬
        final_best_doc = rank_documents_by_recent_date(final_best_docs)

        # print(final_best_doc)

        return final_best_doc
    except Exception as e:
      print(f"An error occured: {e}")





prompt_template = """당신은 경북대학교 컴퓨터학부 공지사항을 전달하는 직원입니다. 사용자의 질문에 대해 올바른 공지사항 내용을 참조하여 정확하고 간결하게 전달해야 할 의무가 있습니다.
현재 한국 시간: {current_time}

주어진 컨텍스트를 기반으로 다음 질문에 답변해주세요:

{context}

질문: {question}

답변 시 다음 사항을 고려해주세요:

1. 질문의 내용이 특정 기간을 요구하거나 현재 진행 여부를 묻는 경우, 문서에 기재된 기한과 현재 시간을 비교하여 이벤트가 예정된 것인지, 진행 중인지, 혹은 종료되었는지를 명확히 알려주세요.
   (예: "해커톤 일정이 언제인가요?"라는 질문을 받았다면, 관련된 해커톤 문서를 찾고, 현재 시간을 기준으로 해당 이벤트가 진행 중인지 명시해 주세요.)

2. 질문에서 핵심 키워드를 골라 관련된 문서를 우선적으로 읽고 해당 정보를 바탕으로 답변을 작성하세요.
   - 질문에 특정 기간이 지정되지 않은 경우, 가장 최근 문서를 우선적으로 참조하여 답변합니다.
   - 예를 들어, 특정 학기(1학기, 2학기)에 대한 정보가 필요할 때, 현재 시간을 기준으로 가장 근접한 학기의 문서를 선택합니다.

3. 비슷한 의미를 가진 단어들을 동일한 키워드로 처리하세요. 예: "수강 정정"과 "수강정정", "수강 변경"과 "수강변경" 등.

4. 문서의 내용을 그대로 전달하지 말고, 질문에 해당하는 부분만 간결하고 일관된 방식으로 답변해주세요.

5. 관련된 문서를 찾지 못한 경우, 정확한 정보가 없다는 점을 분명히 알려주세요.

6. 답변을 작성하기 전에, 사용자의 질문 시간과 현재 한국 시간을 한 줄로 명시하고 답변을 시작하세요.
   예시: "현재 한국 시간 : ____년 __월 __일\n성인지 교육은 12월 31일까지입니다."

7. 첫 번째 문서의 정보를 기반으로 답변을 작성하되, 이후 발견된 문서들을 검토하여 추가적인 정보가 필요한 경우, 내용을 개선하거나 추가하세요.

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
    if doc[2]!="No content":
      doc_titles= [doc[0] for doc in best_docs]
      doc_texts = [doc[2] for doc in best_docs]
      doc_urls = [doc[3] for doc in best_docs]  # URL을 별도로 저장
      doc_dates=[doc[1] for doc in best_docs]




    documents = [
        Document(page_content=text, metadata={"title": title, "url": url, "doc_date": datetime.strptime(date, '작성일%y-%m-%d %H:%M')})
        for title, text, url, date in zip(doc_titles, doc_texts, doc_urls, doc_dates)
    ]
    # 키워드 기반 관련성 필터링 추가 (질문과 관련 없는 문서 제거)
    # 사용자 질문을 전처리하여 공백 제거 후 명사만 추출
    okt = Okt()

    # 모든 공백 제거
    cleaned_question = user_question.replace(" ", "")
    # 접미사, 접두사, 조사 제거를 위한 함수
    def clean_noun(nouns):
        cleaned_nouns = []
        for noun in nouns:
            # 접미사 및 조사를 정규 표현식으로 제거
            # 예: 정정은 → 정정
            cleaned_noun = re.sub(r'은$|는$|이$|가$|을$|를$|의$|에$|에서$|와$|과$', '', noun)

            # 앞뒤의 접두사도 제거하고 남는 부분이 있을 경우만 추가
            if cleaned_noun:
                cleaned_nouns.append(cleaned_noun)
        return cleaned_nouns

    # 불필요한 부분 제거
    cleaned_nouns = clean_noun(okt.nouns(cleaned_question))

    relevant_docs = [doc for doc in documents if any(keyword in doc.page_content for keyword in cleaned_nouns)]
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



def get_ai_message(question):
    top_docs = get_best_docs(question)  # 가장 유사한 문서 가져오기
    # for docs in top_docs:
    #     if len(docs) == 4:
    #         print(f"pinecone\ntitles: {docs[0]}\n text:{len(docs[2])} \ndoc_dates: {docs[1]} \nURL: {docs[3]}")
    #     else:
    #         print(f"tf-idf\ntitles:   {docs[0]}\n text:{len(docs[2])} \ndoc_dates: {docs[1]}\n URL: {docs[3]}")

    ### top_docs에 이미지 URL이 들어있다면?
    if len(top_docs[0])==5 and top_docs[0][4]!="No content":
         image_display = f"<img src='{top_docs[0][4][0]}' alt='관련 이미지' style='max-width: 500px; max-height: 500px;' /><br>"
         doc_references=top_docs[0][3]
         # content 초기화
         content = []
        # top_docs의 내용 확인
         if top_docs[0][2] == "No content":
             content = []  # No content일 경우 비우기
         else:
             content = top_docs[0][2]  # content에 top_docs[0][2] 내용 저장
         if content:
            html_output = f"{image_display}<p>{content}</p><hr>\n"
         else:
            html_output = f"{image_display}<p>>\n"
        # HTML 출력 및 반환할 내용 생성
         display(HTML(image_display))
         return  f"<p>항상 정확한 답변을 제공하지 못할 수 있습니다.아래의 URL들을 참고하여 정확하고 자세한 정보를 확인하세요.\n{doc_references}"

    else:
        qa_chain, retriever, relevant_docs = get_answer_from_chain(top_docs, question)  # 답변 생성 체인 생성
        #print(relevant_docs)
        for top in top_docs:
            if (relevant_docs and relevant_docs[0].metadata['title'] == top[0] and len(top) == 5 and top[4] != "No content"):
                image_display = f"<img src='{top[4][0]}' alt='관련 이미지' style='max-width: 500px; max-height: 500px;' /><br>"
                display(HTML(image_display))
                break

        existing_answer = qa_chain.invoke(question)# 초기 답변 생성 및 문자열로 할당

        # Refine 체인에서 질문과 관련성 높은 문서만 유지
        refined_chain = RetrievalQA.from_chain_type(
            llm=ChatUpstage(api_key=upstage_api_key),
            chain_type="refine",
            retriever=retriever,
            return_source_documents=True,
            verbose=True,
        )

        # 리파인된 최종 답변 생성
        final_answer = refined_chain.invoke({
            "query": question,
            "existing_answer": existing_answer,  # 초기 답변을 전달하여 리파인
            "context": format_docs(relevant_docs)  # 문맥 정보 전달
        })

        # 최종 답변 결과 추출
        answer_result = final_answer.get('result')

        # 상위 3개의 참조한 문서의 URL 포함 형식으로 반환
        doc_references = "\n".join([
            f"\n참고 문서 URL: {doc.metadata['url']}"
            for doc in relevant_docs[:3] if doc.metadata.get('url') != 'No URL'
        ])
        # AI의 최종 답변과 참조 URL을 함께 반환
        return f"{answer_result}\n\n------------------------------------------------\n항상 정확한 답변을 제공하지 못할 수 있습니다.\n아래의 URL들을 참고하여 정확하고 자세한 정보를 확인하세요.\n{doc_references}"




