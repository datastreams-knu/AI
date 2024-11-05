from collections import defaultdict
import Levenshtein
import numpy as np
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

# Pinecone API 키와 인덱스 이름 선언
pinecone_api_key = '423c8b35-3b1a-46fb-aebb-e20c9a36dba1'  # 여기에 Pinecone API 키를 입력
index_name = 'test'

# Upstage API 키 선언
upstage_api_key = 'up_yKqThHL17ZcjIGzeOxYkYCaTVqyLb'  # 여기에 Upstage API 키를 입력

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
for number in range(now_number, 28148, -1):     #2024-08-07 수강신청 안내시작..
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
embedding_model = UpstageEmbeddings(model="solar-embedding-1-large", api_key=upstage_api_key)  # Upstage API 키 사용
dense_doc_vectors = np.array(embedding_model.embed_documents(texts))  # 문서 임베딩

# Pinecone에 문서 임베딩 저장 (문서 텍스트와 URL, 날짜를 메타데이터에 포함)
for i, embedding in enumerate(dense_doc_vectors):
    metadata = {
        "title": titles[i],
        "text": texts[i],
        "url": doc_urls[i],  # URL 메타데이터
        "date": doc_dates[i]  # 날짜 메타데이터 추가
    }
    index.upsert([(str(i), embedding.tolist(), metadata)])  # 문서 ID, 임베딩 벡터, 메타데이터 추가



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
        #print("변환된 질문:", user_question)
        n=clean_noun(okt.nouns(users_question))
        nouns=' '.join(n)
        #print(nouns)
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
        query_dense_vector = np.array(embedding_model.embed_query(user_question))



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

        #print("\n\n\n\nFiltering start\n\n\n")

        # 중복 제목 필터링 및 유사도 조정
        for i, doc in enumerate(all_tfidf_docs):
            # 제목에서 공백과 특수 문자 제거 후 소문자로 변환하여 비교
            title = re.sub(r'\s+|\W+', '', doc[0]).lower()
            current_text = re.sub(r'\s+|\W+', '', doc[1]).lower()  # 내용도 정규화하여 비교
            similarity = all_tfidf_similarities[i]

            # 문서가 이미 존재하는지 확인
            is_duplicate_content = False
            for j, existing_doc in enumerate(filtered_docs):
                existing_title = re.sub(r'\s+|\W+', '', existing_doc[0]).lower()
                existing_text = re.sub(r'\s+|\W+', '', existing_doc[1]).lower()

                if existing_title == title:  # 제목이 동일한 경우
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
        #top_filtered_indices = np.argsort(filtered_similarities)[-10:][::-1]
        # print("\n\nFiltered Top 10 Documents Based on Noun Keywords:")
        # for i in top_filtered_indices:
        #     doc = filtered_docs[i]
        #     similarity = filtered_similarities[i]
        #     print(f"Title: {doc[0]}, Date: {doc[1]}, imageURL:{len(doc[4])} URL: {doc[3]}, Similarity: {similarity}")

        # 결합된 유사도 배열 생성
        combined_similarities = (np.array(filtered_similarities)).tolist() + (np.array(pinecone_similarities) * 4).tolist()

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
        # print("\n\n\nfirst final_best_docs")
        # for docs in final_best_docs:
        #     if len(docs) == 4:
        #         print(f"pinecone titles: {docs[0]} doc_dates: {docs[1]} URL: {docs[3]}")
        #     else:
        #         print(f"tf-idf ntitles:   {docs[0]} ,doc_dates: {docs[1]}  URL: {docs[3]}")

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
        return final_best_doc
    except Exception as e:
      print(f"An error occured: {e}")


# 프롬프트 템플릿 정의
prompt_template = """당신은 경북대학교 컴퓨터학부 공지사항을 전달하는 직원이고, 사용자의 질문에 대해 올바른 공지사항의 내용을 참조하여 정확하게 전달해야 할 의무가 있습니다.
현재 한국 시간: {current_time}

주어진 컨텍스트를 기반으로 다음 질문에 답변해주세요:

{context}

질문: {question}

답변 시 다음 사항을 고려해주세요:


1. 질문의 내용이 현재 진행되고 있는 이벤트를 물을 경우, 문서에 주어진 기한과 현재 시간을 비교하여 해당 이벤트가 예정된 것인지, 진행 중인지, 또는 이미 종료되었는지에 대한 정보를 알려주세요.
  예를 들어, 현재 진행되고 있는 해커톤에 대한 질문을 받을 경우, 존재하는 해커톤에 대한 정보를 제공한 다음, 현재 시간을 기준으로 이벤트가 진행되고 있는지에 대한 정보를 한 줄 정도로 추가로 알려주세요.
  단, 질문의 내용에 현재 진행 여부에 관한 정보가 포함되어 있지 않다면 굳이 이벤트 진행 여부에 관한 내용을 제공하지 않아도 됩니다.
2. 질문에서 핵심적인 키워드들을 골라 키워드들과 관련된 문서를 찾아서 해당 문서를 읽고 정확한 내용을 답변해주세요.
3. 질문의 핵심적인 키워드와 관련된 내용의 문서가 여러 개가 있고 질문 내에 구체적인 기간에 대한 정보가 없는 경우 (ex. 2024년 2학기, 3차 등), 가장 최근의 문서를 우선적으로 사용하여 답변해주세요.
  예를 들어, 모집글이 1차, 2차, 3차가 존재하고, 질문 내에 구체적으로 2차에 대한 정보를 묻는 것이 아니라면 가장 최근 문서인 3차에 대한 정보를 알려주세요.
  (ex. (question : 튜터 모집에 대한 정보를 알려줘. => 만약 튜터 모집에 관한 글이 1~7차까지 존재한다면, 7차에 대한 문서를 검색함.) (question : 2차 튜터 모집에 대한 정보를 알려줘. => 2차 튜터 모집에 관한 문서를 검색함.))
  다른 예시로, 3~8월은 1학기이고, 9월~2월은 2학기입니다. 따라서, 사용자의 질문 시간이 2024년 10월 10일이라고 가정하고, 질문에 포함된 핵심 키워드가 2024년 1학기에도 존재하고 2학기에도 존재하는 경우, 질문 내에 구체적으로 1학기에 대한 정보를 묻는 것이 아니라면 가장 최근 문서인 2학기에 대한 정보를 알려주세요.
  (ex. 현재 한국 시간 : 2024년 10월 10일이라고 가정. (question : 수강 정정에 대한 일정을 알려줘. => 2024년 2학기 수강 정정에 대한 문서를 검색함.) (question : 1학기 수강 정정에 대한 일정을 알려줘. => 2024년 1학기 수강 정정에 대한 문서를 검색함.))
4. 수강 정정과 수강정정, 수강 변경과 수강변경, 수강 신청과 수강신청 등과 같이 띄어쓰기가 존재하지만 같은 의미를 가진 단어들은 동일한 키워드로 인식해주세요.
5. 문서의 내용을 그대로 전달하기보다는 질문에서 요구하는 내용에 해당하는 답변만을 제공함으로써 최대한 간결하고 일관된 방식으로 제공해주세요.
6. 질문의 키워드와 일치하거나 비슷한 맥락의 문서를 발견하지 못한 경우, 잘못된 정보를 제공하지 말고 모른다고 답변해주세요.
7. 질문에 대한 답변을 제공하기 전, 사용자의 질문 시간을 현재 한국 시간을 참고하여 한 줄 알려주고 답변해주세요.
   (ex. 답변의 내용이 '성인지 교육은 12월 31일까지입니다.' 라면, 답변의 형태는 '현재 한국 시간 : ____년 __월 __일\n성인지 교육은 12월 31일까지입니다.'라고 알려줘야 합니다.)


답변:"""


refine_prompt_template = """이전 답변: {existing_answer}

다음 정보를 바탕으로 이전 답변을 개선하거나 추가 정보를 반영해주세요:

{context}

질문: {question}

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
    vector_store = FAISS.from_documents(relevant_docs, embedding_model)
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



from IPython.display import display, HTML
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
        # 초기 답변 생성
        existing_answer = qa_chain.invoke({"query": question})  # 초기 답변 생성 및 문자열로 할당
        #print(existing_answer)
        # Refine 체인에서 질문과 관련성 높은 문서만 유지
        refined_chain = RetrievalQA.from_chain_type(
            llm=ChatUpstage(api_key=upstage_api_key),  # LLM으로 Upstage 사용
            chain_type="refine",  # refine 방식 지정
            retriever=retriever,  # retriever를 사용
            return_source_documents=True,  # 출처 문서 반환
            verbose=True  # 디버깅용 상세 출력
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
    




