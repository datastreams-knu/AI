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




def get_best_docs(user_question):
    try:
        # 사용자 질문을 전처리하여 공백 제거 후 명사만 추출
        okt = Okt()

        # 모든 공백 제거
        #cleaned_question = user_question.replace(" ", "")
        #print(cleaned_question)
        cleaned_question=user_question
        nouns = okt.nouns(cleaned_question)  # 명사 추출
        print(nouns)

        # 리스트를 공백으로 구분된 문자열로 변환
        nouns = ' '.join(nouns)

  ##############################################################
  #####    TEXT 기준으로 TF-IDF 유사도 체크 (NOUN / USER_QUESTION)
  ##############################################################

        # Sparse Retrieval: TF-IDF 벡터화  TEXT 기준임.
        query_tfidf_vector = vectorizer.transform([nouns])
        # TF-IDF 코사인 유사도 계산
        tfidf_cosine_similarities = cosine_similarity(query_tfidf_vector, doc_vectors).flatten()



        # Sparse Retrieval: TF-IDF 벡터화  TEXT 기준임.
        querys_tfidf_vector = vectorizer.transform([user_question])
        # TF-IDF 코사인 유사도 계산
        tfidfs_cosine_similarities = cosine_similarity(querys_tfidf_vector, doc_vectors).flatten()


  ##############################################################
  #####        Title 기준으로 TF-IDF 유사도 체크 (NOUN / USER_QUESTION)
  ##############################################################

        #### nouns 기준으로 실행
        # 제목에 대한 TF-IDF 벡터화
        query_title_vector = title_vectorizer.transform([nouns])
        # 제목과 사용자 질문 간의 유사도 계산
        title_cosine_similarities = cosine_similarity(query_title_vector, title_vectors).flatten()



        #####   user_question 기준으로 실행
        # 제목에 대한 TF-IDF 벡터화
        query_titles_vector = title_vectorizer.transform([user_question])
        # 제목과 사용자 질문 간의 유사도 계산
        titles_cosine_similarities = cosine_similarity(query_titles_vector, title_vectors).flatten()

  ##############################################################
  ##############################################################


        # Dense Retrieval: Upstage 임베딩을 통한 유사도 계산
        query_dense_vector = np.array(embedding_model.embed_query(user_question))

        # Pinecone에서 가장 유사한 벡터 찾기
        pinecone_results = index.query(vector=query_dense_vector.tolist(), top_k=10, include_values=True, include_metadata=True)
        pinecone_similarities = [res['score'] for res in pinecone_results['matches']]
        pinecone_docs = [(res['metadata'].get('title', 'No Title'), res['metadata'].get('date', 'No Date'),
                          res['metadata'].get('text', ''), res['metadata'].get('url', 'No URL')) for res in pinecone_results['matches']]

        # TF-IDF에서 상위 10개 문서의 유사도, texts기준
        top_tfidf_indices = np.argsort(tfidf_cosine_similarities)[-10:][::-1]
        tfidf_best_docs = [(titles[i], doc_dates[i], texts[i], doc_urls[i],image_url[i]) for i in top_tfidf_indices]

        # TF-IDF에서 상위 10개 문서의 유사도, texts기준
        tops_tfidf_indices = np.argsort(tfidfs_cosine_similarities)[-10:][::-1]
        tfidf1_best_docs = [(titles[i], doc_dates[i], texts[i], doc_urls[i],image_url[i]) for i in top_tfidf_indices]
        # TF-IDF에서 상위 10개 문서의 유사도, title기준
        top_title_indices = np.argsort(title_cosine_similarities)[-10:][::-1]
        tfidf2_best_docs = [(titles[i], doc_dates[i], texts[i], doc_urls[i],image_url[i]) for i in top_tfidf_indices]
        # TF-IDF에서 상위 10개 문서의 유사도, title기준
        top_titles_indices = np.argsort(titles_cosine_similarities)[-10:][::-1]
        tfidf3_best_docs = [(titles[i], doc_dates[i], texts[i], doc_urls[i],image_url[i]) for i in top_tfidf_indices]
        


####################################################################################3
        
         # 결과 출력 (TF-IDF)
        print("Noun: Title Best Documents:")
        for idx in top_title_indices:
            print(f"Title: {titles[idx]}, Date: {doc_dates[idx]}, URL: {doc_urls[idx]}, Similarity: {title_cosine_similarities[idx]}")
        print("-----------------------")
        # 결과 출력 (TF-IDF)
        print("user_question: Titles Best Documents:")
        for idx in top_titles_indices:
            print(f"Title: {titles[idx]}, Date: {doc_dates[idx]}, URL: {doc_urls[idx]}, Similarity: {titles_cosine_similarities[idx]}")
        print("-----------------------")
        # 결과 출력 (TF-IDF)
        print("NOUN: TF-IDF Best Documents:")
        for idx in top_tfidf_indices:
            print(f"Title: {titles[idx]}, Date: {doc_dates[idx]}, URL: {doc_urls[idx]}, Similarity: {tfidf_cosine_similarities[idx]}")
        print("-----------------------")
        # 결과 출력 (TF-IDF)
        print("user_question: TF-IDFs Best Documents:")
        for idx in tops_tfidf_indices:
            print(f"Title: {titles[idx]}, Date: {doc_dates[idx]}, URL: {doc_urls[idx]}, Similarity: {tfidfs_cosine_similarities[idx]}")
        print("-----------------------")
        # 결과 출력 (Pinecone)
        print("\nPinecone Best Documents:")
        for i, doc in enumerate(pinecone_docs):
            print(f"Title: {doc[0]}, Date: {doc[1]}, URL: {doc[3]}, Similarity: {pinecone_similarities[i]}")
        print("-----------------------")


#########################################################################################################



        # 필터링된 인덱스와 유사도를 저장할 리스트
        total_top_tfidf_indices = []
        filtered_tfidf_similarities = []

        # 10개의 상위 문서 목록에서 제목에 `nouns` 키워드가 포함되어 있는지 확인하는 함수
        def filter_docs_by_nouns(docs, similarities, nouns):
            filtered_indices = []
            filtered_similarities = []
            for idx, doc in enumerate(docs):
                title = doc[0]  # 제목 정보가 첫 번째 요소에 위치
                # `nouns`에 있는 키워드 중 하나라도 제목에 포함되어 있으면 추가
                if any(noun in title for noun in nouns.split()):
                    filtered_indices.append(idx)
                    filtered_similarities.append(similarities[idx])
            return filtered_indices, filtered_similarities

        # 각 상위 문서 목록에 대해 제목에 키워드가 포함된 인덱스와 유사도 필터링
        filtered_tfidf_indices, tfidf_similarities = filter_docs_by_nouns(tfidf_best_docs, tfidf_cosine_similarities, nouns)
        filtered_tfidf1_indices, tfidf1_similarities = filter_docs_by_nouns(tfidf1_best_docs, tfidfs_cosine_similarities, nouns)
        filtered_tfidf2_indices, tfidf2_similarities = filter_docs_by_nouns(tfidf2_best_docs, title_cosine_similarities, nouns)
        filtered_tfidf3_indices, tfidf3_similarities = filter_docs_by_nouns(tfidf3_best_docs, titles_cosine_similarities, nouns)

        # 필터링된 인덱스와 유사도를 `total_top_tfidf_indices`와 `filtered_tfidf_similarities`에 추가
        total_top_tfidf_indices.extend([top_tfidf_indices[i] for i in filtered_tfidf_indices])
        filtered_tfidf_similarities.extend(tfidf_similarities)

        total_top_tfidf_indices.extend([tops_tfidf_indices[i] for i in filtered_tfidf1_indices])
        filtered_tfidf_similarities.extend(tfidf1_similarities)

        total_top_tfidf_indices.extend([top_title_indices[i] for i in filtered_tfidf2_indices])
        filtered_tfidf_similarities.extend(tfidf2_similarities)

        total_top_tfidf_indices.extend([top_titles_indices[i] for i in filtered_tfidf3_indices])
        filtered_tfidf_similarities.extend(tfidf3_similarities)


        # Pinecone 문서 필터링
        #filtered_pinecone_indices, filtered_pinecone_similarities = filter_docs_by_nouns(pinecone_docs, pinecone_similarities, nouns)
        # 필터링된 문서와 유사도 출력
        #print(f"\n\nFiltered Pinecone Documents (Count: {len(filtered_pinecone_indices)}):")
        #for idx in filtered_pinecone_indices:
        #    doc = pinecone_docs[idx]
        #    similarity = filtered_pinecone_similarities[filtered_pinecone_indices.index(idx)]  # 유사도 매칭
        #    print(f"Title: {doc[0]}, Date: {doc[1]}, URL: {doc[3]}, Similarity: {similarity}")
        
        
        # 필터링된 문서 상위 10개 출력
        print("\n\nFiltered Top 10 Documents Based on Noun Keywords:")
        for idx in total_top_tfidf_indices[:10]:
            print(f"Title: {titles[idx]}, Date: {doc_dates[idx]}, URL: {doc_urls[idx]}, Similarity: {tfidfs_cosine_similarities[idx]}")



        # `filtered_tfidf_similarities`와 Pinecone 유사도를 결합하여 최종 유사도 배열 생성
        combined_similarities = np.concatenate(
            (np.array(filtered_tfidf_similarities), np.array(pinecone_similarities))
        )

        # 최종적으로 가장 높은 유사도를 가진 문서들을 정렬하여 추출
        final_top_indices = np.argsort(combined_similarities)[-10:][::-1]
        # 최종 문서 목록 생성 및 출력
        print("\nCombined Top 10 Final Documents:")
        for i in final_top_indices:
            if i < len(filtered_tfidf_similarities):
                # TF-IDF 기반 문서
                idx = total_top_tfidf_indices[i]
                print(f"Title: {titles[idx]}, Date: {doc_dates[idx]}, URL: {doc_urls[idx]}, Similarity: {filtered_tfidf_similarities[i]}")
            else:
                # Pinecone 기반 문서
                pinecone_idx = i - len(filtered_tfidf_similarities)
                doc = pinecone_docs[pinecone_idx]
                print(f"Title: {doc[0]}, Date: {doc[1]}, URL: {doc[3]}, Similarity: {pinecone_similarities[pinecone_idx]}")

        
        final_best_docs = [
            tfidf_best_docs[i] if i < len(filtered_tfidf_similarities) 
            else pinecone_docs[i - len(filtered_tfidf_similarities)] 
            for i in final_top_indices
        ]
        return final_best_docs
    except Exception as e:
      print(f"An error occured: {e}")



##############################################################################################################






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
    doc_titles= [doc[0] for doc in best_docs]
    doc_texts = [doc[2] for doc in best_docs]
    doc_urls = [doc[3] for doc in best_docs]  # URL을 별도로 저장
    doc_dates=[doc[1] for doc in best_docs]
    doc_image_urls = [doc[4] if doc[4] else "No content" for doc in best_docs]

    #documents = [Document(page_content=text, metadata={"url": url}) for text, url in zip(doc_texts, doc_urls)]

    documents = [
        Document(page_content=text, metadata={"title": title, "url": url, "doc_date": datetime.strptime(date, '작성일%y-%m-%d %H:%M')})
        for title, text, url, date in zip(doc_titles, doc_texts, doc_urls, doc_dates)
    ]

    # 1차: 제목 필터링 - 제목에 키워드 포함 여부 확인
    title_relevant_docs = [doc for doc in documents if any(keyword in doc.metadata["title"] for keyword in user_question.split())]
    for doc in title_relevant_docs:
      title = doc.metadata["title"]
      date = doc.metadata["doc_date"].strftime('%Y-%m-%d %H:%M')  # 날짜를 원하는 형식으로 변환
      url = doc.metadata["url"]

    # 2차: 내용 필터링 - 내용에 키워드 포함 여부 확인
    relevant_docs = [doc for doc in title_relevant_docs if any(keyword in doc.page_content for keyword in user_question.split())]
    # 3차: 최신순 정렬
    relevant_docs = sorted(relevant_docs, key=lambda x: x.metadata["doc_date"], reverse=True)
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



def get_ai_message(question):
    top_docs = get_best_docs(question)  # 가장 유사한 문서 가져오기
    qa_chain, retriever, relevant_docs = get_answer_from_chain(top_docs, question)  # 답변 생성 체인 생성
    for doc in top_docs:
      title = doc.metadata["title"]
      date = doc.metadata["doc_date"].strftime('%Y-%m-%d %H:%M')  # 날짜를 원하는 형식으로 변환
      url = doc.metadata["url"]

      print(f"Title0: {title}")
      print(f"Date0: {date}")
      print(f"URL0: {url}")


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