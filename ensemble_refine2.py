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

# Pinecone API 키와 인덱스 이름 선언
pinecone_api_key = 'cd22a6ee-0b74-4e9d-af1b-a1e83917d39e'  # 여기에 Pinecone API 키를 입력
index_name = 'test'

# Upstage API 키 선언
upstage_api_key = 'up_pGRnryI1JnrxChGycZmswEZm934Tf'  # 여기에 Upstage API 키를 입력

# Pinecone API 설정 및 초기화
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)

# URL에서 텍스트와 날짜를 추출하는 함수
def extract_text_and_date_from_url(urls):
    all_data = []

    def fetch_text_and_date(url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # 텍스트 추출
            paragraphs = soup.find_all('p')
            text = "\n".join([para.get_text() for para in paragraphs])

            # 날짜 추출
            date_element = soup.select_one("strong.if_date")  # 수정된 선택자
            date = date_element.get_text(strip=True) if date_element else "Unknown Date"

            return text, date, url  # 문서 텍스트, 날짜, URL 반환
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return None, None, url


    with ThreadPoolExecutor() as executor:
        results = executor.map(fetch_text_and_date, urls)

    all_data = [(text, date, url) for text, date, url in results if text]
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
for number in range(now_number, 27726, -1):     #27726
    urls.append("https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_1&wr_id=" + str(number))

# URL에서 문서와 날짜 추출
document_data = extract_text_and_date_from_url(urls)


# 텍스트 분리기 초기화
class CharacterTextSplitter:
    def __init__(self, chunk_size=800, chunk_overlap=150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i:i + self.chunk_size])
        return chunks

text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=150)

# 텍스트 분리 및 URL과 날짜 매핑
texts = []
doc_urls = []
doc_dates = []
for doc, date, url in document_data:
    if isinstance(doc, str):
        split_texts = text_splitter.split_text(doc)
        texts.extend(split_texts)
        doc_urls.extend([url] * len(split_texts))
        doc_dates.extend([date] * len(split_texts))  # 분리된 각 텍스트에 동일한 날짜 적용
    else:
        raise TypeError("리스트 내 각 문서는 문자열이어야 합니다.")

# 1. Sparse Retrieval (TF-IDF)
def initialize_tfidf_model(texts):
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(texts)
    return vectorizer, doc_vectors

vectorizer, doc_vectors = initialize_tfidf_model(texts)

# 2. Dense Retrieval (Upstage 임베딩)
embedding_model = UpstageEmbeddings(model="solar-embedding-1-large", api_key=upstage_api_key)  # Upstage API 키 사용
dense_doc_vectors = np.array(embedding_model.embed_documents(texts))  # 문서 임베딩

# Pinecone에 문서 임베딩 저장 (문서 텍스트와 URL, 날짜를 메타데이터에 포함)
for i, embedding in enumerate(dense_doc_vectors):
    metadata = {
        "text": texts[i],
        "url": doc_urls[i],  # URL 메타데이터
        "date": doc_dates[i],  # 날짜 메타데이터 추가
    }
    index.upsert([(str(i), embedding.tolist(), metadata)])  # 문서 ID, 임베딩 벡터, 메타데이터 추가

def get_korean_time():
    return datetime.now(pytz.timezone('Asia/Seoul'))
# 사용자 질문에 대한 AI 답변 생성 (시간 체크 기능 추가)
def get_best_docs(user_question):
    try:
        # Sparse Retrieval: TF-IDF 벡터화
        query_tfidf_vector = vectorizer.transform([user_question])
        tfidf_cosine_similarities = cosine_similarity(query_tfidf_vector, doc_vectors).flatten()

        # Dense Retrieval: Upstage 임베딩을 통한 유사도 계산
        query_dense_vector = np.array(embedding_model.embed_query(user_question))

        # Pinecone에서 가장 유사한 벡터 찾기
        pinecone_results = index.query(vector=query_dense_vector.tolist(), top_k=10, include_values=True, include_metadata=True)
        pinecone_similarities = [res['score'] for res in pinecone_results['matches']]
        pinecone_docs = [(res['metadata']['text'], res['score'], res['metadata'].get('url', 'No URL'),
                                  res['metadata'].get('date', 'No Date')) for res in pinecone_results['matches']]

        # TF-IDF에서 상위 7개 문서의 유사도만 가져옵니다.
        top_tfidf_indices = np.argsort(tfidf_cosine_similarities)[-7:][::-1]  # 상위 5개 인덱스
        tfidf_best_docs = [(texts[i], tfidf_cosine_similarities[i], doc_urls[i]) for i in top_tfidf_indices]  # URL 포함

        # 두 유사도 배열 결합
        combined_similarities = np.concatenate((tfidf_cosine_similarities[top_tfidf_indices], np.array(pinecone_similarities)))

        # 가장 유사한 문서 인덱스 계산
        combined_best_doc_indices = np.argsort(combined_similarities)[-7:][::-1]

        # 결과 문서 목록 생성
        best_docs = []

        for idx in combined_best_doc_indices:
            if idx < len(tfidf_best_docs):
                best_docs.append(tfidf_best_docs[idx])
            else:
                pinecone_index = idx - len(tfidf_best_docs)
                best_docs.append((pinecone_docs[pinecone_index][0], pinecone_docs[pinecone_index][2], combined_similarities[idx], pinecone_docs[pinecone_index][3]))  # 텍스트와 URL

        return best_docs
    except Exception as e:
        return f"답변을 생성하는 중 오류가 발생했습니다: {e}"

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


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


# Refine 체인에서 질문과 관련성 높은 문서만 유지
def get_answer_from_chain(best_docs, user_question):
    documents = []
    doc_texts = [doc[0] for doc in best_docs]
    doc_urls = [doc[2] for doc in best_docs]  # URL을 별도로 저장
    documents = [Document(page_content=text, metadata={"url": url}) for text, url in zip(doc_texts, doc_urls)]

    # 키워드 기반 관련성 필터링 추가 (질문과 관련 없는 문서 제거)
    relevant_docs = [doc for doc in documents if any(keyword in doc.page_content for keyword in user_question.split())]

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

    # 초기 답변 생성
    existing_answer = qa_chain.invoke({"query": question})  # AI의 초기 답변 생성

    # Refine 체인에서 질문과 관련성 높은 문서만 유지
    refined_chain = RetrievalQA.from_chain_type(
        llm=ChatUpstage(api_key=upstage_api_key),  # LLM으로 Upstage 사용
        chain_type="refine",  # refine 방식 지정
        retriever=retriever,  # retriever를 사용
        return_source_documents=True,  # 출처 문서 반환
        verbose=True,  # 디버깅용 상세 출력
    )

    # 이전 답변을 반영하여 리파인된 최종 답변 생성
    final_answer = refined_chain.invoke({
        "query": question,
        "existing_answer": existing_answer,
        "context": format_docs(relevant_docs)
    })

    answer_result = final_answer.get('result')

    # 상위 3개의 참조한 문서의 URL 포함 형식으로 반환
    doc_references = "\n".join([f"\n참고 문서 URL: {doc[1]}" for doc in top_docs[:3] if doc[1] != 'No URL'])

    # AI의 최종 답변과 참조 URL을 함께 반환
    return f"{answer_result}\n\n------------------------------------------------\n항상 정확한 답변을 제공하지 못할 수 있습니다.\n아래의 URL들을 참고하여 정확하고 자세한 정보를 확인하세요.\n{doc_references}"

