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
from langchain_community.vectorstores import FAISS
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
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pinecone API 키와 인덱스 이름 선언
pinecone_api_key = 'cd22a6ee-0b74-4e9d-af1b-a1e83917d39e'  # 여기에 Pinecone API 키를 입력
index_name = 'prof'

# Upstage API 키 선언
upstage_api_key = 'up_pGRnryI1JnrxChGycZmswEZm934Tf'  # 여기에 Upstage API 키를 입력

# Pinecone API 설정 및 초기화
try:
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
except Exception as e:
    logger.error(f"Error initializing Pinecone: {e}")
    index = None

# Upstage Embeddings 설정
try:
    embeddings = UpstageEmbeddings(api_key=upstage_api_key, model="solar-embedding-1-large")
except Exception as e:
    logger.error(f"Error initializing Upstage Embeddings: {e}")
    embeddings = None

def get_korean_time():
    return datetime.now(pytz.timezone('Asia/Seoul'))

def get_answer_from_chain(best_docs, user_question):
    documents = []
    try:
        # 문서 생성
        if best_docs[0][3] != "No content":
            doc_titles = [doc[1] for doc in best_docs]
            doc_dates = [doc[2] for doc in best_docs]
            doc_texts = [doc[3] for doc in best_docs]
            doc_urls = [doc[4] for doc in best_docs]

            documents = [
                Document(page_content=text, metadata={"title": title, "url": url, "doc_date": datetime.strptime(date, '작성일%y-%m-%d %H:%M')})
                for title, text, url, date in zip(doc_titles, doc_texts, doc_urls, doc_dates)
            ]
    except Exception as e:
        logger.error(f"Error generating documents: {e}")
        return None, None, None

    # 빈 문서 리스트 체크
    if not documents:
        logger.warning("No documents available for retrieval.")
        return None, None, None

    # 키워드 기반 관련성 필터링 추가 (질문과 관련 없는 문서 제거)
    try:
        okt = Okt()
        query_nouns = transformed_query(user_question)
        relevant_docs = [doc for doc in documents if any(keyword in doc.page_content for keyword in query_nouns)]
        if not relevant_docs:
            logger.warning("No relevant documents found based on query keywords.")
            return None, None, None
    except Exception as e:
        logger.error(f"Error filtering documents by relevance: {e}")
        return None, None, None

    # FAISS 벡터 스토어 생성
    try:
        vector_store = FAISS.from_documents(relevant_docs, embeddings)
    except Exception as e:
        logger.error(f"Error creating FAISS vector store: {e}")
        return None, None, None

    # Retriever 생성
    try:
        retriever = vector_store.as_retriever()
    except Exception as e:
        logger.error(f"Error creating retriever from vector store: {e}")
        return None, None, None

    # LLM 초기화
    try:
        llm = ChatUpstage(api_key=upstage_api_key)
    except Exception as e:
        logger.error(f"Error initializing ChatUpstage LLM: {e}")
        return None, None, None

    # 체인 생성
    try:
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
    except Exception as e:
        logger.error(f"Error creating QA chain: {e}")
        return None, None, None

    return qa_chain, retriever, relevant_docs

def format_docs(docs):
    try:
        return "\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        logger.error(f"Error formatting documents: {e}")
        return ""

def transformed_query(content):
    try:
        # 중복된 단어를 제거한 명사를 담을 리스트
        query_nouns = []
        okt = Okt()
        additional_nouns = [noun for noun in okt.nouns(content) if len(noun) >= 1]
        query_nouns += additional_nouns
        query_nouns = list(set(query_nouns))
        return query_nouns
    except Exception as e:
        logger.error(f"Error transforming query: {e}")
        return []

def best_docs(user_question):
    try:
        # 사용자 질문에서 가장 유사한 문서 가져오기
        # (이 부분은 실제 문서 유사도 계산 로직이 포함되어야 함)
        pass
    except Exception as e:
        logger.error(f"Error retrieving best documents: {e}")
        return []

# Example usage
if __name__ == "__main__":
    question = "수강 신청 일정 알려줘"
    top_docs = best_docs(question)
    qa_chain, retriever, relevant_docs = get_answer_from_chain(top_docs, question)
    if qa_chain:
        try:
            answer = qa_chain.invoke(question)
            print(answer)
        except Exception as e:
            logger.error(f"Error invoking QA chain: {e}")
    else:
        logger.warning("QA chain could not be created.")