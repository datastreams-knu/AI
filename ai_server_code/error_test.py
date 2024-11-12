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
pinecone_api_key = 'your_pinecone_api_key'  # 여기에 Pinecone API 키를 입력
index_name = 'prof'

# Upstage API 키 선언
upstage_api_key = 'your_upstage_api_key'  # 여기에 Upstage API 키를 입력

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
        if best_docs and best_docs[0] and len(best_docs[0]) > 3 and best_docs[0][3] != "No content":
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
        if embeddings is None:
            logger.error("Embeddings not initialized. Cannot create FAISS vector store.")
            return None, None, None
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
      # 사용자 질문
      okt = Okt()
      query_noun=transformed_query(user_question)
      #print(f"=================\n\n question: {user_question} 추출된 명사: {query_noun}")

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
              if "수강" in query_noun and ("수강꾸러미" or "수강 꾸러미") in title:
                  similarities[idx]-=0.7
              if "외국인" not in query_noun and "외국인" in title:
                  similarities[idx]-=1
              # 본문 내용이 "No content"인 경우 유사도 0.5 추가 (조정값은 필요에 따라 변경 가능)
              if texts[idx] == "No content":
                  similarities[idx] *=3  # 본문이 "No content"인 경우 유사도를 높임
              if '마일리지' in query_noun and '마일리지' in title:
                  similarities[idx]+=1
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


      #####파인콘으로 구한  문서 추출 방식 결합하기.
      combine_dense_docs = []

      # 1. 본문 기반 문서를 combine_dense_docs에 먼저 추가
      for idx, text_doc in enumerate(pinecone_docs_text):
          text_similarity = pinecone_similarities_text[idx]*3.5
          combine_dense_docs.append((text_similarity, text_doc))  # (유사도, (제목, 날짜, 본문, URL))

      ####query_noun에 포함된 키워드로 유사도를 보정
      # 유사도 기준으로 내림차순 정렬
      combine_dense_docs.sort(key=lambda x: x[0], reverse=True)

      ## 결과 출력
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



      def last_filter_keyword(DOCS,query_noun):
          # 필터링에 사용할 키워드 리스트
          Final_best=DOCS
          # 키워드가 포함된 경우 유사도를 조정하고, 유사도 기준으로 내림차순 정렬
          for idx, doc in enumerate(DOCS):
              score, title, date, text, url, image = doc
              # query_nouns에 없는 키워드가 본문에 포함되었는지 확인
              if any(keyword in text for keyword in ['계약학과', '대학원', '타대학원']) and not any(keyword in query_noun for keyword in ['계약학과', '대학원', '타대학원']):
                  score -= 0.1  # 유사도 점수를 0.1 낮추기
              if '대학원' not in query_noun and '대학원생' not in query_noun and ('대학원' in title or '대학원생' in title):
                  score-=1
              # 조정된 유사도 점수를 사용하여 다시 리스트에 저장
              Final_best[idx] = (score, title, date, text,  url, image)
              #print(Final_best[idx])
          return Final_best

      final_best_docs=last_filter_keyword(final_best_docs,query_noun)
      
      
      def cluster_documents_by_similarity(docs, threshold=0.89):
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
                      #print(f"{doc[0]} {cluster[0][0]}  {title} {cluster_title}")
                      if (-doc[0]+cluster[0][0]<0.26):
                        cluster.append(doc)
                      added_to_cluster = True
                      break

              # 유사한 클러스터가 없으면 새로운 클러스터 생성
              if not added_to_cluster:
                  clusters.append([doc])

          return clusters



      # Step 1: Adjust similarity scores based on the presence of query_noun


      # Step 2: Cluster documents by similarity
      clusters = cluster_documents_by_similarity(final_best_docs)

      query_nouns=transformed_query(user_question)
      # 날짜 형식을 datetime 객체로 변환하는 함수
      def parse_date(date_str):
          # '작성일'을 제거하고 공백을 제거한 뒤 날짜 형식으로 변환
          clean_date_str = date_str.replace("작성일", "").strip()
          return datetime.strptime(clean_date_str, "%y-%m-%d %H:%M")
      # Step 3: Compare cluster[0] cluster[1] top similarity and check condition
      top_0_cluster_similar=clusters[0][0][0]
      top_1_cluster_similar=clusters[1][0][0]
      #print(f"{top_0_cluster_similar} {top_1_cluster_similar}")
      if (top_0_cluster_similar-top_1_cluster_similar<=0.38): ## 질문이 모호했다는 의미일 수 있음.. (예를 들면 수강신청 언제야? 인데 구체적으로 1학기인지, 2학기인지, 겨울, 여름인지 모르게..)
            # 날짜를 비교해 더 최근 날짜를 가진 클러스터 선택
          date1 = parse_date(clusters[0][0][2])
          date2 = parse_date(clusters[1][0][2])
          if date1<date2:
            result_docs=clusters[1]
          else:
            result_docs=clusters[0]
          sorted_cluster = sorted(result_docs, key=lambda doc: doc[2], reverse=True)
      else: #질문이 모호하지 않을 가능성 업업
          number_pattern = r"\d"
          keywords = ["최근", "최신", "현재", "지금"]
          if (any(keyword in word for word in query_nouns for keyword in keywords) or not any(re.search(number_pattern, word) for word in query_nouns)):
              result_docs=clusters[0]
              sorted_cluster = sorted(result_docs, key=lambda doc: doc[2], reverse=True)
          else:
            result_docs=clusters[0]
            sorted_clusted=last_filter_keyword(result_docs,query_nouns)
            sorted_cluster = sorted(clusters[0], key=lambda doc: doc[0], reverse=True)

      return [sorted_cluster[0]]
      '''
      final_best_docs=sorted_cluster
      # Step 4: 결과 출력
      # print("\n\n\n\n최종 상위 문서 (유사도 및 날짜 기준 정렬):")
      # for idx, (scor, titl, dat, tex, ur, image_ur) in enumerate(sorted_cluster):
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
      '''

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
