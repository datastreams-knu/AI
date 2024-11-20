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
from pymongo import MongoClien

#############################################
#		enviroment                          #
#############################################

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


db = client["test_db"]
collection = db["test_collection"]

titles = collection.find({"_id"}, {"data" : 1, "_id" : 0})


def best_docs(user_question):
     
      # 사용자 질문에서 추출한 명사와 각 문서 제목에 대한 유사도를 조정하는 함수
      def adjust_similarity_scores(query_noun, titles, similarities):
          # 각 제목에 대해 명사가 포함되어 있는지 확인 후 유사도 조정
          # titles 데베에서 불러오기 코드 필요할듯.
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

      query_noun = transformed_query(user_question) # 사용자 질문 명사화
      
      title_question_similarities = bm25_titles.get_scores(query_noun)  # 제목과 사용자 질문 간의 유사도 계산
      title_question_similarities /= 25
      
      top_20_titles_idx = np.argsort(title_question_similarities)[-20:][::-1] # 유사도 기준 상위 20개 문서 선택
      Bm25_best_docs = [(titles[i], doc_dates[i], texts[i], doc_urls[i],image_url[i]) for i in top_20_titles_idx]
      
      adjusted_similarities = adjust_similarity_scores(query_noun, titles, title_question_similarities) # 조정된 유사도 계산
      

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
      dense_noun = transformed_query(user_question)
      query_title_dense_vector = np.array(embeddings.embed_query(dense_noun))  # 사용자 질문에 대한 제목 임베딩

      #####파인콘으로 구한  문서 추출 방식 결합하기.
      combine_dense_docs = []

      # 1. 본문 기반 문서를 combine_dense_docs에 먼저 추가
      for idx, text_doc in enumerate(pinecone_docs_text):
          text_similarity = pinecone_similarities_text[idx] * 3.5
          combine_dense_docs.append((text_similarity, text_doc))  # (유사도, (제목, 날짜, 본문, URL))

      ####query_noun에 포함된 키워드로 유사도를 보정
      # 유사도 기준으로 내림차순 정렬
      combine_dense_docs.sort(key=lambda x: x[0], reverse=True)

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

      # Step 2: Cluster documents by similarity
      clusters = cluster_documents_by_similarity(final_best_docs)

      query_nouns = transformed_query(user_question)
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
    
    # top_docs 인덱스 구성
    # 0: 유사도, 1: 제목, 2: 날짜, 3: 본문내용, 4: url, 5: 이미지url

    # 이미지 + 공지사항만 존재하는 경우.
    if len(top_docs[0]) == 6 and top_docs[0][5] != "No content" and top_docs[0][3] == "No content" and top_docs[0][0] > 1.8:
        # 이미지 URL 조인
        if isinstance(top_docs[0][5], list):
            image_urls = "\n".join(top_docs[0][5])
        else:
            image_urls = top_docs[0][5]
        
        doc_references = top_docs[0][4]

        # JSON 형식으로 반환할 객체 생성
        only_image_response = {
            "answer": None,
            "references": doc_references,
            "disclaimer": "항상 정확한 답변을 제공하지 못할 수 있습니다. 아래의 URL들을 참고하여 정확하고 자세한 정보를 확인하세요.",
            "images": image_urls
        }
        return only_image_response

    # 이미지 + LLM 답변이 있는 경우.
    else:
        qa_chain, retriever, relevant_docs = get_answer_from_chain(top_docs, question)

        # 기존의 교수님 이미지 URL 저장 코드 중 중복된 URL 방지 부분
        if top_docs[0][5] and top_docs[0][5] != "No content":
            if isinstance(top_docs[0][5], list):
                image_urls = "\n".join(top_docs[0][5])
            else:
                image_urls = top_docs[0][5]
        else:
            image_urls = None

        # 공지사항에 존재하지 않을 경우
        notice_url = "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_1"
        not_in_notices_response = {
            "answer": "해당 질문은 공지사항에 없는 내용입니다.\n 자세한 사항은 공지사항을 살펴봐주세요.",
            "references": notice_url,
            "disclaimer": "항상 정확한 답변을 제공하지 못할 수 있습니다. 아래의 URL들을 참고하여 정확하고 자세한 정보를 확인하세요.",
            "images": None
        }

        # 답변 생성 실패
        if not qa_chain or not relevant_docs:
            if top_docs[0][5] != "No content" and top_docs[0][0] > 1.8:
                data = {
                    "answer": "해당 질문에 대한 내용은 이미지 파일로 확인해주세요.",
                    "references": top_docs[0][4],
                    "disclaimer": "항상 정확한 답변을 제공하지 못할 수 있습니다. 아래의 URL들을 참고하여 정확하고 자세한 정보를 확인하세요.",
                    "images": image_urls
                }
                return data
            else:
                return not_in_notices_response

        # 유사도가 낮은 경우
        if top_docs[0][0] < 1.8:
            return not_in_notices_response

        # LLM에서 답변을 생성하는 경우
        answer_result = qa_chain.invoke(question)
        doc_references = "\n".join([
            f"\n참고 문서 URL: {doc.metadata['url']}"
            for doc in relevant_docs[:1] if doc.metadata.get('url') != 'No URL'
        ])

        # JSON 형식으로 반환할 객체 생성
        data = {
            "answer": answer_result,
            "references": doc_references,
            "disclaimer": "항상 정확한 답변을 제공하지 못할 수 있습니다. 아래의 URL들을 참고하여 정확하고 자세한 정보를 확인하세요.",
            "images": image_urls
        }

        return data
