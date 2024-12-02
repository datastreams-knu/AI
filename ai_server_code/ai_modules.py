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
import re
from datetime import datetime
import pytz
from langchain.schema.runnable import Runnable
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain.schema.runnable import RunnableSequence, RunnableMap
from langchain_core.runnables import RunnableLambda
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from konlpy.tag import Okt
from collections import defaultdict
import numpy as np
from IPython.display import display, HTML
from rank_bm25 import BM25Okapi
from difflib import SequenceMatcher
from pymongo import MongoClient
from pinecone import Index

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


# 단어 명사화 함수.
def transformed_query(content):
    # 중복된 단어를 제거한 명사를 담을 리스트
    query_nouns = []

    # 1. 숫자와 특정 단어가 결합된 패턴 추출 (예: '2024학년도', '1월' 등)
    pattern = r'\d+(?:학년도|년|학년|월|일|학기|시|분|초|기|개|차)?'
    number_matches = re.findall(pattern, content)
    query_nouns += number_matches
    # 추출된 단어를 content에서 제거
    for match in number_matches:
        content = content.replace(match, '')


    # 1. 영어 단어를 단독으로 또는 한글과 결합된 경우 추출 (영어만 추출)
    english_pattern = r'[a-zA-Z]+'
    english_matches = re.findall(english_pattern, content)

    # 대문자로 변환 후 query_nouns에 추가
    english_matches_upper = [match.upper() for match in english_matches]
    query_nouns += english_matches_upper

    # content에서 영어 단어 제거
    for match in english_matches:
        content = re.sub(rf'\b{re.escape(match)}\b', '', content)

    ## 직원 E9호관 있는거 추가하려고함.
    if '공대' in content:
        query_nouns.append('E')
    if '설명회' in content:
        query_nouns.append('설명회')
    if '컴학' in content:
        query_nouns.append('컴퓨터학부')
    if '컴퓨터' in content and '비전' in content:
        query_nouns.append('컴퓨터비전')
        content = content.replace('컴퓨터 비전', '컴퓨터비전')
        content = content.replace('컴퓨터비전', '')
    if '컴퓨터' in content and '학부' in content:
        query_nouns.append('컴퓨터학부')
        content = content.replace('컴퓨터 학부', '컴퓨터학부')
        content = content.replace('컴퓨터학부', '')
    if '선발' in content:
        content=content.replace('선발','')
    if '차' in content:
        query_nouns.append('차')
    if '국가 장학금' in content:
        query_nouns.append('국가장학금')
        content=content.replace('국가 장학금','')
    if '종프' in content:
        query_nouns.append('종합설계프로젝트')
    if '종합설계프로젝트' in content:
        query_nouns.append('종합설계프로젝트')
    if '대회' in content:
        query_nouns.append('경진대회')
        content=content.replace('대회','')
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
    if '에이빅' in content:
        query_nouns.append('에이빅')
        query_nouns.append('ABEEK')
        content=content.replace('에이빅','')
    if '선이수' in content:
        query_nouns.append('선이수')
        content=content.replace('선이수','')
    if '선후수' in content:
        query_nouns.append('선이수')
        content=content.replace('선후수','')
    if '학자금' in content:
        query_nouns.append('학자금')
        content=content.replace('학자금','')   
    if  any(keyword in content for keyword in ['오픈 소스','오픈소스']):
        query_nouns.append('오픈소스')
        content=content.replace('오픈 소스','')
        content=content.replace('오픈소스','') 
    if '채용' in content and any(keyword in content for keyword in ['모집','공고']):
        if '모집' in content:
          content=content.replace('모집','')
        if '공고' in content:
          content=content.replace('공고','')
    # 비슷한 의미 모두 추가 (세미나)
    related_keywords = ['세미나','특강', '강연']
    if any(keyword in content for keyword in related_keywords):
        for keyword in related_keywords:
            query_nouns.append(keyword)
    # "공지", "사항", "공지사항"을 query_nouns에서 '공지사항'이라고 고정하고 나머지 부분 삭제
    keywords=['공지','사항','공지사항']
    if any(keyword in content for keyword in keywords):
      # 키워드 제거
      for keyword in keywords:
          content = content.replace(keyword, '')
          query_nouns.append('공지사항')

    keywords=['사원','신입사원']
    if any(keyword in content for keyword in keywords):
        for keyword in keywords:
          content = content.replace(keyword, '')
          query_nouns.append('신입')
    # 5. Okt 형태소 분석기를 이용한 추가 명사 추출
    okt = Okt()
    additional_nouns = [noun for noun in okt.nouns(content) if len(noun) > 1]
    query_nouns += additional_nouns
    if '인도' not in query_nouns and  '인턴십' in query_nouns:
        query_nouns.append('베트남')

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

###################################################################################################


# Dense Retrieval (Upstage 임베딩)
embeddings = UpstageEmbeddings(
  api_key=upstage_api_key,
  model="solar-embedding-1-large"
) # Upstage API 키 사용
# dense_doc_vectors = np.array(embeddings.embed_documents(texts))  # 문서 임베딩


def fetch_titles_from_pinecone():
    # 메타데이터 기반 검색을 위한 임의 쿼리
    query_results = index.query(
        vector=[0] * 4096,  # Pinecone에서 사용 중인 벡터 크기에 맞게 0으로 채운 벡터
        top_k=1250,         # 최대 1000개의 결과 가져오기
        include_metadata=True  # 메타데이터 포함
    )

    # 메타데이터에서 필요한 값들 추출.
    titles = [match["metadata"]["title"] for match in query_results["matches"]]
    texts = [match["metadata"]["text"] for match in query_results["matches"]]
    urls = [match["metadata"]["url"] for match in query_results["matches"]]
    dates = [match["metadata"]["date"] for match in query_results["matches"]]

    return titles, texts, urls, dates



                    #################################   24.11.16기준 정확도 측정완료 #####################################################
######################################################################################################################

# 날짜를 파싱하는 함수
def parse_date_change_korea_time(date_str):
    clean_date_str = date_str.replace("작성일", "").strip()
    naive_date = datetime.strptime(clean_date_str, "%y-%m-%d %H:%M")
    # 한국 시간대 추가
    korea_timezone = pytz.timezone('Asia/Seoul')
    return korea_timezone.localize(naive_date)


def calculate_weight_by_days_difference(post_date, current_date, query_nouns):

    # 날짜 차이 계산 (일 단위)
    days_diff = (current_date - post_date).days

    # 기준 날짜 (24-01-01 00:00) 설정
    baseline_date_str = "24-01-01 00:00"
    baseline_date = parse_date_change_korea_time(baseline_date_str)

    # 작성일이 기준 날짜 이전이면 가중치를 1.35로 고정
    if post_date <= baseline_date:
        return 1.35

    # '최근', '최신' 등의 키워드가 있는 경우, 최근 가중치를 추가
    add_recent_weight = 1.0 if any(keyword in query_nouns for keyword in ['최근', '최신', '지금', '현재']) else 0

    # **10일 단위 구분**: 최근 문서에 대한 세밀한 가중치 부여
    if days_diff<=5:
        return 1.35 + add_recent_weight
    elif days_diff <= 10:
        return 1.33 + add_recent_weight/1.3
    elif days_diff <= 15:
        return 1.31 + add_recent_weight/1.9
    elif days_diff <= 20:
        return 1.29 + add_recent_weight/2.6
    elif days_diff <= 25:
        return 1.27 + add_recent_weight/3.1
    elif days_diff <= 30:
        return 1.25 + add_recent_weight/3.5
    elif days_diff<=  45:
        return 1.23+ add_recent_weight/3.8
    elif days_diff <= 60:
        return 1.21 + add_recent_weight/4.0
    elif days_diff <= 90:
        return 1.20
    # **월 단위 구분**: 2개월 이후는 월 단위로 단순화
    month_diff = (days_diff - 90) // 30
    month_weight_map = {
        0: 1.17 ,  # 2.5~3.5개월
        1: 1.16 - add_recent_weight / 3,  # 3.5~4.5개월
        2: 1.15 - add_recent_weight / 4,  # 4.5~5.5개월
        3: 1.12 - add_recent_weight / 5,  # 5.5~6.5개월
        4: 1.11 - add_recent_weight/6,
    }

    # 기본 가중치 반환 (6개월 이후)
    return month_weight_map.get(month_diff, 1-add_recent_weight/5)


# 유사도를 조정하는 함수
def adjust_date_similarity(similarity, date_str,query_nouns):
    # 현재 한국 시간
    current_time = get_korean_time()
    # 작성일 파싱
    post_date = parse_date_change_korea_time(date_str)
    # 가중치 계산
    weight = calculate_weight_by_days_difference(post_date, current_time,query_nouns)
    # 조정된 유사도 반환
    return similarity * weight

# 사용자 질문에서 추출한 명사와 각 문서 제목에 대한 유사도를 조정하는 함수
def adjust_similarity_scores(query_noun, title,texts,similarities):

    for idx, titl in enumerate(title):
        # 제목에 포함된 query_noun 요소의 개수를 센다

        matching_noun = [noun for noun in query_noun if noun in titl]
        if texts[idx] == "No content":
            if "국가장학금" in titl and "국가장학금" in query_noun:
              similarities[idx]*=5.0
            else:
              similarities[idx] *=1.5 # 본문이 "No content"인 경우 유사도를 높임
        for noun in matching_noun:
            similarities[idx] += len(noun)*0.21
            if re.search(r'\d', noun):  # 숫자가 포함된 단어 확인
                if noun in title:  # 본문에도 숫자 포함 단어가 있는 경우 추가 조정
                    similarities[idx] += len(noun)*0.22
                else:
                    similarities[idx]+=len(noun)*0.19
        # query_noun에 "대학원"이 없고 제목에 "대학원"이 포함된 경우 유사도를 0.1 감소
        keywords = ['대학원', '대학원생']
        # 조건 1: 둘 다 키워드 포함
        if any(keyword in query_noun for keyword in keywords) and any(keyword in titl for keyword in keywords):
            similarities[idx] += 2.0
        # 조건 2: query_noun에 없고, title에만 키워드가 포함된 경우
        if not any(keyword in query_noun for keyword in keywords) and any(keyword in titl for keyword in keywords):
            similarities[idx] -= 2.0
        if not any(keyword in query_noun for keyword in["현장", "실습", "현장실습"]) and any(keyword in titl for keyword in ["현장실습","대체","기준"]):
            similarities[idx]-=2
        if "외국인" not in query_noun and "외국인" in title:
            similarities[idx]-=2.0
        if texts[idx] == "No content":
            similarities[idx] *=1.45# 본문이 "No content"인 경우 유사도를 높임
        if '마일리지' in query_noun and '마일리지' in title:
            similarities[idx]+=1
        if '신입생' in query_noun and '수강신청' in query_noun and '일괄수강신청' in title:
            similarities[idx]+=4.0
    return similarities


#############################################################################################

def last_filter_keyword(DOCS,query_noun,user_question):
        # 필터링에 사용할 키워드 리스트
        Final_best=DOCS
        # 키워드가 포함된 경우 유사도를 조정하고, 유사도 기준으로 내림차순 정렬
        for idx, doc in enumerate(DOCS):
            score, title, date, text, url = doc
            if not any(keyword in query_noun for keyword in["현장", "실습", "현장실습"]) and any(keyword in title for keyword in ["현장실습","대체","기준"]):
              score-=1.0
            # wr_id 뒤에 오는 숫자 추출
            target_numbers = [27510, 27047, 27614, 27246, 25900, 27553, 25896, 28183,27807,25817]

            match = re.search(r"wr_id=(\d+)", url)
            if match:
                extracted_number = int(match.group(1))
                # 숫자가 target_numbers에 포함되면 score 증가
                if extracted_number in target_numbers:
                    if any(keyword in query_noun for keyword in ['에이빅','ABEEK']) and any(keyword in text for keyword in ['에이빅','ABEEK']):
                        if extracted_number==27047:
                           score+=0.3
                        else:
                           score+=1.5
                    else:
                        score+=0.8

            if '설계' in title:
                score-=0.4
            if '오픈소스' in query_noun and '오픈소스' in title:
                score+=0.5
            if 'SDG' in query_noun and 'SDG' in title:
                score+=2.9
            if any(keyword in query_noun for keyword in ['인턴','인턴십'])  and any(keyword in query_noun for keyword in ['인도','베트남']):
                score+=1.0
            if any(keyword in title for keyword in ['수요','조사']) and not any(keyword in query_noun for keyword in ['수요','조사']):
                score-=0.6
            if '여름' in query_noun and any(keyword in title for keyword in['겨울',"동계"]):
                score-=1.0
            if '겨울' in query_noun and any(keyword in title for keyword in['하계',"여름"]):
                score-=1.0
            if '여름' in query_noun and any(keyword in title for keyword in['하계',"여름"]):
                score+=1.5
            if '겨울' in query_noun and any(keyword in title for keyword in['겨울',"동계"]):
                score+=1.5
            if '변경' in query_noun and '변경' in title:
                score+=1.0
            if '1학기' in query_noun and '1학기' in title:
                score+=1.0
            if '2학기' in query_noun and '2학기' in title:
                score+=1.0
            if '1학기' in query_noun and '2학기' in title:
                score-=1.0
            if '2학기' in query_noun and '1학기' in title:
                score-=1.0
            if any(keyword in text for keyword in ['종프','종합설계프로젝트']) and any(keyword in user_question for keyword in ['종프','종합설계프로젝트']):
                score+=0.7
                if '설명회' in query_noun and '설명회' in title:
                  score+=0.7
                else:
                  score-=1.0
            if any(keyword in query_noun for keyword in ['복전','복수','복수전공']) and  any(keyword in title for keyword in ['복수']):
                score+=0.7
            if not any(keyword in query_noun for keyword in ['복전','복수','복수전공']) and any(keyword in title for keyword in ['복수']):
                score-=0.7
            if any(keyword in title for keyword in ['심컴','심화컴퓨터전공','심화 컴퓨터공학','심화컴퓨터공학']):
              if any(keyword in user_question for keyword in['심컴','심화컴퓨터전공']):
                score+=0.7
              else:
                if not "컴퓨터비전" in query_noun:
                  score-=0.7
            elif any(keyword in title for keyword in ['글로벌소프트웨어전공','글로벌SW전공','글로벌소프트웨어융합전공','글솝','글솦']):
              if any(keyword in user_question for keyword in ['글로벌소프트웨어융합전공','글로벌소프트웨어전공','글로벌SW전공','글솝','글솦']):
                score+=0.7
              else:
                score-=0.8
            elif any(keyword in title for keyword in['인컴','인공지능컴퓨팅']):
              if any(keyword in user_question for keyword in ['인컴','인공지능컴퓨팅']):
                score+=0.9
              else:
                score-=0.8
            if any(keyword in user_question for keyword in ['벤처','아카데미']) and any(keyword in title for keyword in ['벤처아카데미','벤처스타트업아카데미','벤처스타트업']):
                if any(keyword in user_question for keyword in ['스타트업']) and any(keyword in title for keyword in ['스타트업']):
                  score+=0.5
                elif not any(keyword in user_question for keyword in ['스타트업']) and any(keyword in title for keyword in ['벤처스타트업아카데미','벤처스타트업아카데미','스타트업','스타트','벤처스타트업']):
                  score-=2.5
                else:
                  score+=2.0
            if any(keyword in text for keyword in ['계약학과', '대학원', '타대학원']) and not any(keyword in query_noun for keyword in ['계약학과', '대학원', '타대학원']):
                score -= 0.4  # 유사도 점수를 0.4 낮추기
            keywords = ['대학원', '대학원생']

            # 조건 1: 둘 다 키워드 포함
            if any(keyword in query_noun for keyword in keywords) and any(keyword in title for keyword in keywords):
                score += 2.0
            # 조건 2: query_noun에 없고, title에만 키워드가 포함된 경우
            elif not any(keyword in query_noun for keyword in keywords) and any(keyword in title for keyword in keywords):
                score -= 2.0
            if any(keyword in query_noun for keyword in ['대학원','대학원생']) and any (keyword in title for keyword in ['대학원','대학원생']):
                score+=2.0

            if any(keyword in user_question for keyword in ['담당','업무','일','근무','관련']) and any(keyword in query_noun for keyword in ['직원','선생','선생님']):
                if url!= "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub2_5&lang=kor":
                    score-=3.0
                else:
                    score+=1.0
                    # IT와 E 모두 처리
                    for keyword in ['IT', 'E']:
                        if keyword in query_noun:
                            # 'IT'의 경우 숫자 4, 5 / 'E'의 경우 숫자 9 확인
                            valid_numbers = ['4', '5'] if keyword == 'IT' else ['9']
                            building_number = [num for num in query_noun if num in valid_numbers]
                            if building_number:
                                # IT4, IT5, E9 형식으로 결합
                                combined_building = f"{keyword}{building_number[0]}"
                                # 텍스트에 해당 건물 정보가 있는지 확인
                                if combined_building in text:
                                    score += 0.5  # 정확히 매칭된 경우 가중치 부여
                                else:
                                    score -= 0.8  # 매칭 실패 시 패널티
                    if '대학원' in query_noun:
                      if not any(keyword in query_noun for keyword in ['지원','계약']) and any(keyword in text for keyword in ['지원','계약']):
                        score-=0.8
                      else:
                        score+=0.5



            if (any(keyword in query_noun for keyword in ['담당','업무','일','근무']) or any(keyword in query_noun for keyword in ['직원','교수','선생','선생님'])) and date=="작성일24-01-01 00:00":
                ### 종프 팀과제 담당 교수 누구야와 같은 질문인데 엉뚱하게 파인콘에서 직원이 유사도 높게 측정된 경우를 방지하기 위함.
                if (any(keys in query_noun for keys in ['교수'])):
                  check=0
                  compare_url="https://cse.knu.ac.kr/bbs/board.php?bo_table=sub2_5&lang=kor" ## 직원에 해당하는 URL임.
                  if compare_url==url:
                    check=1
                  if check==0:
                    score+=0.5
                  else:
                    score-=0.9 ###직원이니까 유사도 나가라..
                else:
                  score+=4.0

            if not any(keys in query_noun for keys in['교수']) and any(keys in title for keys in ['담당교수','교수']):
              score-=0.7

            match = re.search(r"(?<![\[\(])\b수강\w*\b(?![\]\)])", title)
            if match:
                full_keyword = match.group(0)
                # query_nouns에 포함 여부 확인
                if full_keyword not in query_noun:
                  match = re.search(r"wr_id=(\d+)", url)
                  if match:
                      extracted_number = int(match.group(1))
                      if extracted_number in target_numbers:
                          score-=0.2
                      else:
                          score-=0.7
                else:
                  score+=0.8
            # 조정된 유사도 점수를 사용하여 다시 리스트에 저장
            Final_best[idx] = (score, title, date, text,  url)
            #print(Final_best[idx])
        return Final_best

#################################################################################################

def best_docs(user_question):
      okt = Okt()
      remove_noticement = ['제일','가장','공고', '공지사항','필독','첨부파일','수업','컴학','상위','관련']
      query_noun = transformed_query(user_question)  # 명사화된 user query

      titles_from_pinecone, texts_from_pinecone, urls_from_pinecone, dates_from_pinecone = fetch_titles_from_pinecone()
      tokenized_titles = [transformed_query(title) for title in titles_from_pinecone]

      # 기존과 동일한 파라미터를 사용하고 있는지 확인
      bm25_titles = BM25Okapi(tokenized_titles, k1=1.5, b=0.75)  # 기존 파라미터 확인

      title_question_similarities = bm25_titles.get_scores(query_noun)  # 제목과 사용자 질문 간의 유사도
      title_question_similarities /= 24


      adjusted_similarities = adjust_similarity_scores(query_noun, titles_from_pinecone,texts_from_pinecone, title_question_similarities)
      # 유사도 기준 상위 15개 문서 선택
      top_20_titles_idx = np.argsort(title_question_similarities)[-25:][::-1]

       # 결과 출력
      # print("최종 정렬된 BM25 문서:")
      # for idx in top_20_titles_idx:  # top_20_titles_idx에서 각 인덱스를 가져옴
      #     print(f"  제목: {titles[idx]}")
      #     print(f"  유사도: {title_question_similarities[idx]}")
      #     print(f" URL: {doc_urls[idx]}")
      #     print("-" * 50)

      Bm25_best_docs = [(titles_from_pinecone[i], dates_from_pinecone[i], texts_from_pinecone[i], urls_from_pinecone[i]) for i in top_20_titles_idx]

      ####################################################################################################

      # 1. Dense Retrieval - Text 임베딩 기반 20개 문서 추출
      query_dense_vector = np.array(embeddings.embed_query(user_question))  # 사용자 질문 임베딩

      # Pinecone에서 텍스트에 대한 가장 유사한 벡터 20개 추출
      pinecone_results_text = index.query(vector=query_dense_vector.tolist(), top_k=25, include_values=False, include_metadata=True)
      pinecone_similarities_text = [res['score'] for res in pinecone_results_text['matches']]
      pinecone_docs_text = [(res['metadata'].get('title', 'No Title'),
                            res['metadata'].get('date', 'No Date'),
                            res['metadata'].get('text', ''),
                            res['metadata'].get('url', 'No URL')) for res in pinecone_results_text['matches']]

     


      #####파인콘으로 구한  문서 추출 방식 결합하기.
      combine_dense_docs = []

      # 1. 본문 기반 문서를 combine_dense_docs에 먼저 추가
      for idx, text_doc in enumerate(pinecone_docs_text):
          text_similarity = pinecone_similarities_text[idx]*3.26
          text_similarity=adjust_date_similarity(text_similarity,text_doc[1],query_noun)
          matching_noun = [noun for noun in query_noun if noun in text_doc[2]]

          # # 본문에 포함된 명사 수 기반으로 유사도 조정
          for noun in matching_noun:
              text_similarity += len(noun)*0.20
              if re.search(r'\d', noun):  # 숫자가 포함된 단어 확인
                  if noun in text_doc[2]:  # 본문에도 숫자 포함 단어가 있는 경우 추가 조정
                      text_similarity += len(noun)*0.24
                  else:
                      text_similarity+=len(noun)*0.20
          combine_dense_docs.append((text_similarity, text_doc))

      ####query_noun에 포함된 키워드로 유사도를 보정
      # 유사도 기준으로 내림차순 정렬
      combine_dense_docs.sort(key=lambda x: x[0], reverse=True)

      # ## 결과 출력
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
        
      combine_dense_doc=last_filter_keyword(combine_dense_doc,query_noun,user_question)
      # Step 2: combine_dense_docs와 BM25 결과 합치기
      final_best_docs = []

      # combine_dense_docs와 BM25 결과를 합쳐서 처리
      for score, title, text, date, url in combine_dense_doc:
          matched = False
          for bm25_doc in Bm25_best_docs:
              if bm25_doc[0] == title:  # 제목이 일치하면 유사도를 합산
                  combined_similarity = score + adjusted_similarities[titles_from_pinecone.index(bm25_doc[0])]
                  final_best_docs.append((combined_similarity, bm25_doc[0], bm25_doc[1], bm25_doc[2], bm25_doc[3]))
                  matched = True
                  break
          if not matched:

              # 제목이 일치하지 않으면 combine_dense_docs에서만 유사도 사용
              final_best_docs.append((score,title, date, text, url))


      # 제목이 일치하지 않는 BM25 문서도 추가
      for bm25_doc in Bm25_best_docs:
          matched = False
          for score, title, text, date, url in combine_dense_doc:
              if bm25_doc[0] == title and bm25_doc[2]==text:  # 제목이 일치하면 matched = True로 처리됨
                  matched = True
                  break
          if not matched:
              # 제목이 일치하지 않으면 BM25 문서만 final_best_docs에 추가
              combined_similarity = adjusted_similarities[titles_from_pinecone.index(bm25_doc[0])]  # BM25 유사도 가져오기
              combined_similarity= adjust_date_similarity(combined_similarity,bm25_doc[1],query_noun)
              final_best_docs.append((combined_similarity, bm25_doc[0], bm25_doc[1], bm25_doc[2], bm25_doc[3]))
      final_best_docs.sort(key=lambda x: x[0], reverse=True)
      final_best_docs=final_best_docs[:20]


      # print("\n\n\n\n필터링 전 최종문서 (유사도 큰 순):")
      # for idx, (scor, titl, dat, tex, ur, image_ur) in enumerate(final_best_docs):
      #     print(f"순위 {idx+1}: 제목: {titl}, 유사도: {scor},본문 {len(tex)} 날짜: {dat}, URL: {ur}")
      #     print("-" * 50)
      
      final_best_docs=last_filter_keyword(final_best_docs,query_noun,user_question)
      final_best_docs.sort(key=lambda x: x[0], reverse=True)

      # print("\n\n\n\n중간필터 최종문서 (유사도 큰 순):")
      # for idx, (scor, titl, dat, tex, ur, image_ur) in enumerate(final_best_docs):
      #     print(f"순위 {idx+1}: 제목: {titl}, 유사도: {scor},본문 {len(tex)} 날짜: {dat}, URL: {ur}")
      #     print("-" * 50)

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
                      cluster_date=parse_date_change_korea_time(cluster[0][2])
                      doc_in_date=parse_date_change_korea_time(doc[2])
                      compare_date=abs(cluster_date-doc_in_date).days
                      if cluster_title==title or(-doc[0]+cluster[0][0]<0.6 and cluster[0][3]!=doc[2] and compare_date<60):
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
      # print(clusters[0])
      # print(clusters[1])
      # 날짜 형식을 datetime 객체로 변환하는 함수
      def parse_date(date_str):
          # '작성일'을 제거하고 공백을 제거한 뒤 날짜 형식으로 변환
          clean_date_str = date_str.replace("작성일", "").strip()
          return datetime.strptime(clean_date_str, "%y-%m-%d %H:%M")
      # Step 3: Compare cluster[0] cluster[1] top similarity and check condition
      top_0_cluster_similar=clusters[0][0][0]
      top_1_cluster_similar=clusters[1][0][0]
      keywords = ["최근", "최신", "현재", "지금"]
      #print(f"{top_0_cluster_similar} {top_1_cluster_similar}")
      if (top_0_cluster_similar-top_1_cluster_similar<=0.3): ## 질문이 모호했다는 의미일 수 있음.. (예를 들면 수강신청 언제야? 인데 구체적으로 1학기인지, 2학기인지, 겨울, 여름인지 모르게..)
          # 날짜를 비교해 더 최근 날짜를 가진 클러스터 선택
          #조금더 세밀하게 들어가자면?
          #print("세밀하게..")
          if (any(keyword in word for word in query_nouns for keyword in keywords) or top_0_cluster_similar-clusters[len(clusters)-1][0][0]<=0.3):
            #print("최근이거나 뽑은 문서들이 유사도 0.3이내")
            if (top_0_cluster_similar-clusters[len(clusters)-1][0][0]<=0.3):
              #print("최근이면서 뽑은 문서들이 유사도 0.3이내 real")
              sorted_cluster=sorted(clusters, key=lambda doc: doc[0][2], reverse=True)
              sorted_cluster=sorted_cluster[0]
            else:
              #print("최근이면서 뽑은 문서들이 유사도 0.3이상")
              if (top_0_cluster_similar-top_1_cluster_similar<=0.3):
                #print("최근이면서 뽑은 두문서의 유사도 0.3이하이라서 두 문서로 줄임")
                date1 = parse_date_change_korea_time(clusters[0][0][2])
                date2 = parse_date_change_korea_time(clusters[1][0][2])
                result_date=(date1-date2).days
                if result_date<0:
                  result_docs=clusters[1]
                else:
                  result_docs=clusters[0]
                sorted_cluster = sorted(result_docs, key=lambda doc: doc[2], reverse=True)

              else:
                sorted_cluster=sorted(clusters, key=lambda doc: doc[0][0], reverse=True)
                sorted_cluster=sorted_cluster[0]
          else:
           # print("두 클러스터로 판단해보자..")
            if (top_0_cluster_similar-top_1_cluster_similar<=0.1):
             # print("진짜 차이가 없는듯..?")
              date1 =parse_date_change_korea_time(clusters[0][0][2])
              date2 = parse_date_change_korea_time(clusters[1][0][2])
              result_date=(date1-date2).days
              if result_date<0:
                #print("두번째 클러스터가 더 크네..?")
                result_docs=clusters[1]
              else:
                #print("첫번째 클러스터가 더 크네..?")
                result_docs=clusters[0]
              sorted_cluster = sorted(result_docs, key=lambda doc: doc[2], reverse=True)
            else:
              #print("에이 그래도 유사도 차이가 있긴하네..")
              result_docs=clusters[0]
              sorted_cluster=sorted(result_docs,key=lambda doc: doc[0],reverse=True)
      else: #질문이 모호하지 않을 가능성 업업
          number_pattern = r"\d"
          period_word=["여름","겨울"]
          if (any(keyword in word for word in query_noun for keyword in keywords) or not any(re.search(number_pattern, word) for word in query_noun) or not any(key in word for word in query_noun for key in period_word)):
              #print("최근 최신이라는 말 드가거나 2가지 모호한 판단 기준")
              if (any(re.search(number_pattern, word) for word in query_noun) or any(key in word for word in query_noun for key in period_word)):
                #print("최신인줄 알았지만 유사도순..")
                result_docs=clusters[0]
                num=0
                for doc in result_docs:
                  if re.search(r'\d+차', doc[1]):
                    num+=1
                if num>1:
                  sorted_cluster=sorted(result_docs,key=lambda doc:doc[2],reverse=True)
                else:
                  sorted_cluster=sorted(result_docs,key=lambda doc:doc[0],reverse=True)
              else:
                #print("너는 그냥 최신순이 맞는거여..")
                result_docs=clusters[0]
                sorted_cluster=sorted(result_docs,key=lambda doc: doc[2],reverse=True)
          else:
            #print("진짜 유사도순대로")
            result_docs=clusters[0]
            sorted_cluster = sorted(clusters[0], key=lambda doc: doc[0], reverse=True)

      # print("\n\n\n\nadd_similar넣기전 상위 문서 (유사도 및 날짜 기준 정렬):")
      # for idx, (scor, titl, dat, tex, ur, image_ur) in enumerate(sorted_cluster):
      #     print(f"순위 {idx+1}: 제목: {titl}, 유사도: {scor}, 날짜: {dat}, URL: {ur} 내용: {len(tex)}   이미지{len(image_ur)}")
      #     print("-" * 50)
      # print("\n\n\n")

      def organize_documents_v2(sorted_cluster, titles, doc_dates, texts, doc_urls):
          # 첫 번째 문서를 기준으로 초기 설정
          top_doc = sorted_cluster[0]
          top_title = top_doc[1]

          # new_sorted_cluster 초기화 및 첫 번째 문서와 동일한 제목을 가진 문서들을 모두 추가
          new_sorted_cluster = []
          # titles에서 top_title과 같은 제목을 가진 모든 문서를 new_sorted_cluster에 추가
          count=0
          for i, title in enumerate(titles):
              if title == top_title:
                  new_similar=top_doc[0]
                  count+=1
                  new_doc = (top_doc[0], titles[i], doc_dates[i], texts[i], doc_urls[i])
                  new_sorted_cluster.append(new_doc)
          for i in range(count-1):
            fix_similar=list(new_sorted_cluster[i])
            fix_similar[0]=fix_similar[0]+0.2*count
            new_sorted_cluster[i]=tuple(fix_similar)
          # sorted_cluster에서 new_sorted_cluster에 없는 제목만 추가
          for doc in sorted_cluster:
              doc_title = doc[1]
              # 이미 new_sorted_cluster에 추가된 제목은 제외
              if doc_title != top_title:
                  new_sorted_cluster.append(doc)

          return new_sorted_cluster,count

      # 예시 사용
      final_cluster,count = organize_documents_v2(sorted_cluster, titles_from_pinecone, dates_from_pinecone, texts_from_pinecone, urls_from_pinecone)
      return final_cluster[:count], query_noun

prompt_template = """당신은 경북대학교 컴퓨터학부 공지사항을 전달하는 직원이고, 사용자의 질문에 대해 올바른 공지사항의 내용을 참조하여 정확하게 전달해야 할 의무가 있습니다.
현재 한국 시간: {current_time}

주어진 컨텍스트를 기반으로 다음 질문에 답변해주세요:

{context}

질문: {question}

답변 시 다음 사항을 고려해주세요:

1. 질문의 내용이 이벤트의 기간에 대한 것일 경우, 문서에 주어진 기한과 현재 한국 시간을 비교하여 해당 이벤트가 예정된 것인지, 진행 중인지, 또는 이미 종료되었는지에 대한 정보를 알려주세요.
  예를 들어, "2학기 수강신청 일정은 언제야?"라는 질문을 받았을 경우, 현재 시간은 11월이라고 가정하면 수강신청은 기간은 8월이었으므로 이미 종료된 이벤트입니다.
  따라서, "2학기 수강신청은 이미 종료되었습니다."와 같은 문구를 추가로 사용자에게 제공해주고, 2학기 수강신청 일정에 대한 정보를 사용자에게 제공해주어야 합니다.
  또 다른 예시로 현재 시간이 11월 12일이라고 가정하였을 때, "겨울 계절 신청기간은 언제야?"라는 질문을 받았고, 겨울 계절 신청기간이 11월 13일이라면 아직 시작되지 않은 이벤트입니다.
  따라서, "겨울 계절 신청은 아직 시작 전입니다."와 같은 문구를 추가로 사용자에게 제공해주고, 겨울 계절 신청 일정에 대한 정보를 사용자에게 제공해주어야 합니다.
  또 다른 예시로 현재 시간이 11월 13일이라고 가정하였을 때, "겨울 계절 신청기간은 언제야?"라는 질문을 받았고, 겨울 계절 신청기간이 11월 13일이라면 현재 진행 중인 이벤트입니다.
  따라서, "현재 겨울 계절 신청기간입니다."와 같은 문구를 추가로 사용자에게 제공해주고, 겨울 계절 신청 일정에 대한 정보를 사용자에게 제공해주어야 합니다.
2. 질문에서 핵심적인 키워드들을 골라 키워드들과 관련된 문서를 찾아서 해당 문서를 읽고 정확한 내용을 답변해주세요.
3. 문서의 내용을 그대로 길게 전달하기보다는 질문에서 요구하는 내용에 해당하는 답변만을 제공함으로써 최대한 답변을 간결하고 일관된 방식으로 제공하세요.
4. 에이빅과 관련된 질문이 들어오면 임의로 판단해서 네 아니오 하지 말고 문서에 있는 내용을 그대로 알려주세요.
5. 답변은 친절하게 존댓말로 제공하세요.
6. 질문이 공지사항의 내용과 전혀 관련이 없다고 판단하면 응답하지 말아주세요. 예를 들면 "너는 무엇을 알까", "점심메뉴 추천"과 같이 일반 상식을 요구하는 질문은 거절해주세요.

답변:"""

# PromptTemplate 객체 생성
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["current_time", "context", "question"]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_answer_from_chain(best_docs, user_question,query_noun):

    documents = []
    doc_titles = []
    doc_dates = []
    doc_texts = []
    doc_urls = []
    for doc in best_docs:
        tit = doc[1]
        date = doc[2]
        text = doc[3]
        url = doc[4]
        # score,tit, date, text, url,im_url = doc
        doc_titles.append(tit)  # 제목
        doc_dates.append(date)    # 날짜
        doc_texts.append(text)    # 본문
        doc_urls.append(url)     # URL

    documents = [
        Document(page_content=text, metadata={"title": title, "url": url, "doc_date": datetime.strptime(date, '작성일%y-%m-%d %H:%M')})
        for title, text, url, date in zip(doc_titles, doc_texts, doc_urls, doc_dates)
    ]
    # 키워드 기반 관련성 필터링 추가 (질문과 관련 없는 문서 제거)
    # 사용자 질문을 전처리하여 공백 제거 후 명사만 추출
    relevant_docs = [doc for doc in documents if any(keyword in doc.page_content for keyword in query_noun)]
    if not relevant_docs:
      return None, None

    llm = ChatUpstage(api_key=upstage_api_key)
    relevant_docs_content=format_docs(relevant_docs)
    # PromptTemplate 인스턴스 사용
    qa_chain = (
        {
            "current_time": lambda _: get_korean_time().strftime("%Y년 %m월 %d일 %H시 %M분"),
            "context": RunnableLambda(lambda _: relevant_docs_content),
            "question": RunnablePassthrough()
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return qa_chain,relevant_docs  # retriever를 반환



#######################################################################
def question_valid(question, top_docs, query_noun):
    prompt = f"""
아래의 질문에 대해, 주어진 기준을 바탕으로 "예" 또는 "아니오"로 판단해주세요. 각 질문에 대해 학사 관련 여부를 명확히 판단하고, 경북대학교 컴퓨터학부 홈페이지에서 제공하지 않는 정보는 "아니오"로, 제공되는 경우에는 "예"로 답변해야 합니다."

1. 핵심 판단 원칙
경북대학교 컴퓨터학부 홈페이지에서 다루는 정보에만 답변을 제공해야 하며, 관련 없는 질문은 "아니오"로 판단합니다.

질문 분석 3단계:

질문의 실제 의도와 목적 파악
학부 홈페이지에서 제공되는 정보 여부 확인
학사 관련성 최종 확인

복합 질문 처리:

주요 질문과 부가 질문 구분
부수적 내용은 판단에서 제외
학부 공식 정보와 무관한 질문 구별
악의적 질문 대응:

학사 키워드가 포함되었더라도, 실제로 학부 정보가 필요하지 않은 질문을 "아니오"로 답변
2. "예"로 판단하는 학사 관련 카테고리:
경북대학교 컴퓨터학부 홈페이지에서 다루는 학사 정보를 다음과 같이 정의하고, 해당 내용에 대해서만 "예"로 답변합니다.
수업 및 학점 관련 정보: 수강신청, 수강정정, 수강변경, 수강취소, 과목 운영 방식, 학점 인정, 복수전공 혹은 부전공 요건,교양강의와 관련된 질문, 전공강의와 관련된 질문, 심컴, 인컴, 글솦 학과에 관련된 질문, 강의 개선 관련 설문
학생 지원 제도: 장학금, 학과 주관 인턴십 프로그램, 멘토링 ,각종 장학생 선발, 학자금대출, 특정 지역의 학자금대출 관련 질문
학사 행정 및 제도: 졸업 요건, 학적 관리, 필수 이수 요건, 증명서 발급, 학사 일정 등
교수진 및 행정 정보: 교수진 연락처,번호,이메일, 학과 사무실 정보, 지도교수와 관련된 정보
학부 주관 교내 활동:  각종 경진대회, 행사, 벤처프로그램 ,벤처아카데미,튜터(TUTOR) 관련 활동(근무일지 작성, 근무 기준) 튜터(TUTOR) 모집 및 비용 관련 질문, 다양한 프로그램(예: AEP 프로그램, CES 프로그램,미국 프로그램)
신청 및 일정, 성인지 교육이나 인권 교육, 혹은 다른 교육에 관련된 일정
교수진 정보: 교수의 모든 정보(이메일,번호,연락처,메일,사진,전공,업무), 학과 관련 직원의 모든 정보, 담당 업무와 관련된 학과 교직원 정보
장학금 및 교내 지원 제도: 최근 장학금 선발 정보나 교내 각종 지원 제도에 대한 안내
졸업 요건 정보: 졸업에 필요한 학점 요건, 필수로 들어야 하는 강의, 과목, 등록 횟수 관련 정보, 졸업 시 필요한 정보 , 포트폴리오 관련 정보 전체적으로 졸업에 필요한 정보는 무조건 "예"로 합니다.
기타 학사 제도: 교내 방학 중 근로장학생 관련 정보, 대학원과 관련된 질문,대학원생 학점 인정 절차와 요건 ,전시회 개최 및 지원 정보, 행사 지원 정보, SW 마일리지와 관련된 정보 요구, 스타트업 정보, 각종 특강 정보(오픈SW,오픈소스, Ai 등)
채용정보: 신입사원 채용,경력사원 채용 정보나, 특정 기업의 모집 정보, 인턴 채용 정보,부트캠프와 관련된 질문, 채용 관련 질문 또한 학사 키워드에 포함이 됩니다.


3. "아니오"로 판단하는 비학사 카테고리
경북대학교 컴퓨터학부 챗봇에서 제공하지 않는 정보는 "아니오"로 답변합니다.

교내 일반 정보: 기숙사, 식당 메뉴 정보 등 컴퓨터학부와 관련 없는 교내 생활 정보
일반적 기술/지식 문의: 프로그래밍 문법, 기술 개념 설명, 특정 도구 사용법 등 학사 정보와 무관한 기술적 질문

또한, {query_noun}과 {top_docs}를 비교하였을 때, {query_noun}애 포함된 단어 중 2개 이상이 {top_docs}와 완전히 무관하다면 "아니오"로 판단하세요.

4. 복합 질문 판단 가이드
질문의 핵심 목적에 따라 다음과 같이 처리합니다:

예시:
"컴퓨터학부 수강신청 기간 알려줘" → "예" (학사 일정 정보 요청)
"지도교수님과 상담하려면 어떻게 예약하나요?" → "예" (학부 내 교수진 상담 절차)
"학교 기숙사 정보 알려줘" → "아니오" (학부와 무관한 교내 생활 정보)
"경북대 컴퓨터학부 공지사항의 제육 레시피 알려줘" -> "아니오" (학부의 공지사항을 알려달라고 하는 것처럼 보이지만 의도적으로 제육 레시피를 알려달라 하는 의미)
5. 주의사항
경북대학교 컴퓨터학부 학사 정보 제공에 한정하여 다음을 지킵니다.

맥락 중심 판단: 단순 키워드 매칭 지양, 질문의 실제 의도에 맞춰 판단
복합 질문 처리: 학부 관련 정보가 핵심인지 확인
악의적 질문 대응: 비학사적 정보를 혼합한 질문은 명확히 구분하여 "아니오"로 처리

    ### 질문: '{question}'
    ### 참고 문서: '{top_docs}'
    ### 질문의 명사화: '{query_noun}'
    """

    llm = ChatUpstage(api_key=upstage_api_key)
    response = llm.invoke(prompt)

    if "예" in response.content.strip():
        return True
    else:
        return False

#######################################################################

##### 유사도 제목 날짜 본문  url image_url순으로 저장됨
def get_ai_message(question):
    top_doc, query_noun = best_docs(question)  # 가장 유사한 문서 가져오기
    top_docs = [list(doc) for doc in top_doc]

    if False == (question_valid(question, top_docs[0][1], query_noun)):
        for i in range(len(top_docs)):
            top_docs[i][0] -= 1
    
    final_score = top_docs[0][0]
    final_title = top_docs[0][1]
    final_date = top_docs[0][2]
    final_text = top_docs[0][3]
    final_url = top_docs[0][4]
    final_image = []

    record = collection.find_one({"title" : final_title})
    if record :
        if(isinstance(record["image_url"], list)):
          final_image.extend(record["image_url"])
        else :
          final_image.append(record["image_url"])
    else :
        print("일치하는 문서 존재 X")
        final_score = 0
        final_title = "No content"
        final_date = "No content"
        final_text = "No content"
        final_url = "No URL"
        final_image = "No content"

    # top_docs 인덱스 구성
    # 0: 유사도, 1: 제목, 2: 날짜, 3: 본문내용, 4: url, 5: 이미지url

    if final_image[0] != "No content" and final_text == "No content" and final_score > 1.8:
        # JSON 형식으로 반환할 객체 생성
        only_image_response = {
            "answer": None,
            "references": final_url,
            "disclaimer": "항상 정확한 답변을 제공하지 못할 수 있습니다. 아래의 URL들을 참고하여 정확하고 자세한 정보를 확인하세요.",
            "images": final_image
        }
        return only_image_response

    # 이미지 + LLM 답변이 있는 경우.
    else:
        qa_chain, relevant_docs = get_answer_from_chain(top_docs, question,query_noun)


        # 공지사항에 존재하지 않을 경우
        notice_url = "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_1"
        not_in_notices_response = {
            "answer": "해당 질문은 공지사항에 없는 내용입니다.\n 자세한 사항은 공지사항을 살펴봐주세요.",
            "references": notice_url,
            "disclaimer": "항상 정확한 답변을 제공하지 못할 수 있습니다. 아래의 URL들을 참고하여 정확하고 자세한 정보를 확인하세요.",
            "images": "No Content"
        }

        # 답변 생성 실패
        if not qa_chain or not relevant_docs:
            if final_image[0] != "No content" and final_score > 1.8:
                data = {
                    "answer": "해당 질문에 대한 내용은 이미지 파일로 확인해주세요.",
                    "references": final_url,
                    "disclaimer": "항상 정확한 답변을 제공하지 못할 수 있습니다. 아래의 URL들을 참고하여 정확하고 자세한 정보를 확인하세요.",
                    "images": final_image
                }
                return data
            else:
                return not_in_notices_response

        # 유사도가 낮은 경우
        if final_score < 1.8:
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
            "images": final_image
        }

        return data
