# ai

### RAG 시스템 구현한 과정

### 1. 경북대학교 컴퓨터학부 공지사항(세미나 ,채용 정보 포함), 교수진 및 교직원 정보 크롤링

 공지사항에는 데이터가 제목, 작성자, 날짜,URL 그리고 본문 내용(이미지 URL 포함)으로 이루어져 있다. 이 데이터들 중 필요한 정보 - 제목 날짜 URL 본문 내용을 크롤링한다. 이 때, 제목 날짜 URL 본문 내용 중 없는 데이터는 "No content"를 삽입해 일관성을 데이터 수집에 있어 일관성을 유지한다.
수집한 데이터들 중 본문 내용이 가장 핵심이기 때문에 텍스트를 850자, 중복 100자로 나누어 저장한 다음, Pinecone Vector DB에 임베딩 저장하게 된다. 그리고 upstage의 "solar-embedding-1-large"을 사용해 본문 내용을 임베딩한다.

Q. Vector DB 선택할 때, FAISS 대신 Pinecone 선택한 이유가 있는가?
-> FAISS는 뛰어난 성능을 보여주지만 직접 인프라를 구축하고 관리해야 하는 부담이 있다. 반면 파인콘은 서버리스 방식으로 제공되어 인프라 구축과 관리에 대한 부담이 없고, 실시간 업데이트가 용이하며 확장성도 뛰어나서 선택하였다. 

### 2. 파인콘에서 데이터 불러오기 + RAG의 핵심 문서 검색하기

  파인콘에서 데이터를 불러올 때 Redis 캐시 메모리를 활용하였다. 그 이유는 초기에는 질문이 들어올 때마다 파인콘에서 데이터를 불러와 변수에 저장한 다음 문서검색을 수행했는데, 파인콘에서 불러와 저장하는데 소요되는 시간을 체크해본 결과, 0.5초에서 1초정도 걸린다는 것을 확인하고 
 응답속도를 개선하기 위해 처음 서버를 돌릴 때 Redis 캐시 메모리에 파인콘에서 불러온 데이터를 저장하여 질문이 들어올 때마다 데이터를 불러오는 작업을 제거하여 응답 속도를 개선하고자 도입하였다.
 RAG의 핵심 문서 검색은 sparse방식(키워드 중심)과 dense방식(문맥 중심)을 결합한 하이브리드 방식으로 문서 검색 시스템을 구현하였고, 문서 검색하는 알고리즘은 아래와 같다.
 1) transformed_query()함수를 통해서 Okt() (한국어에서 명사를 추출하는 라이브러리) 를 활용하여 질문을 명사화한다.
 2) BM25 라이브러리를 활용해 제목과 질문 간의 유사도를 측정해 상위 문서를 뽑는다. (키워드 중심 방식)
 3) 파인콘에서 질문과 본문 내용간의 문맥상 유사한 문서를 뽑는다. (문맥 중심 방식)
 4) 날짜와 명사 키워드에 따른 유사도 가중치를 조정한다.
 5) 뽑은 BM25 문서와 파인콘의 문서를 결합하여 상위 문서 20개를 뽑아 유사도를 기준으로 정렬한다. ----- 1차 flitering
 6) 상위 문서 20개를 사전에 정의한 키워드의 존재 유무에 따라 유사도를 재조정한 후, 유사도를 기준으로 정렬한다. -----2차 filtering
 7) 다음으로는 문서끼리의 군집화 (제목을 기준으로 군집화함) 과정을 통해 뽑은 문서들을 나눈다. 이 때 임계값은 직접 반복하여 테스트하는 과정을 통해 가장 최적의 값, 0.89로 설정하였다.
 8) 마지막으로 상위 1,2위 군집화 내에 첫 번째에 있는 문서 간의 비교를 통해 최종 문서를 뽑는다. -----3차 filtering
 9) 이 때 군집화 내에 있는 문서 모두 반환하는 것이 아닌 군집화된 문서들 중 가장 첫 번째에 위치한 문서와 동일한 제목을 가진 문서들을 모두 찾는다. (왜냐하면 크롤링 과정에서 850자로 나누었기 때문에 동일한 제목을 가진 문서가 여러 개 있기 때문에 이를 합치는 과정이다.)
 10) 최종 문서를 반환할 때 (score,title,date,url,text)의 값을 반환한다. (score는 유사도 값을 의미함.)


 Q. Tf-idf 대신 BM25를 선택한 이유는?
-> Tf-idf의 경우 단어 빈도가 증가할수록 가중치가 선형적 증가한다는 문제점이 있는데, BM25의 경우 단어 빈도에 대한 포화도를 고려해 가중치가 무한하게 커지지 않고 적절한 수준에서 포화된다. 또한 문서 길이 차이를 보정하는 기능이 있기 때문에 공지사항처럼 길이가 다양한 문서들의 유사도를 측정하는데 더 유리하다.

Q. sparse 방식과 dense 방식중 하나를 택하지 않고 두 방법을 결합한 Ensemble 방식을 택한 이유는?
그리고 어느 쪽에 더 가중치를 두었는가?

-> 질문에 따라 키워드 방식으로 충분히 뽑히는 경우가 있지만 복잡한 질문 예를 들면 계절학기가 아닌 일반학기에는 심컴 글솝 인컴이 아닌 전공을 들어도 에이빅으로 인정이 될까?와 같은 질문의 경우 의미론적 분석이 필요하기 떄문에 dense방식도 함께 사용하여 구현했다. 다만, 공지사항의 특성상 키워드가 핵심이 되는 경우가 대부분이므로 sparse 방식과 dense 방식의 가중치는 대략 7:3정도라고 볼 수 있을 것 같다.

Q. 명사 키워드를 굳이 추출하는 이유는?

-> 키워드 추출이 공지사항의 핵심이다. 제목과 본문 내용에 키워드가 무수히 많은데 질문에서 명사를 뽑아내 키워드 중심으로 검색을 수행해 효율성을 높일 수 있었다. 또한 조사나 어미 같은 불필요한 정보를 제거함으로써 검색의 정확도도 향상시킬 수 있었다.

Q. 명사 키워드를 추출한 이유는 이해가 갔다. 그런데 날짜 가중치를 도입한 이유는?

-> 도입하지 않아도 질문과 관련된 문서는 충분히 검색할 수 있다. 하지만 사용자는 기본적으로 과거의 정보보다는 최근의 정보를 원할 것이라고 생각했다. 그래서 수강과 관련된 질문을 예시로 들면 수강신청이 언제인지를 요청했을 때 8월달에 진행한 2학기 수강신청 안내보다는 11월에 진행한 겨울계절학기 수강신청 안내를 알려주는 것이 더 바람직하다고 생각하여 현재 날짜를 시점으로 일수가 차이날수록 가중치를 감소시켜 최근의 정보를 우선적으로 제공하고자 했다.

### 3. 검색된 문서를 바탕으로 답변 생성하기

 검색된 문서를 바탕으로 LLM 모델 Upstage를 활용하여 답변을 생성한다. 이 때, 생성하는 과정은 아래와 같다.
 1) LLM 모델 upstage에게 질문의 유효성을 체크한다. (질문이 학사정보제공과 관련된 내용이 맞는지를 LLM 모델이 True/False로 체크한다.) False를 반환한 경우, 유사도 값을 -3하여 답변 생성하지 못하도록 방지한다.
 2) 다음 프롬프트 템플릿을 작성한 후, Upstage가 이 프롬프트의 내용을 기반으로 답변을 생성하는데 이 때, Stuff 방식을 활용하여 답변한다.
 3) 생성된 답변은 유사도의 값에 따라 반환하는 내용을 달리하여 json 형식으로 반환한다.

Q. LLM중 Upstage모델을 선택한 이유는?

-> 한국어 성능이 다른 LLM 모델보다 우수한 편이고, 무료 크레딧을 10$ 제공하기 떄문에 프로젝트 기간동안 충분히 쓸 수 있으므로 비용 부담이 적어 선택하였다.

Q. Refine 대신 Stuff 방식을 선택한 이유는?

-> 프로젝트를 진행하면서 검색한 문서의 개수를 줄였고 길이가 긴 편이 아니기 때문에 Stuff 방식으로도 충분히 처리 가능하며 응답 시간을 빠르게 하기 위해 선택하였다.  Refine 방식은 문서를 순차적으로 처리하기 때문에 시간이 더 오래 걸리는데, 현 시스템이 응답 속도가 빠른 편은 아니므로  Refine 방식을 도입하면 마이너스 요인이 될 수 있다고 판단해 Stuff 방식을 선택하였다.

Q. 생성된 답변이 유사도의 값에 따라 반환하는 내용이 다르다는게 무슨 의미인가? 
-> 이 또한 테스트 과정을 거쳐 최적의 값, 1.8을 기준으로 유사도 값이 낮다면 공지사항에 없는 내용으로 간주하고 반환할 때 URL은 공지사항의 첫 페이지를 반환하고 공지사항에 없는 내용이라는 문구 텍스트를 함께 반환하게 된다.
만약 이보다 값이 크다면 생성된 답변을 해당 문서의 URL과 함께 반환하게 된다.

Q. 혹시 테스트할 때 질문지 같은 것들을 미리 작성해두고 테스트를 진행하였는가?
-> 테스트를 진행할 때 초기에는 50개의 질문지를 만들어 시작하였고, 12월 초에는 200개의 질문지 셋을 활용해 모든 질문에 올바른 문서가 뽑힐 때까지 테스트를 진행하였다.

#### 보완할 점

1. 이미지 텍스트 추출
-> 이미지를 그대로 보여주는 것도 좋지만 이미지 안의 내용을 추출하는 기능을 구현하지 못해 제목만으로 상위 문서로 뽑는 과정이 쉽지않았다. 이미지 텍스트 추출 기능을 구현했으면 좋았을 것 같다.
2. 관련없는 질문 필터링 기능 미흡
-> 공지사항과 관련 있는 키워드와 관련없는 내용을 Mix하여 질문을 하게 되면 질문에 따라 유사도 값이 높게 측정되어 원래는 공지사항에 없는 내용이라는 문구 텍스트를 함께 반환해야하는데 답변을 생성해 반환하는 경우가 있다.
3. 응답속도 느림
-> 직접 사용해본 사용자들이 가장 많이 언급한 문제라고 볼 수 있는데 Colab 환경에서 테스트 할 떄는 평균 4.4초 정도 소요되었지만, 서버에서의 응답시간을 체크해봤을 때는 길면 10초가 넘어가고, 6~8초 사이에서 답변이 생성된다.
이 부분에 대해서는 개선을 시도해보았으나, 여전히 보완이 필요하다고 생각한다.
