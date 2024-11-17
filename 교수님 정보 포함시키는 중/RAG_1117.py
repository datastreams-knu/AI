import requests
from bs4 import BeautifulSoup
from langchain_upstage import UpstageEmbeddings
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pinecone import Pinecone
from langchain_upstage import ChatUpstage
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document
from langchain.vectorstores import FAISS
import re
from datetime import datetime
import pytz
from konlpy.tag import Okt
import numpy as np
from IPython.display import display, HTML
import numpy as np
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt
from difflib import SequenceMatcher

pinecone_api_key = '423c8b35-3b1a-46fb-aebb-e20c9a36dba1'  
index_name = 'info'

upstage_api_key = 'up_yKqThHL17ZcjIGzeOxYkYCaTVqyLb' 

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)
def get_korean_time():
    return datetime.now(pytz.timezone('Asia/Seoul'))

def get_latest_wr_id():
    url = "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_1"
    response = requests.get(url)
    if response.status_code == 200:
        match = re.search(r'wr_id=(\d+)', response.text)
        if match:
            return int(match.group(1))
    return None

now_number = get_latest_wr_id()
urls = []
for number in range(now_number, 27726, -1):   
    urls.append("https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_1&wr_id=" + str(number))

def extract_text_and_date_from_url(urls):
    all_data = []

    def fetch_text_and_date(url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            title_element = soup.find('span', class_='bo_v_tit')
            title = title_element.get_text(strip=True) if title_element else "Unknown Title"

            text_content = "Unknown Content"  
            image_content = []  
            
            paragraphs = soup.find('div', id='bo_v_con')
            if paragraphs:
                text_content = "\n".join([element.get_text(strip=True) for element in paragraphs.find_all(['p', 'div', 'li'])])
               
                if text_content.strip() == "":
                    text_content = ""
            
                for img in paragraphs.find_all('img'):
                    img_src = img.get('src')
                    if img_src:
                        image_content.append(img_src)

            date_element = soup.select_one("strong.if_date") 
            date = date_element.get_text(strip=True) if date_element else "Unknown Date"

            if title != "Unknown Title":
                return title, text_content, image_content, date, url
            else:
                return None, None, None, None, None  
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return None, None, None, None, url

    with ThreadPoolExecutor() as executor:
        results = executor.map(fetch_text_and_date, urls)

    all_data = [(title, text_content, image_content, date, url) for title, text_content, image_content, date, url in results if title is not None]
    return all_data

document_data = extract_text_and_date_from_url(urls)

class CharacterTextSplitter:
    def __init__(self, chunk_size=1100, chunk_overlap=150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i:i + self.chunk_size])
        return chunks

text_splitter = CharacterTextSplitter(chunk_size=1100, chunk_overlap=150)

texts = []
image_url=[]
titles = []
doc_urls = []
doc_dates = []
for title, doc, image, date, url in document_data:
    if isinstance(doc, str) and doc.strip(): 
        split_texts = text_splitter.split_text(doc)
        texts.extend(split_texts)
        titles.extend([title] * len(split_texts))
        doc_urls.extend([url] * len(split_texts))
        doc_dates.extend([date] * len(split_texts)) 

        if image:
            image_url.extend([image] * len(split_texts))  
        else: 
            image_url.extend(["No content"] * len(split_texts))  

    elif image:  
        texts.append("No content")
        titles.append(title)
        doc_urls.append(url)
        doc_dates.append(date)
        image_url.append(image)  
    else:  
        texts.append("No content")
        image_url.append("No content") 
        titles.append(title)
        doc_urls.append(url)
        doc_dates.append(date)

def extract_professor_info_from_urls(urls):
    all_data = []

    def fetch_professor_info(url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            professor_elements = soup.find("div", id="dr").find_all("li")

            for professor in professor_elements:
           
                image_element = professor.find("div", class_="dr_img").find("img")
                image_content = image_element["src"] if image_element else "Unknown Image URL"

                name_element = professor.find("div", class_="dr_txt").find("h3")
                title = name_element.get_text(strip=True) if name_element else "Unknown Name"

                contact_info = professor.find("div", class_="dr_txt").find_all("dd")
                contact_number = contact_info[0].get_text(strip=True) if len(contact_info) > 0 else "Unknown Contact Number"
                email = contact_info[1].get_text(strip=True) if len(contact_info) > 1 else "Unknown Email"
                text_content = f"{title}, {contact_number}, {email}"

                date = "작성일24-01-01 00:00"

                prof_url_element = professor.find("a")
                prof_url = prof_url_element["href"] if prof_url_element else "Unknown URL"

                all_data.append((title, text_content, image_content, date, prof_url))

        except Exception as e:
            print(f"Error processing {url}: {e}")

    with ThreadPoolExecutor() as executor:
        results = executor.map(fetch_professor_info, urls)

    return all_data

def extract_professor_info_from_urls_2(urls):
    all_data = []

    def fetch_professor_info(url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            professor_elements = soup.find("div", id="Student").find_all("li")

            for professor in professor_elements:
           
                image_element = professor.find("div", class_="img").find("img")
                image_content = image_element["src"] if image_element else "Unknown Image URL"

                name_element = professor.find("div", class_="cnt").find("div", class_="name")
                title = name_element.get_text(strip=True) if name_element else "Unknown Name"

                contact_place = professor.find("div", class_="dep").get_text(strip=True) if professor.find("div", class_="dep") else "Unknown Contact Place"
                email_element = professor.find("dl", class_="email").find("dd").find("a")
                email = email_element.get_text(strip=True) if email_element else "Unknown Email"

                text_content = f"성함(이름):{title}, 연구실(장소):{contact_place}, 이메일:{email}"

                date = "작성일24-01-01 00:00"
                prof_url = url

                all_data.append((title, text_content, image_content, date, prof_url))

        except Exception as e:
            print(f"Error processing {url}: {e}")

    with ThreadPoolExecutor() as executor:
        executor.map(fetch_professor_info, urls)

    return all_data

def extract_professor_info_from_urls_3(urls):
    all_data = []

    def fetch_professor_info(url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            professor_elements = soup.find("div", id="Student").find_all("li")

            for professor in professor_elements:
              
                image_element = professor.find("div", class_="img").find("img")
                image_content = image_element["src"] if image_element else "Unknown Image URL"

                
                name_element = professor.find("div", class_="cnt").find("h1")
                title = name_element.get_text(strip=True) if name_element else "Unknown Name"

                contact_number_element = professor.find("span", class_="period")
                contact_number = contact_number_element.get_text(strip=True) if contact_number_element else "Unknown Contact Number"

                contact_info = professor.find_all("dl", class_="dep")
                contact_place = contact_info[0].find("dd").get_text(strip=True) if len(contact_info) > 0 else "Unknown Contact Place"

                email = contact_info[1].find("dd").find("a").get_text(strip=True) if len(contact_info) > 1 else "Unknown Email"

                role = contact_info[2].find("dd").get_text(strip=True) if len(contact_info) > 2 else "Unknown Role"

                text_content = f"성함(이름):{title}, 연락처(전화번호):{contact_number}, 사무실(장소):{contact_place}, 이메일:{email}, 담당업무:{role}"

                date = "작성일24-01-01 00:00"
                prof_url = url

                all_data.append((title, text_content, image_content, date, prof_url))

        except Exception as e:
            print(f"Error processing {url}: {e}")

    with ThreadPoolExecutor() as executor:
        executor.map(fetch_professor_info, urls)

    return all_data

urls2 = [
    "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub2_1&lang=kor",
]

urls3 = [
    "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub2_2&lang=kor",
]

urls4 = [
    "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub2_5&lang=kor",
]

prof_data = extract_professor_info_from_urls(urls2)
prof_data_2 = extract_professor_info_from_urls_2(urls3)
prof_data_3 = extract_professor_info_from_urls_3(urls4)

combined_prof_data = prof_data + prof_data_2 + prof_data_3

professor_texts = []
professor_image_urls = []
professor_titles = []
professor_doc_urls = []
professor_doc_dates = []

for title, doc, image, date, url in combined_prof_data :
    if isinstance(doc, str) and doc.strip(): 
        split_texts = text_splitter.split_text(doc)
        professor_texts.extend(split_texts)
        professor_titles.extend([title] * len(split_texts))  
        professor_doc_urls.extend([url] * len(split_texts))
        professor_doc_dates.extend([date] * len(split_texts))

        if image:
            professor_image_urls.extend([image] * len(split_texts))  
        else:
            professor_image_urls.extend(["No content"] * len(split_texts)) 

    elif image:  
        professor_texts.append("No content")
        professor_titles.append(title)
        professor_doc_urls.append(url)
        professor_doc_dates.append(date)
        professor_image_urls.append(image) 

    else:  
        professor_texts.append("No content")
        professor_image_urls.append("No content")  
        professor_titles.append(title)
        professor_doc_urls.append(url)
        professor_doc_dates.append(date)

texts.extend(professor_texts)
image_url.extend(professor_image_urls)
titles.extend(professor_titles)
doc_urls.extend(professor_doc_urls)
doc_dates.extend(professor_doc_dates)


def get_latest_wr_id_1():
    url = "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_3_b"
    response = requests.get(url)
    if response.status_code == 200:
        match = re.findall(r'wr_id=(\d+)', response.text)
        if match:
          max_wr_id = max(int(wr_id) for wr_id in match)
          return max_wr_id
    return None


now_company_number=get_latest_wr_id_1()

company_urls=[]
for number in range(now_company_number,1149,-1):
  company_urls.append("https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_3_b&wr_id="+str(number))


def extract_company_from_url(urls):
    all_data = []

    def fetch_text_and_date(url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            title_element = soup.find('span', class_='bo_v_tit')
            title = title_element.get_text(strip=True) if title_element else "Unknown Title"

            text_content = "Unknown Content" 
            image_content = []  

            paragraphs = soup.find('div', id='bo_v_con')
            if paragraphs:
                text_content = "\n".join([element.get_text(strip=True) for element in paragraphs.find_all(['p', 'div', 'li'])])
         
                if text_content.strip() == "":
                    text_content = ""
            
                for img in paragraphs.find_all('img'):
                    img_src = img.get('src')
                    if img_src:
                        image_content.append(img_src)

            date_element = soup.select_one("strong.if_date") 
            date = date_element.get_text(strip=True) if date_element else "Unknown Date"

            if title != "Unknown Title":
                return title, text_content, image_content, date, url  
            else:
                return None, None, None, None, None
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return None, None, None, None, url

    with ThreadPoolExecutor() as executor:
        results = executor.map(fetch_text_and_date, urls)

    all_data = [(title, text_content, image_content, date, url) for title, text_content, image_content, date, url in results if title is not None]
    return all_data

company_data= extract_company_from_url(company_urls)

for title, doc, image, date, url in company_data:
    if isinstance(doc, str) and doc.strip():
        split_texts = text_splitter.split_text(doc)
        texts.extend(split_texts)
        titles.extend([title] * len(split_texts))  
        doc_urls.extend([url] * len(split_texts))
        doc_dates.extend([date] * len(split_texts)) 

        if image: 
            image_url.extend([image] * len(split_texts))
        else: 
            image_url.extend(["No content"] * len(split_texts)) 

    elif image:  
        texts.append("No content")
        titles.append(title)
        doc_urls.append(url)
        doc_dates.append(date)
        image_url.append(image) 
    else:  
        texts.append("No content")
        image_url.append("No content")  
        titles.append(title)
        doc_urls.append(url)
        doc_dates.append(date)

def get_latest_wr_id_2():
    url = "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_4"
    response = requests.get(url)
    if response.status_code == 200:
        match = re.findall(r'wr_id=(\d+)', response.text)
        if match:
          max_wr_id = max(int(wr_id) for wr_id in match)
          return max_wr_id
    return None

now_seminar_number=get_latest_wr_id_2()

seminar_urls=[]
for number in range(now_seminar_number,246,-1):
  seminar_urls.append("https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_4&wr_id="+str(number))


def extract_seminar_from_url(urls):
    all_data = []

    def fetch_text_and_date(url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            title_element = soup.find('span', class_='bo_v_tit')
            title = title_element.get_text(strip=True) if title_element else "Unknown Title"

            text_content = "Unknown Content"  
            image_content = [] 

            paragraphs = soup.find('div', id='bo_v_con')
            if paragraphs:
                
                text_content = "\n".join([element.get_text(strip=True) for element in paragraphs.find_all(['p', 'div', 'li'])])
          
                if text_content.strip() == "":
                    text_content = ""
            
                for img in paragraphs.find_all('img'):
                    img_src = img.get('src')
                    if img_src:
                        image_content.append(img_src)

            date_element = soup.select_one("strong.if_date")
            date = date_element.get_text(strip=True) if date_element else "Unknown Date"

            if title != "Unknown Title":
                return title, text_content, image_content, date, url  
            else:
                return None, None, None, None, None
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return None, None, None, None, url

    with ThreadPoolExecutor() as executor:
        results = executor.map(fetch_text_and_date, urls)

    all_data = [(title, text_content, image_content, date, url) for title, text_content, image_content, date, url in results if title is not None]
    return all_data

seminar_data= extract_seminar_from_url(seminar_urls)

for title, doc, image, date, url in seminar_data:
    if isinstance(doc, str) and doc.strip():  
        split_texts = text_splitter.split_text(doc)
        texts.extend(split_texts)
        titles.extend([title] * len(split_texts)) 
        doc_urls.extend([url] * len(split_texts))
        doc_dates.extend([date] * len(split_texts)) 

        if image: 
            image_url.extend([image] * len(split_texts)) 
        else:  
            image_url.extend(["No content"] * len(split_texts))  

    elif image:  
        texts.append("No content")
        titles.append(title)
        doc_urls.append(url)
        doc_dates.append(date)
        image_url.append(image)  

    else:  
        texts.append("No content")
        image_url.append("No content")  
        titles.append(title)
        doc_urls.append(url)
        doc_dates.append(date)


embeddings = UpstageEmbeddings(
  api_key=upstage_api_key,
  model="solar-embedding-1-large"
) 
dense_doc_vectors = np.array(embeddings.embed_documents(texts))  

for i, embedding in enumerate(dense_doc_vectors):
    metadata = {
        "title": titles[i],
        "text": texts[i],
        "url": doc_urls[i],  
        "date": doc_dates[i]  
    }
    index.upsert([(str(i), embedding.tolist(), metadata)])


def transformed_query(content):
    query_nouns = []

    pattern = r'\d+(?:학년도|년|월|일|학기|시|분|초|기|개)?'
    number_matches = re.findall(pattern, content)
    query_nouns += number_matches
    for match in number_matches:
        content = content.replace(match, '')

    eng_kor_pattern = r'\b[a-zA-Z]+[가-힣]+\b'
    eng_kor_matches = re.findall(eng_kor_pattern, content)
    query_nouns += eng_kor_matches
    for match in eng_kor_matches:
        content = content.replace(match, '')

    english_words = re.findall(r'\b[a-zA-Z]+\b', content)
    query_nouns += english_words
    for match in english_words:
        content = content.replace(match, '')
    if '튜터' in content:
        query_nouns.append('TUTOR')
        content = content.replace('튜터', '')
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
   
    related_keywords = ['세미나', '행사', '특강', '강연']
    if any(keyword in content for keyword in related_keywords):
        for keyword in related_keywords:
            query_nouns.append(keyword)
    keywords=['공지','사항','공지사항']
    if any(keyword in content for keyword in keywords):
    
      for keyword in keywords:
          content = content.replace(keyword, '')
          query_nouns.append('공지사항')  
    okt = Okt()
    additional_nouns = [noun for noun in okt.nouns(content) if len(noun) > 1]
    query_nouns += additional_nouns

    if '수강' in content:
        related_keywords = ['변경', '신청', '정정', '취소','꾸러미']
        for keyword in related_keywords:
            if keyword in content:
                combined_keyword = '수강' + keyword
                query_nouns.append(combined_keyword)
                if ('수강' in query_nouns):
                  query_nouns.remove('수강')
                for keyword in related_keywords:
                  if keyword in query_nouns:
                    query_nouns.remove(keyword)
   
    query_nouns = list(set(query_nouns))
    return query_nouns

tokenized_titles = [transformed_query(title) for title in titles]

bm25_titles = BM25Okapi(tokenized_titles, k1=1.5, b=0.75)  

def best_docs(user_question):
    
      query_noun=transformed_query(user_question)
      print(f"=================\n\n question: {user_question} 추출된 명사: {query_noun}")

      title_question_similarities = bm25_titles.get_scores(query_noun) 
      title_question_similarities/=24

      def adjust_similarity_scores(query_noun, titles, similarities):
          for idx, title in enumerate(titles):
              matching_nouns = [noun for noun in query_noun if noun in title]

              if matching_nouns:
                  similarities[idx] += 0.21* len(matching_nouns)
                  for noun in matching_nouns:
                    if re.search(r'\d', noun): 
                        similarities[idx] += 0.21
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
              if texts[idx] == "No content":
                  similarities[idx] *=2.5 
              if '마일리지' in query_noun and '마일리지' in title:
                  similarities[idx]+=1
              if '신입생' in query_noun and '수강신청' in query_noun and '일괄수강신청' in title:
                  similarities[idx]+=2.5
              if '채용' in query_noun:
                  similarities[idx]+=1
          return similarities

      adjusted_similarities = adjust_similarity_scores(query_noun, titles, title_question_similarities)
      
      top_20_titles_idx = np.argsort(title_question_similarities)[-20:][::-1]

      Bm25_best_docs = [(titles[i], doc_dates[i], texts[i], doc_urls[i],image_url[i]) for i in top_20_titles_idx]

      query_dense_vector = np.array(embeddings.embed_query(user_question)) 
      
      pinecone_results_text = index.query(vector=query_dense_vector.tolist(), top_k=20, include_values=True, include_metadata=True)
      pinecone_similarities_text = [res['score'] for res in pinecone_results_text['matches']]
      pinecone_docs_text = [(res['metadata'].get('title', 'No Title'),
                            res['metadata'].get('date', 'No Date'),
                            res['metadata'].get('text', ''),
                            res['metadata'].get('url', 'No URL')) for res in pinecone_results_text['matches']]

      combine_dense_docs = []

      for idx, text_doc in enumerate(pinecone_docs_text):
          text_similarity = pinecone_similarities_text[idx]*3.65
          combine_dense_docs.append((text_similarity, text_doc))  
          
      combine_dense_docs.sort(key=lambda x: x[0], reverse=True)

      combine_dense_doc = []

      for score, (title, date, text, url) in combine_dense_docs:
          combine_dense_doc.append((score, title, text, date, url))

      final_best_docs = []

      for score, title, text, date, url in combine_dense_doc:
          matched = False
          for bm25_doc in Bm25_best_docs:
              if bm25_doc[0] == title:  
                  combined_similarity = score + adjusted_similarities[titles.index(bm25_doc[0])]
                  final_best_docs.append((combined_similarity, bm25_doc[0], bm25_doc[1], bm25_doc[2], bm25_doc[3], bm25_doc[4]))
                  matched = True
                  break
          if not matched:

              final_best_docs.append((score,title, date, text, url, "No content"))

      for bm25_doc in Bm25_best_docs:
          matched = False
          for score, title, text, date, url in combine_dense_doc:
              if bm25_doc[0] == title and bm25_doc[1]==text:  
                  matched = True
                  break
          if not matched:
              combined_similarity = adjusted_similarities[titles.index(bm25_doc[0])] 
              final_best_docs.append((combined_similarity, bm25_doc[0], bm25_doc[1], bm25_doc[2], bm25_doc[3], bm25_doc[4]))
      final_best_docs.sort(key=lambda x: x[0], reverse=True)
      final_best_docs=final_best_docs[:20]

      def last_filter_keyword(DOCS,query_noun):
          Final_best=DOCS
          for idx, doc in enumerate(DOCS):
              score, title, date, text, url, image = doc
              if any(keyword in query_noun for keyword in ['세미나','행사','특강']):
                 for i in range(0,len(seminar_data)):
                  if title==seminar_data[i][0]:
                    score+=1.5
              if any(keyword in text for keyword in ['계약학과', '대학원', '타대학원']) and not any(keyword in query_noun for keyword in ['계약학과', '대학원', '타대학원']):
                  score -= 0.1 
              if '대학원' not in query_noun and '대학원생' not in query_noun and ('대학원' in title or '대학원생' in title):
                  score-=1
              if any(keyword in query_noun for keyword in ['대학원','대학원생']) and any (keyword in title for keyword in ['대학원','대학원생']):
                  score+=1.5
              if (any(keyword in query_noun for keyword in ['담당','업무','일']) or any(keyword in query_noun for keyword in ['직원','교수','선생','선생님'])) and date=="작성일24-01-01 00:00":
                  if (any(keys in query_noun for keys in ['교수'])):
                    check=0
                    for i in range(0,len(prof_data_3)):
                        if title==prof_data_3[i][0]:
                          check=1
                          break
                    if check==0:
                      score+=0.4
                    else:
                      score-=0.9
                  else:
                    score+=0.9
              match = re.search(r"(?<![\[\(])\b수강\w*\b(?![\]\)])", title)

              if match:
                  full_keyword = match.group(0)
                  if full_keyword not in query_noun:
                    score-=0.7
                  else:
                    score+=0.7
              Final_best[idx] = (score, title, date, text,  url, image)
              
          return Final_best

      final_best_docs=last_filter_keyword(final_best_docs,query_noun)
      final_best_docs.sort(key=lambda x: x[0], reverse=True)

      def cluster_documents_by_similarity(docs, threshold=0.89):
          clusters = []

          for doc in docs:
              title = doc[1]
              added_to_cluster = False
              for cluster in clusters:
                  cluster_title = cluster[0][1]
                  similarity = SequenceMatcher(None, cluster_title, title).ratio()
                  if similarity >= threshold:
                      if (-doc[0]+cluster[0][0]<0.26 or cluster_title==title and cluster[0][3]!=doc[2]):
                        cluster.append(doc)
                      added_to_cluster = True
                      break

              if not added_to_cluster:
                  clusters.append([doc])

          return clusters
      
      clusters = cluster_documents_by_similarity(final_best_docs)
      query_nouns=transformed_query(user_question)
      def parse_date(date_str):
          clean_date_str = date_str.replace("작성일", "").strip()
          return datetime.strptime(clean_date_str, "%y-%m-%d %H:%M")
      top_0_cluster_similar=clusters[0][0][0]
      top_1_cluster_similar=clusters[1][0][0]
      keywords = ["최근", "최신", "현재", "지금"]
      if (top_0_cluster_similar-top_1_cluster_similar<=0.3):
          if (any(keyword in word for word in query_nouns for keyword in keywords) or top_0_cluster_similar-clusters[len(clusters)-1][0][0]<=0.3):
            if (top_0_cluster_similar-clusters[len(clusters)-1][0][0]<=0.3):
              sorted_cluster=sorted(clusters, key=lambda doc: doc[0][2], reverse=True)
              sorted_cluster=sorted_cluster[0]
            else:
              if (top_0_cluster_similar-top_1_cluster_similar<=0.3):
                date1 = parse_date(clusters[0][0][2])
                date2 = parse_date(clusters[1][0][2])
                if date1<date2:
                  result_docs=clusters[1]
                else:
                  result_docs=clusters[0]
                sorted_cluster = sorted(result_docs, key=lambda doc: doc[2], reverse=True)

              else:
                sorted_cluster=sorted(clusters, key=lambda doc: doc[0][0], reverse=True)
                sorted_cluster=sorted_cluster[0]
          else:
            if (top_0_cluster_similar-top_1_cluster_similar<=0.15):
              date1 = parse_date(clusters[0][0][2])
              date2 = parse_date(clusters[1][0][2])
              if date1<date2:
                result_docs=clusters[1]
              else:
                result_docs=clusters[0]
              sorted_cluster = sorted(result_docs, key=lambda doc: doc[2], reverse=True)
            else:
              result_docs=clusters[0]
              sorted_cluster=sorted(result_docs,key=lambda doc: doc[0],reverse=True)
      else: 
          number_pattern = r"\d"
          period_word=["여름","겨울"]
          if (any(keyword in word for word in query_nouns for keyword in keywords) or not any(re.search(number_pattern, word) for word in query_nouns) or not any(key in word for word in query_nouns for key in period_word)):
              if (any(re.search(number_pattern, word) for word in query_nouns) or any(key in word for word in query_nouns for key in period_word)):
                result_docs=clusters[0]
                sorted_cluster = sorted(result_docs, key=lambda doc: doc[0], reverse=True)
              else:
                result_docs=clusters[0]
                sorted_cluster=sorted(result_docs,key=lambda doc: doc[2],reverse=True)
          else:
            result_docs=clusters[0]
            sorted_cluster = sorted(clusters[0], key=lambda doc: doc[0], reverse=True)

      def organize_documents_v2(sorted_cluster, titles, doc_dates, texts, doc_urls, image_urls):
          top_doc = sorted_cluster[0]
          top_title = top_doc[1]

          new_sorted_cluster = []
          count=0
          for i, title in enumerate(titles):
              if title == top_title:
                  count+=1
                  new_doc = (top_doc[0], titles[i], doc_dates[i], texts[i], doc_urls[i], image_urls[i])
                  new_sorted_cluster.append(new_doc)
          for i in range(count-1):
            fix_similar=list(new_sorted_cluster[i])
            fix_similar[0]=fix_similar[0]+0.3*count
            new_sorted_cluster[i]=tuple(fix_similar)
          for doc in sorted_cluster:
              doc_title = doc[1]
              if doc_title != top_title:
                  new_sorted_cluster.append(doc)

          return new_sorted_cluster,count

      final_cluster,count = organize_documents_v2(sorted_cluster, titles, doc_dates, texts, doc_urls, image_url)
  
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
8. 질문이 공지사항의 내용과 전혀 관련이 없다고 판단하면 응답하지 말아주세요. 예를 들면 "너는 무엇을 알까", "점심메뉴 추천"과 같이 일반 상식을 요구하는 질문은 거절해주세요.

답변:"""

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
      doc_urls = [doc[4] for doc in best_docs] 

    documents = [
        Document(page_content=text, metadata={"title": title, "url": url, "doc_date": datetime.strptime(date, '작성일%y-%m-%d %H:%M')})
        for title, text, url, date in zip(doc_titles, doc_texts, doc_urls, doc_dates)
    ]
    
    query_nouns=transformed_query(user_question)
   
    relevant_docs = [doc for doc in documents if any(keyword in doc.page_content for keyword in query_nouns)]
  
    if not relevant_docs:
      return None, None, None
    vector_store = FAISS.from_documents(relevant_docs, embeddings)
    retriever = vector_store.as_retriever()
    llm = ChatUpstage(api_key=upstage_api_key)
  
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

    return qa_chain, relevant_docs 

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

def get_ai_message(question):
    top_doc, query_noun = best_docs(question) 
    top_docs = [list(doc) for doc in top_doc]
    if False ==(question_valid(question, top_docs[0][1], query_noun)):
      for i in range(len(top_docs)):
          top_docs[i][0]-=1
    print(f"\n\ntitles: {top_docs[0][1]} similarity: {top_docs[0][0]}, text:{(len(top_docs[0][3]))} doc_dates: {top_docs[0][2]} URL: {top_docs[0][4]}\n\n\n")

    if len(top_docs[0])==6 and top_docs[0][5]!="No content" and top_docs[0][3]=="No content" and top_docs[0][0]>1.8:
         image_display = ""
         for img_url in top_docs[0][5]: 
              image_display += f"<img src='{img_url}' alt='관련 이미지' style='max-width: 500px; max-height: 500px;' /><br>"
         doc_references = top_docs[0][4]
         display(HTML(image_display))
         return  f"항상 정확한 답변을 제공하지 못할 수 있습니다.아래의 URL들을 참고하여 정확하고 자세한 정보를 확인하세요.\n{doc_references}"

    else:
        qa_chain, relevant_docs = get_answer_from_chain(top_docs, question) 
        image_display = ""
        seen_img_urls = set() 
        
        if top_docs[0][5] and top_docs[0][5] != "No content":
            if isinstance(top_docs[0][5], list):
                for img_url in top_docs[0][5]: 
                    if img_url not in seen_img_urls:  
                        image_display += f"<img src='{img_url}' alt='관련 이미지' style='max-width: 500px; max-height: 500px;' /><br>"
                        seen_img_urls.add(img_url) 
            else:
                img_url = top_docs[0][5]
                if img_url not in seen_img_urls:
                    image_display += f"<img src='{img_url}' alt='관련 이미지' style='max-width: 500px; max-height: 500px;' /><br>"
                    seen_img_urls.add(img_url)

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
        existing_answer = qa_chain.invoke(question)
        answer_result=existing_answer
        display(HTML(image_display))
        doc_references = "\n".join([
            f"\n참고 문서 URL: {doc.metadata['url']}"
            for doc in relevant_docs[:1] if doc.metadata.get('url') != 'No URL'
        ])
        return f"{answer_result}\n\n------------------------------------------------\n항상 정확한 답변을 제공하지 못할 수 있습니다.\n아래의 URL들을 참고하여 정확하고 자세한 정보를 확인하세요.\n{doc_references}"
