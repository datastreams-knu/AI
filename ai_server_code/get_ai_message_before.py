##### 유사도 제목 날짜 본문  url image_url순으로 저장됨
def get_ai_message(question):
    top_doc = best_docs(question)  # 가장 유사한 문서 가져오기
    top_docs = [list(doc) for doc in top_doc]
    
    #top_docs 인덱스 구성
    # 0: 유사도, 
    # 1: 제목 
    # 2: 날짜 
    # 3: 본문내용
    # 4: url
    # 5: 이미지url
    
    #이미지만 존재하는 공지사항인 경우.
    if len(top_docs[0])==6 and top_docs[0][5] !="No content" and top_docs[0][3]=="No content" and top_docs[0][0]>1.8:
        # image url 한 string으로 연결하기
         image_urls = "\n".join(top_docs[0][5])  
         doc_references = top_docs[0][4]
         
         '''
        #이거 뭐하는 코드인지 모르겠음
         doc_references = top_docs[0][4]
         # content 초기화
         content = []
        # top_docs의 내용 확인
         if top_docs[0][3] == "No content":
             content = []  # No content일 경우 비우기
         else:
             content = top_docs[0][3]  # content에 top_docs[0][3] 내용 저장
         if content:
            html_output = f"{image_display}<p>{content}</p><hr>\n"
         else:
            html_output = f"{image_display}<p>>\n"
        
        # HTML 출력 및 반환할 내용 생성
         display(HTML(image_display))
         return  f"<p>항상 정확한 답변을 제공하지 못할 수 있습니다.아래의 URL들을 참고하여 정확하고 자세한 정보를 확인하세요.\n{doc_references}"
         '''
         
         # JSON 형식으로 반환할 객체 생성
         only_image_response = {
            "answer": None,
            "references": doc_references,
            "disclaimer": "항상 정확한 답변을 제공하지 못할 수 있습니다. 아래의 URL들을 참고하여 정확하고 자세한 정보를 확인하세요.",
            "images": image_urls
         }
         return only_image_response
         
         
    #이미지 + llm 답변이 있는 경우.
    else:
        qa_chain, retriever, relevant_docs = get_answer_from_chain(top_docs, question)  # 답변 생성 체인 생성

        # 기존의 교수님 이미지 URL 저장 코드 중 중복된 URL 방지 부분
        image_urls = ""
        seen_img_urls = set()  # 이미 출력된 이미지 URL을 추적하는 set
        
        #return 준비
        disclaimer = "항상 정확한 답변을 제공하지 못할 수 있습니다. 아래의 URL들을 참고하여 정확하고 자세한 정보를 확인하세요."
        # docs에 이미지가 있을 경우.
        if top_docs[0][5] and top_docs[0][5] != "No content":
            # 이미지 URL이 리스트 형태인지 확인하고, 문자열로 잘라서 처리
            if isinstance(top_docs[0][5], list):
                image_urls = "\n".join(top_docs)
                '''
                for img_url in top_docs[0][5]:  # 여러 이미지 URL에 대해 반복
                    if img_url not in seen_img_urls:  # img_url이 이미 출력되지 않은 경우
                        image_display += f"<img src='{img_url}' alt='관련 이미지' style='max-width: 500px; max-height: 500px;' /><br>"
                        seen_img_urls.add(img_url)  # img_url을 set에 추가하여 중복을 방지
                
            else:
                # top_docs[0][5]가 단일 문자열일 경우, 이를 그대로 출력
                img_url = top_docs[0][5]
                if img_url not in seen_img_urls:
                    image_display += f"<img src='{img_url}' alt='관련 이미지' style='max-width: 500px; max-height: 500px;' /><br>"
                    seen_img_urls.add(img_url)
                '''
        doc_references = top_docs[0][4]

        #답변이 공지사항에 존재하지 않을 경우 답변 형식
        notice_url = "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_1"
        not_in_notices_response = {
            "answer" : "해당 질문은 공지사항에 없는 내용입니다.\n 자세한 사항은 공지사항을 살펴봐주세요.",
            "refenrences" : notice_url,
            "disclaimer" : "항상 정확한 답변을 제공하지 못할 수 있습니다. 아래의 URL들을 참고하여 정확하고 자세한 정보를 확인하세요.",
            "images" : None
        }
        
        #이미지가 포함된 경우.

        if not qa_chain or not relevant_docs:
          if (top_docs[0][5]!="No content") and top_docs[0][0]>1.8:
            data = {
                "answer": "해당 질문에 대한 내용은 이미지 파일로 확인해주세요.",
                "references": doc_references,
                "disclaimer": disclaimer,
                "images" : image_urls
            }
            return data
          else:
            return not_in_notices_response

        #유사도 1.8이하 답변 처리
        if (top_docs[0][0]<1.8):
          return not_in_notices_response

        answer_result = qa_chain.invoke(question)# 초기 답변 생성 및 문자열로 할당
        
        # 상위 3개의 참조한 문서의 URL 포함 형식으로 반환
        doc_references = "\n".join([
            f"\n참고 문서 URL: {doc.metadata['url']}"
            for doc in relevant_docs[:1] if doc.metadata.get('url') != 'No URL'
        ])
        
        # dictionary으로 반환.
        data = {
        "answer": answer_result,
        "references": doc_references,
        "disclaimer": disclaimer,
        "images" : image_urls
        }

        return data
