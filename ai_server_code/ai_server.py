from flask import Flask, request, jsonify
from ai_modules import get_ai_message  # ai_module에서 get_ai_response 함수 가져오기
import logging

app = Flask(__name__)

# AI 서버에서 질문을 받아 답변을 반환하는 API 엔드포인트
@app.route('/api/ai-response', methods=['POST'])
def ai_response():
    try:
        # 요청에서 질문 텍스트를 가져옴
        data = request.get_json()
        question = data.get('question')

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # AI 응답 생성
        response = get_ai_message(question)

        # 응답 반환
        return response
    except Exception as e:
        app.logger.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Gunicorn을 사용하여 서버 실행 권장
    # 예시: gunicorn -w 4 -b 0.0.0.0:5000 ai_server_endpoint:app
    app.run(host='0.0.0.0', port=5000, debug=True)
