from flask import Flask, request, jsonify
from ai_modules import get_ai_message, initialize_cache
from flask_cors import CORS
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask 앱 생성 함수
def create_app():
    app = Flask(__name__)
    CORS(app)

    @app.route('/ai/ai-response', methods=['POST'])
    def ai_response():
        try:
            # 요청 데이터 가져오기
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400

            # 'question' 필드 가져오기
            question = data.get('question')
            if not isinstance(question, str) or not question.strip():
                return jsonify({'error': 'Invalid or missing question'}), 400

            # AI 응답 생성
            response = get_ai_message(question)

            # JSON 객체로 응답 반환
            if isinstance(response, dict):
                return jsonify(response)
            else:
                return jsonify({'error': 'Invalid response format from AI module'}), 500

        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

    return app

# 캐시 초기화
if __name__ == "__main__":
    initialize_cache()
    app = create_app()
    app.run(host="0.0.0.0", port=5000)
else:
    initialize_cache()
    app = create_app()
