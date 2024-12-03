from flask import Flask, request, jsonify
from ai_modules import get_ai_message, initialize_cache
import logging

app = Flask(__name__)

@app.route('/ai/ai-response', methods=['POST'])
def ai_response():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'No JSON data provided'}), 400

        question = data.get('question')

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # AI 응답 생성
        response = get_ai_message(question)

        # JSON 객체로 응답 반환
        if isinstance(response, dict):
            return jsonify(response)
        else:
            return jsonify({'error': 'Invalid response format from AI module'}), 500

    except Exception as e:
        app.logger.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# 서버 실행시 캐시 초기화.
if __name__ != "__main__":
    initialize_cache()