from flask import Flask, request, jsonify
from ai_modules import get_ai_message
import logging

app = Flask(__name__)

@app.route('/api/ai-response', methods=['POST'])
def ai_response():
    try:
        data = request.get_json()
        question = data.get('question')

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # AI 응답 생성
        response = get_ai_message(question)

        # JSON 객체로 응답 반환
        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
