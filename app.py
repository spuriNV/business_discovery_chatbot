from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chatbot import LocalBusinessDiscoveryBot
import json

app = Flask(__name__)
CORS(app)

# Initialize the local business discovery chatbot
chatbot = LocalBusinessDiscoveryBot()

@app.route('/')
def index():
    """Serve the main chat interface."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'response': 'Please enter a message.',
                'error': 'Empty message'
            }), 400
        
        # Validate message length
        if len(user_message) > 500:
            return jsonify({
                'response': 'Message too long. Please keep it under 500 characters.',
                'error': 'Message too long'
            }), 400
        
        # Get chatbot response
        result = chatbot.get_chat_analysis(user_message)
        
        return jsonify({
            'response': result['response'],
            'intent': result['intent'],
            'confidence': round(result['confidence'], 3),
            'sentiment': result['sentiment'],
            'entities': result['entities']
        })
        
    except Exception as e:
        return jsonify({
            'response': 'Sorry, I encountered an error processing your message.',
            'error': str(e)
        }), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'chatbot': 'ready'})

if __name__ == '__main__':
    print("🤖 Starting NLP Chatbot Web Interface...")
    print("📱 Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)