# 🏪 Local Business Discovery Bot

A specialized chatbot built with Python using NLP libraries like spaCy and NLTK. This chatbot helps users discover local businesses including restaurants, shops, services, and entertainment venues in their area.

## ✨ Features

- **Real API Integration**: Uses OpenStreetMap Nominatim API for live business data (no API keys required!)
- **Business Discovery**: Finds restaurants, shops, services, and entertainment based on user preferences
- **Intent Recognition**: Uses TF-IDF vectorization and cosine similarity to classify business search intents
- **Sentiment Analysis**: Analyzes the emotional tone of user messages
- **Entity Extraction**: Extracts named entities using spaCy
- **Web Interface**: Beautiful, responsive web interface built with Flask
- **Real-time Chat**: Interactive chat experience with typing indicators
- **Confidence Scoring**: Shows confidence levels for intent recognition
- **Business Categories**: Specialized responses for restaurants, shopping, services, and entertainment
- **Fallback System**: Uses sample data when APIs are unavailable

## 🚀 Quick Start

### 1. Setup (One-time)

```bash
# Create virtual environment
python3 -m venv chatbot_env

# Activate it
source chatbot_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -c "import spacy; spacy.cli.download('en_core_web_sm')"
```

### 2. Run the Chatbot

```bash
# Activate virtual environment
source chatbot_env/bin/activate

# Start the chatbot
python app.py
```

Open your browser: **http://localhost:5000**

That's it! 🎉

## 📁 Project Structure

```
nlp_csa/
├── chatbot.py          # Local business discovery AI brain
├── app.py              # Flask web server
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Business discovery web interface
└── README.md           # This file
```

## 🧠 How It Works

### Intent Recognition
The chatbot uses TF-IDF vectorization to convert text into numerical vectors, then uses cosine similarity to match user input with predefined intent patterns.

### Sentiment Analysis
Simple sentiment analysis based on positive/negative word patterns, with room for enhancement using more sophisticated models.

### Entity Extraction
Uses spaCy's named entity recognition to identify and extract entities like names, locations, organizations, etc.

### Response Generation
Based on the classified intent, the chatbot selects appropriate responses from predefined response templates.

## 🎯 Supported Business Intents

- **Restaurant Search**: "Find me a good restaurant", "Where can I get coffee?", "I want Italian food"
- **Business Search**: "Find me a store", "I need a service", "What's nearby?"
- **Entertainment Search**: "What entertainment options?", "Find me a bar", "I want to see a movie"
- **Shopping Search**: "Where can I buy clothes?", "Find me a pharmacy", "I need groceries"
- **Service Search**: "I need a hair salon", "Find me a gym", "Where can I get my car fixed?"
- **Location Query**: "What's in downtown?", "Find businesses near me", "What's in this area?"
- **Price Query**: "Find cheap restaurants", "What's affordable?", "Show me expensive options"
- **Rating Query**: "Find the best restaurants", "Show me highly rated places", "What's popular?"
- **Hours Query**: "What's open now?", "Find late night options", "What's open on weekends?"

## 🔧 Customization

### Adding New Intents

1. Edit the `_load_training_data()` method in `chatbot.py`
2. Add new intent patterns and responses
3. The chatbot will automatically learn the new patterns

### Improving Accuracy

- Add more training patterns for each intent
- Adjust the confidence threshold in `classify_intent()`
- Use more sophisticated NLP models
- Implement machine learning for intent classification

### Styling the Web Interface

Edit `templates/index.html` to customize the appearance, colors, and layout.

## 🛠️ Technical Details

### Dependencies
- **spaCy**: Advanced NLP processing and entity extraction
- **NLTK**: Text preprocessing and tokenization
- **scikit-learn**: TF-IDF vectorization and similarity calculations
- **Flask**: Web framework for the interface
- **Flask-CORS**: Cross-origin resource sharing

### Performance
- Fast response times with TF-IDF vectorization
- Efficient similarity calculations using cosine similarity
- Lightweight web interface with minimal dependencies

## 🚀 Advanced Features

### Confidence Scoring
The chatbot provides confidence scores for intent recognition, helping identify when it's uncertain about user intent.

### Entity Recognition
Extracts and displays named entities from user messages, providing additional context for responses.

### Sentiment Analysis
Analyzes the emotional tone of messages to provide more contextually appropriate responses.

## 📊 API Endpoints

- `GET /`: Main chat interface
- `POST /chat`: Send message and get response
- `GET /health`: Health check endpoint

### Chat API Response Format

```json
{
  "response": "Hello! How can I help you today?",
  "intent": "greeting",
  "confidence": 0.85,
  "sentiment": "positive",
  "entities": [
    {
      "text": "John",
      "label": "PERSON",
      "start": 0,
      "end": 4
    }
  ]
}
```

## 🔍 Troubleshooting

### Common Issues

1. **spaCy model not found**: Run `python -m spacy download en_core_web_sm`
2. **NLTK data missing**: The script automatically downloads required NLTK data
3. **Port already in use**: Change the port in `app.py` if 5000 is occupied

### Performance Tips

- Use a virtual environment to avoid dependency conflicts
- Ensure you have sufficient RAM for spaCy models
- Consider using smaller spaCy models for faster loading

## 🤝 Contributing

Feel free to enhance the chatbot by:
- Adding more sophisticated NLP models
- Implementing machine learning for better intent recognition
- Adding more training data and response patterns
- Improving the web interface
- Adding voice capabilities


## 🙏 Acknowledgments

- spaCy team for the excellent NLP library
- NLTK contributors for text processing tools
- Flask team for the web framework
- The open-source community for inspiration and tools

