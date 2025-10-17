import json
import random
import re
from typing import List, Dict, Tuple

import numpy as np
import nltk
import requests
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

class LocalBusinessDiscoveryBot:
    def __init__(self):
        """Initialize the local business discovery chatbot with NLP models and business data."""
        # Load spaCy model (you may need to download it: python -m spacy download en_core_web_sm)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy English model not found. Please install it with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Initialize NLTK components
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.lemmatizer = nltk.WordNetLemmatizer()
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            max_features=1000
        )
        
        # Training data for the chatbot
        self.training_data = self._load_training_data()
        self.intents = self.training_data['intents']
        self.responses = self.training_data['responses']
        
        # Business categories and keywords
        self.business_categories = self._load_business_categories()
        
        # Vectorize training data
        self._vectorize_training_data()
        
        # Sample business data (in a real app, this would come from APIs)
        self.sample_businesses = self._load_sample_businesses()
        
    def _load_training_data(self) -> Dict:
        """Load training data for the local business discovery chatbot."""
        return {
            'intents': [
                {
                    'intent': 'greeting',
                    'patterns': [
                        'hello', 'hi', 'hey', 'good morning', 'good afternoon',
                        'good evening', 'howdy', 'greetings', 'what\'s up'
                    ]
                },
                {
                    'intent': 'goodbye',
                    'patterns': [
                        'bye', 'goodbye', 'see you later', 'farewell',
                        'take care', 'catch you later', 'adios'
                    ]
                },
                {
                    'intent': 'thanks',
                    'patterns': [
                        'thank you', 'thanks', 'thank you very much',
                        'appreciate it', 'much obliged', 'grateful'
                    ]
                },
                {
                    'intent': 'restaurant_search',
                    'patterns': [
                        'restaurant', 'food', 'dining', 'eat', 'lunch', 'dinner',
                        'breakfast', 'cafe', 'coffee', 'pizza', 'burger', 'sushi',
                        'chinese', 'italian', 'mexican', 'thai', 'indian', 'vegan',
                        'vegetarian', 'gluten free', 'halal', 'kosher'
                    ]
                },
                {
                    'intent': 'business_search',
                    'patterns': [
                        'business', 'store', 'shop', 'service', 'find', 'near me',
                        'local', 'nearby', 'close', 'around', 'in the area'
                    ]
                },
                {
                    'intent': 'entertainment_search',
                    'patterns': [
                        'entertainment', 'movie', 'theater', 'cinema', 'bar', 'pub',
                        'club', 'nightlife', 'music', 'concert', 'show', 'event',
                        'museum', 'gallery', 'park', 'recreation', 'fun', 'activity'
                    ]
                },
                {
                    'intent': 'shopping_search',
                    'patterns': [
                        'shopping', 'mall', 'store', 'retail', 'clothes', 'fashion',
                        'shoes', 'electronics', 'bookstore', 'gift', 'jewelry',
                        'pharmacy', 'grocery', 'supermarket', 'market'
                    ]
                },
                {
                    'intent': 'service_search',
                    'patterns': [
                        'service', 'repair', 'maintenance', 'cleaning', 'laundry',
                        'dry cleaning', 'salon', 'barber', 'spa', 'massage',
                        'gym', 'fitness', 'bank', 'atm', 'gas station', 'car wash'
                    ]
                },
                {
                    'intent': 'location_query',
                    'patterns': [
                        'where', 'location', 'address', 'directions', 'how to get',
                        'near', 'close to', 'around', 'in', 'at', 'downtown',
                        'uptown', 'suburbs', 'neighborhood', 'area'
                    ]
                },
                {
                    'intent': 'price_query',
                    'patterns': [
                        'price', 'cost', 'expensive', 'cheap', 'affordable',
                        'budget', 'money', 'dollar', 'costly', 'inexpensive',
                        'free', 'discount', 'deal', 'special'
                    ]
                },
                {
                    'intent': 'rating_query',
                    'patterns': [
                        'rating', 'review', 'stars', 'good', 'bad', 'best',
                        'worst', 'recommend', 'popular', 'famous', 'well known',
                        'quality', 'excellent', 'terrible', 'amazing'
                    ]
                },
                {
                    'intent': 'hours_query',
                    'patterns': [
                        'hours', 'open', 'closed', 'time', 'schedule', 'when',
                        'available', 'operating', 'business hours', 'weekend',
                        'weekday', 'monday', 'tuesday', 'wednesday', 'thursday',
                        'friday', 'saturday', 'sunday'
                    ]
                },
                {
                    'intent': 'help',
                    'patterns': [
                        'help', 'can you help me', 'what can you do',
                        'how can you help', 'what are your capabilities',
                        'how does this work', 'what can you find'
                    ]
                }
            ],
            'responses': {
                'greeting': [
                    "Hello! I'm your local business discovery assistant! I can help you find restaurants, shops, services, and entertainment in your area. What are you looking for?",
                    "Hi there! I specialize in finding great local businesses. Whether you need food, shopping, services, or fun activities, I'm here to help!",
                    "Hey! I'm your neighborhood business guide. I can help you discover hidden gems and popular spots in your area. What can I find for you?",
                    "Greetings! I'm your local business discovery bot. I know all the best spots in town - restaurants, shops, services, and more. How can I help?"
                ],
                'goodbye': [
                    "Goodbye! Hope you found some great local spots!",
                    "See you later! Enjoy exploring your neighborhood!",
                    "Farewell! Come back anytime you need local business recommendations!",
                    "Bye! Happy discovering in your area!"
                ],
                'thanks': [
                    "You're welcome! Happy to help you discover great local businesses!",
                    "No problem! I love helping people find amazing local spots!",
                    "My pleasure! Is there anything else I can help you discover?",
                    "Glad I could help you find what you're looking for!"
                ],
                'restaurant_search': [
                    "I'd love to help you find great restaurants! I can search for specific cuisines, price ranges, and locations. What type of food are you craving?",
                    "Let me help you discover amazing local restaurants! Are you looking for a specific cuisine, price range, or location?",
                    "I know all the best local dining spots! Tell me what you're in the mood for and I'll find the perfect place for you.",
                    "Great choice! I can find restaurants based on cuisine, price, location, and ratings. What are you looking for?"
                ],
                'business_search': [
                    "I can help you find local businesses and services! What type of business are you looking for?",
                    "Let me help you discover local businesses in your area. What do you need?",
                    "I know all the local businesses! Tell me what you're looking for and I'll find the best options.",
                    "I can find any type of local business for you! What service or store do you need?"
                ],
                'entertainment_search': [
                    "I can help you find great entertainment options! Are you looking for movies, bars, events, or activities?",
                    "Let me help you discover fun things to do! What type of entertainment are you interested in?",
                    "I know all the best entertainment spots! Tell me what you're in the mood for and I'll find something perfect.",
                    "Great! I can find entertainment based on your preferences. What are you looking for?"
                ],
                'shopping_search': [
                    "I can help you find great shopping spots! What are you looking to buy?",
                    "Let me help you discover local shopping options! What do you need to find?",
                    "I know all the best local shops! Tell me what you're shopping for and I'll find the perfect places.",
                    "I can find any type of store for you! What are you looking to buy?"
                ],
                'service_search': [
                    "I can help you find local services! What type of service do you need?",
                    "Let me help you discover local service providers! What do you need done?",
                    "I know all the local service businesses! Tell me what you need and I'll find the best options.",
                    "I can find any type of service for you! What do you need help with?"
                ],
                'location_query': [
                    "I can help you find businesses in specific locations! What area are you interested in?",
                    "Let me help you find businesses by location! Where are you looking?",
                    "I know businesses all over town! Tell me the area you're interested in and I'll find great options.",
                    "I can search by location! What neighborhood or area are you looking in?"
                ],
                'price_query': [
                    "I can help you find businesses within your budget! What's your price range?",
                    "Let me help you find affordable options! What's your budget like?",
                    "I know businesses at all price points! Tell me your budget and I'll find great options.",
                    "I can filter by price! What's your budget range?"
                ],
                'rating_query': [
                    "I can help you find highly-rated businesses! I know all the top-rated spots in town.",
                    "Let me help you find the best-rated options! I can filter by ratings and reviews.",
                    "I know all the highly-rated businesses! Tell me what you're looking for and I'll find the best options.",
                    "I can find top-rated businesses for you! What are you looking for?"
                ],
                'hours_query': [
                    "I can help you find businesses with specific hours! What time do you need?",
                    "Let me help you find businesses that are open when you need them! What's your schedule?",
                    "I know business hours for all local spots! Tell me when you need to go and I'll find open options.",
                    "I can filter by hours! What time are you looking to visit?"
                ],
                'help': [
                    "I'm your local business discovery assistant! I can help you find restaurants, shops, services, and entertainment in your area. Just tell me what you're looking for!",
                    "I specialize in finding great local businesses! I can search by type, location, price, ratings, and hours. What can I help you discover?",
                    "I'm here to help you discover amazing local businesses! I can find restaurants, shops, services, and entertainment based on your preferences. What are you looking for?",
                    "I'm your neighborhood business guide! I can help you find any type of local business. Just tell me what you need and I'll find the best options for you!"
                ]
            }
        }
    
    def _load_business_categories(self) -> Dict:
        """Load business categories and keywords for better classification."""
        return {
            'restaurants': {
                'keywords': ['restaurant', 'food', 'dining', 'eat', 'lunch', 'dinner', 'breakfast', 'cafe', 'coffee', 'pizza', 'burger', 'sushi', 'chinese', 'italian', 'mexican', 'thai', 'indian', 'vegan', 'vegetarian', 'gluten free', 'halal', 'kosher'],
                'subcategories': ['fine dining', 'casual dining', 'fast food', 'cafe', 'bar', 'pub', 'food truck', 'bakery', 'deli']
            },
            'shopping': {
                'keywords': ['shopping', 'mall', 'store', 'retail', 'clothes', 'fashion', 'shoes', 'electronics', 'bookstore', 'gift', 'jewelry', 'pharmacy', 'grocery', 'supermarket', 'market'],
                'subcategories': ['clothing', 'electronics', 'books', 'gifts', 'jewelry', 'pharmacy', 'grocery', 'department store', 'specialty store']
            },
            'services': {
                'keywords': ['service', 'repair', 'maintenance', 'cleaning', 'laundry', 'dry cleaning', 'salon', 'barber', 'spa', 'massage', 'gym', 'fitness', 'bank', 'atm', 'gas station', 'car wash'],
                'subcategories': ['beauty', 'fitness', 'automotive', 'financial', 'cleaning', 'repair', 'professional services']
            },
            'entertainment': {
                'keywords': ['entertainment', 'movie', 'theater', 'cinema', 'bar', 'pub', 'club', 'nightlife', 'music', 'concert', 'show', 'event', 'museum', 'gallery', 'park', 'recreation', 'fun', 'activity'],
                'subcategories': ['movies', 'music', 'nightlife', 'sports', 'museums', 'parks', 'events', 'recreation']
            }
        }
    
    def extract_location_from_text(self, text: str) -> str:
        """Extract location from user input using NLP."""
        if not self.nlp:
            return "New York"  # Default fallback
        
        doc = self.nlp(text)
        locations = []
        
        # Extract GPE (Geopolitical entities) and LOC (Locations)
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC', 'FAC']:
                locations.append(ent.text)
        
        # Also look for common location patterns
        location_patterns = [
            r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\bnear\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\baround\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\bclose\s+to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                locations.extend(matches)
        
        # Return the first location found, or default
        if locations:
            return locations[0]
        
        return "New York"  # Default fallback
    
    def fetch_businesses_from_apis(self, query: str, location: str = None) -> List[Dict]:
        """Fetch real business data from public APIs (no API keys required)."""
        businesses = []
        
        try:
            # Using OpenStreetMap Nominatim API (completely free, no API key needed)
            # This is a real public API that doesn't require authentication
            base_url = "https://nominatim.openstreetmap.org/search"
            
            # Use provided location or default
            search_location = location if location else "New York"
            
            params = {
                'q': f"{query} {search_location}",
                'format': 'json',
                'limit': 5,
                'addressdetails': 1
            }
            
            headers = {
                'User-Agent': 'LocalBusinessDiscoveryBot/1.0 (Educational Purpose)'
            }
            response = requests.get(base_url, params=params, headers=headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                for item in data:
                    if 'display_name' in item and 'lat' in item and 'lon' in item:
                        business = {
                            'name': item.get('display_name', 'Unknown'),
                            'address': item.get('display_name', ''),
                            'latitude': float(item.get('lat', 0)),
                            'longitude': float(item.get('lon', 0)),
                            'category': query,
                            'rating': 4.0,  # Default rating since OSM doesn't provide ratings
                            'price_range': '$$',  # Default price range
                            'hours': 'Hours not available',
                            'phone': 'Phone not available',
                            'description': f"Found via OpenStreetMap: {item.get('display_name', '')}",
                            'source': 'OpenStreetMap'
                        }
                        businesses.append(business)
        except Exception as e:
            print(f"API Error: {e}")
            # Log the error for debugging
            import logging
            logging.error(f"Failed to fetch businesses from API: {e}")
        
        return businesses
    
    def _load_sample_businesses(self) -> Dict:
        """Load sample business data (fallback when APIs fail)."""
        return {
            'restaurants': [
                {
                    'name': 'Mario\'s Italian Bistro',
                    'category': 'Italian',
                    'rating': 4.5,
                    'price_range': '$$',
                    'address': '123 Main St, Downtown',
                    'hours': '11:00 AM - 10:00 PM',
                    'phone': '(555) 123-4567',
                    'description': 'Authentic Italian cuisine with fresh pasta and wood-fired pizza'
                },
                {
                    'name': 'The Coffee Bean',
                    'category': 'Coffee Shop',
                    'rating': 4.2,
                    'price_range': '$',
                    'address': '456 Oak Ave, Midtown',
                    'hours': '6:00 AM - 8:00 PM',
                    'phone': '(555) 234-5678',
                    'description': 'Cozy coffee shop with artisanal brews and pastries'
                },
                {
                    'name': 'Sushi Zen',
                    'category': 'Japanese',
                    'rating': 4.7,
                    'price_range': '$$$',
                    'address': '789 Pine St, Uptown',
                    'hours': '5:00 PM - 11:00 PM',
                    'phone': '(555) 345-6789',
                    'description': 'Fresh sushi and sashimi with traditional Japanese atmosphere'
                }
            ],
            'shopping': [
                {
                    'name': 'Fashion Forward',
                    'category': 'Clothing Store',
                    'rating': 4.3,
                    'price_range': '$$',
                    'address': '321 Fashion Blvd, Shopping District',
                    'hours': '10:00 AM - 9:00 PM',
                    'phone': '(555) 456-7890',
                    'description': 'Trendy clothing for men and women'
                },
                {
                    'name': 'Tech World',
                    'category': 'Electronics',
                    'rating': 4.1,
                    'price_range': '$$$',
                    'address': '654 Tech Lane, Business District',
                    'hours': '9:00 AM - 8:00 PM',
                    'phone': '(555) 567-8901',
                    'description': 'Latest electronics and gadgets'
                }
            ],
            'services': [
                {
                    'name': 'Elite Hair Salon',
                    'category': 'Hair Salon',
                    'rating': 4.6,
                    'price_range': '$$',
                    'address': '987 Beauty Ave, Style District',
                    'hours': '9:00 AM - 7:00 PM',
                    'phone': '(555) 678-9012',
                    'description': 'Professional hair styling and coloring services'
                },
                {
                    'name': 'FitLife Gym',
                    'category': 'Fitness Center',
                    'rating': 4.4,
                    'price_range': '$$',
                    'address': '147 Fitness Way, Health District',
                    'hours': '5:00 AM - 11:00 PM',
                    'phone': '(555) 789-0123',
                    'description': 'Full-service gym with personal trainers'
                }
            ],
            'entertainment': [
                {
                    'name': 'Cinema Palace',
                    'category': 'Movie Theater',
                    'rating': 4.2,
                    'price_range': '$$',
                    'address': '258 Entertainment Blvd, Arts District',
                    'hours': '12:00 PM - 12:00 AM',
                    'phone': '(555) 890-1234',
                    'description': 'Latest movies with IMAX and 3D options'
                },
                {
                    'name': 'The Jazz Club',
                    'category': 'Live Music',
                    'rating': 4.8,
                    'price_range': '$$$',
                    'address': '369 Music St, Nightlife District',
                    'hours': '7:00 PM - 2:00 AM',
                    'phone': '(555) 901-2345',
                    'description': 'Intimate jazz performances and craft cocktails'
                }
            ]
        }
    
    def _vectorize_training_data(self):
        """Vectorize the training data for similarity matching."""
        all_patterns = []
        for intent in self.intents:
            all_patterns.extend(intent['patterns'])
        
        self.tfidf_matrix = self.vectorizer.fit_transform(all_patterns)
        self.pattern_to_intent = {}
        
        for intent in self.intents:
            for pattern in intent['patterns']:
                self.pattern_to_intent[pattern] = intent['intent']
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better matching."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities using spaCy."""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities
    
    def get_sentiment(self, text: str) -> str:
        """Analyze sentiment of the text (simplified version)."""
        if not self.nlp:
            return "neutral"
        
        doc = self.nlp(text)
        
        # Simple sentiment analysis based on word patterns
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'sad', 'angry']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"
    
    def classify_intent(self, text: str) -> Tuple[str, float]:
        """Classify the intent of the user input."""
        processed_text = self.preprocess_text(text)
        
        # Vectorize the input text
        text_vector = self.vectorizer.transform([processed_text])
        
        # Calculate similarity with training patterns
        similarities = cosine_similarity(text_vector, self.tfidf_matrix)[0]
        
        # Get the most similar pattern
        max_similarity_idx = np.argmax(similarities)
        max_similarity = similarities[max_similarity_idx]
        
        # Get the corresponding pattern and intent
        all_patterns = []
        for intent in self.intents:
            all_patterns.extend(intent['patterns'])
        
        best_pattern = all_patterns[max_similarity_idx]
        intent = self.pattern_to_intent[best_pattern]
        
        return intent, max_similarity
    
    def search_businesses(self, category: str, filters: Dict = None) -> List[Dict]:
        """Search for businesses in a specific category with optional filters."""
        if category not in self.sample_businesses:
            return []
        
        businesses = self.sample_businesses[category]
        
        if not filters:
            return businesses
        
        # Apply filters
        filtered = businesses.copy()
        
        if 'min_rating' in filters:
            filtered = [b for b in filtered if b['rating'] >= filters['min_rating']]
        
        if 'max_price' in filters:
            price_map = {'$': 1, '$$': 2, '$$$': 3, '$$$$': 4}
            max_price_level = price_map.get(filters['max_price'], 4)
            filtered = [b for b in filtered if price_map.get(b['price_range'], 4) <= max_price_level]
        
        if 'keyword' in filters:
            keyword = filters['keyword'].lower()
            filtered = [b for b in filtered if 
                       keyword in b['name'].lower() or 
                       keyword in b['description'].lower() or
                       keyword in b['category'].lower()]
        
        return filtered
    
    def get_business_recommendations(self, intent: str, user_input: str) -> List[Dict]:
        """Get business recommendations based on intent and user input."""
        recommendations = []
        
        # Extract location from user input
        detected_location = self.extract_location_from_text(user_input)
        
        # Try to get real data from APIs first
        try:
            if intent == 'restaurant_search':
                # Look for cuisine keywords
                cuisine_keywords = ['italian', 'chinese', 'japanese', 'mexican', 'thai', 'indian', 'coffee', 'pizza', 'sushi']
                detected_cuisine = None
                for keyword in cuisine_keywords:
                    if keyword in user_input.lower():
                        detected_cuisine = keyword
                        break
                
                if detected_cuisine:
                    # Try real API first with detected location
                    api_results = self.fetch_businesses_from_apis(f"{detected_cuisine} restaurant", detected_location)
                    if api_results:
                        recommendations = api_results[:2]
                    else:
                        # Fallback to sample data
                        recommendations = [b for b in self.sample_businesses['restaurants'] 
                                        if detected_cuisine in b['category'].lower() or detected_cuisine in b['description'].lower()]
                else:
                    # Try real API for general restaurant search with detected location
                    api_results = self.fetch_businesses_from_apis("restaurant", detected_location)
                    if api_results:
                        recommendations = api_results[:2]
                    else:
                        recommendations = self.sample_businesses['restaurants'][:2]
            
            elif intent == 'business_search':
                # Try real API for general business search with detected location
                api_results = self.fetch_businesses_from_apis("business", detected_location)
                if api_results:
                    recommendations = api_results[:3]
                else:
                    # Fallback to sample data
                    recommendations = []
                    for category in ['restaurants', 'shopping', 'services', 'entertainment']:
                        recommendations.extend(self.sample_businesses[category][:1])
            
            elif intent == 'entertainment_search':
                # Try real API for entertainment with detected location
                api_results = self.fetch_businesses_from_apis("entertainment", detected_location)
                if api_results:
                    recommendations = api_results[:2]
                else:
                    recommendations = self.sample_businesses['entertainment']
            
            elif intent == 'shopping_search':
                # Try real API for shopping with detected location
                api_results = self.fetch_businesses_from_apis("shopping", detected_location)
                if api_results:
                    recommendations = api_results[:2]
                else:
                    recommendations = self.sample_businesses['shopping']
            
            elif intent == 'service_search':
                # Try real API for services with detected location
                api_results = self.fetch_businesses_from_apis("service", detected_location)
                if api_results:
                    recommendations = api_results[:2]
                else:
                    recommendations = self.sample_businesses['services']
        
        except Exception as e:
            print(f"API Error: {e}")
            # Fallback to sample data
            if intent == 'restaurant_search':
                recommendations = self.sample_businesses['restaurants'][:2]
            elif intent == 'entertainment_search':
                recommendations = self.sample_businesses['entertainment']
            elif intent == 'shopping_search':
                recommendations = self.sample_businesses['shopping']
            elif intent == 'service_search':
                recommendations = self.sample_businesses['services']
            else:
                recommendations = []
                for category in ['restaurants', 'shopping', 'services', 'entertainment']:
                    recommendations.extend(self.sample_businesses[category][:1])
        
        return recommendations[:3]  # Return top 3 recommendations
    
    def format_business_response(self, businesses: List[Dict]) -> str:
        """Format business recommendations into a readable response."""
        if not businesses:
            return "I couldn't find any businesses matching your criteria. Try being more specific!"
        
        response = "Here are some great options I found for you:\n\n"
        
        for i, business in enumerate(businesses, 1):
            response += f"**{i}. {business['name']}**\n"
            response += f"   📍 {business['address']}\n"
            response += f"   ⭐ {business['rating']}/5.0 | {business['price_range']}\n"
            response += f"   🕒 {business['hours']}\n"
            response += f"   📞 {business['phone']}\n"
            response += f"   💬 {business['description']}\n\n"
        
        return response
    
    def generate_response(self, text: str) -> Dict:
        """Generate a response to the user input."""
        # Preprocess the input
        processed_text = self.preprocess_text(text)
        
        # Extract entities
        entities = self.extract_entities(text)
        
        # Analyze sentiment
        sentiment = self.get_sentiment(text)
        
        # Classify intent
        intent, confidence = self.classify_intent(text)
        
        # Generate response with real business data
        if confidence > 0.3:  # Threshold for confidence
            # Check if this is a business search intent
            business_intents = ['restaurant_search', 'business_search', 'entertainment_search', 
                              'shopping_search', 'service_search']
            
            if intent in business_intents:
                # Get business recommendations
                recommendations = self.get_business_recommendations(intent, text)
                if recommendations:
                    response = self.format_business_response(recommendations)
                else:
                    response = random.choice(self.responses[intent])
            else:
                response = random.choice(self.responses[intent])
        else:
            # Fallback responses for unclear intents
            fallback_responses = [
                "I'm not sure I understand. Could you rephrase that?",
                "That's interesting! Could you tell me more?",
                "I'm still learning. Could you be more specific?",
                "I'm not quite sure what you mean. Can you clarify?"
            ]
            response = random.choice(fallback_responses)
            intent = "unclear"
        
        return {
            'response': response,
            'intent': intent,
            'confidence': confidence,
            'sentiment': sentiment,
            'entities': entities,
            'processed_text': processed_text
        }
    
    def chat(self, user_input: str) -> str:
        """Main chat function that returns a response."""
        result = self.generate_response(user_input)
        return result['response']
    
    def get_chat_analysis(self, user_input: str) -> Dict:
        """Get detailed analysis of the user input."""
        return self.generate_response(user_input)


def main():
    """Main function to run the local business discovery chatbot in console mode."""
    print("🏪 Welcome to the Local Business Discovery Bot!")
    print("I can help you find restaurants, shops, services, and entertainment in your area.")
    print("Type 'quit' to exit the chat.")
    print("-" * 60)
    
    # Initialize chatbot
    chatbot = LocalBusinessDiscoveryBot()
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
            print("Bot: Goodbye! Hope you found some great local spots!")
            break
        
        if not user_input:
            continue
        
        # Get response
        response = chatbot.chat(user_input)
        print(f"Bot: {response}")


if __name__ == "__main__":
    main()
