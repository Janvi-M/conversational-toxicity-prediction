import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
from collections import Counter
import json
import time
from textblob import TextBlob
import urllib.parse
from urllib.request import urlopen, Request
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree

import nltk

# Download required NLTK data (run once)
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
    
try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')
    
try:
    nltk.data.find('chunkers/maxent_ne_chunker_tab')
except LookupError:
    nltk.download('maxent_ne_chunker_tab')

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

class ImprovedHybridSpikeAnalyzer:
    def __init__(self, guardian_api_key=None):
        """
        Initialize with Guardian API key (free version)
        Get your free API key from: https://open-platform.theguardian.com/access/
        """
        self.guardian_api_key = guardian_api_key
        self.hate_keywords = [
            # Violence & Death
            'hate', 'kill', 'die', 'murder', 'slaughter', 'massacre', 'genocide', 'exterminate',
            'lynch', 'execute', 'shoot', 'stab', 'bomb', 'nuke', 'obliterate', 'wipeout', 'hang',
            # Terrorism / War / Conflict
            'terrorist', 'terror', 'jihadi', 'islamist', 'radical', 'extremist',
            'attack', 'enemy', 'war', 'fight', 'battle', 'combat', 'raid', 'ambush', 'strike',
            'infidel', 'crusader', 'kaffir', 'zionist',
            # Riots / Crime / Disorder
            'antifa', 'riots', 'violence', 'burn', 'looting', 'thugs', 'gangs',
            'criminals', 'felons', 'junkies', 'dealers', 'cartel',
            # Dehumanization
            'animals', 'savages', 'apes', 'monkeys', 'rats', 'vermin', 'parasites', 'plague',
            'cockroaches', 'worms', 'pigs', 'dogs', 'snakes', 'goats', 'scum', 'trash',
            # Immigration / Racist terms
            'illegals', 'invaders', 'aliens', 'outsiders', 'migrants', 'refugees',
            'anchor babies', 'wetbacks', 'chinks', 'gooks', 'slavs', 'gypsies',
            # Disease / Poison Metaphors
            'cancer', 'disease', 'virus', 'infection', 'toxic', 'poison', 'contagion', 'tumor',
            # Insults & Character Attacks
            'lying', 'liar', 'fraud', 'fake', 'corrupt', 'evil', 'sick', 'disgusting',
            'pathetic', 'loser', 'coward', 'traitor', 'backstabber', 'stupid', 'dumb',
            'idiot', 'moron', 'retard', 'degenerate', 'pervert', 'filth',
            # Abbreviations & Slang
            'kkk', 'nazi', 'neo-nazi', 'skinhead', 'heil', 'wpww',
            'islamo', 'sjw', 'cuck', 'libtard', 'snowflake', 'npc',
            'fascist', 'commie', 'marxist', 'pinko',
            # Short forms & Leetspeak
            'h8', 'h8r', 'fuk', 'fck', 'kys', 'stfu', 'gtfo', 'pos', 'smd',
            'lame', 'btch', 'fuq', 'fkn', 'fkin'
        ]
        
        # Enhanced features for better analysis
        self.toxicity_amplifiers = ['always', 'never', 'all', 'every', 'completely', 'totally']
        self.emotional_intensifiers = ['furious', 'enraged', 'disgusted', 'outraged', 'livid']
        
        # Contextual categories for better correlation matching
        self.context_categories = {
            'politics': ['election', 'vote', 'candidate', 'president', 'congress', 'senate', 'government', 
                        'policy', 'democrat', 'republican', 'biden', 'trump', 'politics', 'political'],
            'immigration': ['immigration', 'border', 'migrant', 'refugee', 'asylum', 'deportation',
                          'illegal', 'visa', 'citizen', 'naturalization', 'homeland'],
            'protest': ['protest', 'demonstration', 'riot', 'march', 'rally', 'crowd', 'police',
                       'arrest', 'violence', 'peaceful', 'activist', 'movement'],
            'terrorism': ['terrorist', 'attack', 'bombing', 'security', 'threat', 'extremist',
                         'radical', 'isis', 'al-qaeda', 'homeland', 'fbi'],
            'crime': ['crime', 'murder', 'shooting', 'robbery', 'assault', 'violence', 'criminal',
                     'police', 'arrest', 'investigation', 'victim'],
            'economy': ['economy', 'inflation', 'recession', 'unemployment', 'jobs', 'market',
                       'stocks', 'recession', 'crisis', 'financial'],
            'international': ['war', 'conflict', 'invasion', 'sanctions', 'military', 'defense',
                            'nato', 'china', 'russia', 'ukraine', 'iran', 'israel'],
            'social': ['racism', 'discrimination', 'equality', 'rights', 'justice', 'blm',
                      'lgbtq', 'gender', 'abortion', 'healthcare']
        }
        
    def extract_entities_robust(self, text):
        """Robust entity extraction using multiple methods"""
        entities = set()
        
        if pd.isna(text) or not text:
            return []
        
        text_str = str(text)
        
        try:
            # Method 1: NLTK Named Entity Recognition
            tokens = word_tokenize(text_str)
            pos_tags = pos_tag(tokens)
            chunks = ne_chunk(pos_tags, binary=False)
            
            for chunk in chunks:
                if isinstance(chunk, Tree):
                    entity = ' '.join([token for token, pos in chunk.leaves()])
                    if len(entity) > 2:
                        entities.add(entity.lower())
            
            # Method 2: Extract capitalized words/phrases (likely proper nouns)
            capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text_str)
            for word in capitalized_words:
                if len(word) > 2:
                    entities.add(word.lower())
            
            # Method 3: Extract hashtags and mentions
            hashtags = re.findall(r'#\w+', text_str)
            mentions = re.findall(r'@\w+', text_str)
            
            for hashtag in hashtags:
                entities.add(hashtag.lower())
            for mention in mentions:
                entities.add(mention.lower())
            
            # Method 4: Extract meaningful noun phrases
            stop_words = set(stopwords.words('english'))
            tokens = [word.lower() for word in word_tokenize(text_str) 
                     if word.lower() not in stop_words and len(word) > 2 and word.isalpha()]
            
            # Get bigrams and trigrams that might be entities
            for i in range(len(tokens) - 1):
                bigram = ' '.join(tokens[i:i+2])
                if len(bigram) > 5:
                    entities.add(bigram)
            
            # Method 5: Look for contextual keywords from our categories
            text_lower = text_str.lower()
            for category, keywords in self.context_categories.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        entities.add(keyword)
            
        except Exception as e:
            print(f"Entity extraction error: {e}")
            # Fallback: simple word extraction
            words = re.findall(r'\b[A-Za-z]{3,}\b', text_str)
            entities.update([word.lower() for word in words[:5]])
        
        return list(entities)[:20]  # Limit to top 20 entities
    
    def categorize_content(self, text, entities):
        """Categorize content based on text and entities"""
        if pd.isna(text):
            return 'general'
        
        text_lower = str(text).lower()
        entity_str = ' '.join(entities).lower()
        combined = text_lower + ' ' + entity_str
        
        category_scores = {}
        for category, keywords in self.context_categories.items():
            score = sum(combined.count(keyword) for keyword in keywords)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        return 'general'
    
    def enhanced_hate_detection(self, text, cleaned_text):
        """Enhanced hate detection with multiple features"""
        if pd.isna(text) and pd.isna(cleaned_text):
            return 0.0
            
        text_str = str(text).lower() if not pd.isna(text) else ""
        cleaned_str = str(cleaned_text).lower() if not pd.isna(cleaned_text) else ""
        combined_text = text_str + " " + cleaned_str
        
        # Basic hate keyword count
        hate_score = 0
        for keyword in self.hate_keywords:
            if keyword in combined_text:
                hate_score += combined_text.count(keyword)
        
        # Toxicity amplifiers
        amplifier_score = 0
        for amplifier in self.toxicity_amplifiers:
            if amplifier in combined_text:
                amplifier_score += 0.1
        
        # Emotional intensifiers
        emotion_score = 0
        for intensifier in self.emotional_intensifiers:
            if intensifier in combined_text:
                emotion_score += 0.15
        
        # Caps lock intensity (shouting)
        caps_ratio = sum(1 for c in text_str if c.isupper()) / max(len(text_str), 1)
        caps_score = min(0.2, caps_ratio * 0.5)
        
        # Exclamation marks intensity
        exclamation_count = combined_text.count('!')
        exclamation_score = min(0.15, exclamation_count * 0.05)
        
        # Combine all scores
        total_score = (hate_score * 0.1) + amplifier_score + emotion_score + caps_score + exclamation_score
        return min(1.0, total_score)
    
    def calculate_virality_score(self, row):
        """Calculate how viral a post is"""
        likes = row.get('likes', 0) if not pd.isna(row.get('likes', 0)) else 0
        shares = row.get('shares', 0) if not pd.isna(row.get('shares', 0)) else 0
        comments = row.get('num_comments', 0) if not pd.isna(row.get('num_comments', 0)) else 0
        
        # Weighted engagement score
        engagement = likes + (shares * 3) + (comments * 2)
        return min(1.0, engagement / 1000.0)  # Normalize to 0-1
    
    def detect_spikes(self, df):
        """Enhanced spike detection with multiple metrics"""
        # Create a copy to avoid modifying original
        df = df.copy()
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        
        # Extract entities if not present or improve existing ones
        if 'named_entities' not in df.columns or df['named_entities'].isna().all():
            print("Extracting entities using robust method...")
            df['extracted_entities'] = df['text'].apply(self.extract_entities_robust)
            df['named_entities'] = df['extracted_entities'].apply(lambda x: ', '.join(x))
        else:
            # Improve existing entities
            print("Improving existing entity extraction...")
            df['extracted_entities'] = df.apply(
                lambda row: self.extract_entities_robust(row['text']) if pd.isna(row['named_entities']) 
                else row['named_entities'].split(',') + self.extract_entities_robust(row['text']), 
                axis=1
            )
            df['named_entities'] = df['extracted_entities'].apply(
                lambda x: ', '.join(list(set([e.strip() for e in x if e.strip()])))
            )
        
        # Categorize content
        df['content_category'] = df.apply(
            lambda row: self.categorize_content(row['text'], row['extracted_entities']), 
            axis=1
        )
        
        # Calculate enhanced hate scores
        df['enhanced_hate_score'] = df.apply(
            lambda row: self.enhanced_hate_detection(row.get('text'), row.get('cleaned_text')), 
            axis=1
        )
        
        # Calculate virality scores
        df['virality_score'] = df.apply(self.calculate_virality_score, axis=1)
        
        # Daily aggregation
        daily_metrics = df.groupby('date').agg({
            'sentiment': 'mean',
            'enhanced_hate_score': 'mean',
            'virality_score': 'sum',
            'likes': 'sum',
            'shares': 'sum', 
            'num_comments': 'sum',
            'text': 'count'
        }).rename(columns={'text': 'volume'})
        
        # Add category distribution per day
        category_dist = df.groupby(['date', 'content_category']).size().unstack(fill_value=0)
        for col in category_dist.columns:
            daily_metrics[f'{col}_volume'] = category_dist[col]
        
        # Multiple spike detection criteria
        spike_conditions = []
        sensitivity = 1.2
        
        # Hate sentiment spikes (more negative than usual)
        hate_threshold = daily_metrics['sentiment'].rolling(7).mean() - sensitivity * daily_metrics['sentiment'].rolling(7).std()
        sentiment_spikes = daily_metrics['sentiment'] < hate_threshold
        spike_conditions.append(sentiment_spikes)
        
        # Enhanced hate score spikes
        hate_score_threshold = daily_metrics['enhanced_hate_score'].rolling(7).mean() + sensitivity * daily_metrics['enhanced_hate_score'].rolling(7).std()
        hate_score_spikes = daily_metrics['enhanced_hate_score'] > hate_score_threshold
        spike_conditions.append(hate_score_spikes)
        
        # Volume spikes
        volume_threshold = daily_metrics['volume'].rolling(7).mean() + sensitivity * daily_metrics['volume'].rolling(7).std()
        volume_spikes = daily_metrics['volume'] > volume_threshold
        spike_conditions.append(volume_spikes)
        
        # Virality spikes
        virality_threshold = daily_metrics['virality_score'].rolling(7).mean() + sensitivity * daily_metrics['virality_score'].rolling(7).std()
        virality_spikes = daily_metrics['virality_score'] > virality_threshold
        spike_conditions.append(virality_spikes)
        
        # Combined spike detection (any condition triggers spike)
        spike_days = spike_conditions[0]
        for condition in spike_conditions[1:]:
            spike_days = spike_days | condition
        
        # Store enhanced dataframe for later use
        self.enhanced_df = df
        
        return set(daily_metrics[spike_days].index), daily_metrics
    
    def get_contextual_search_terms(self, spike_data):
        """Generate contextual search terms based on spike characteristics"""
        search_terms = []
        
        # Get dominant content category for the spike
        category_counts = spike_data['content_category'].value_counts()
        if not category_counts.empty:
            dominant_category = category_counts.index[0]
            
            # Add category-specific search terms
            if dominant_category in self.context_categories:
                search_terms.extend(self.context_categories[dominant_category][:3])
        
        # Extract meaningful entities
        all_entities = []
        for entities_str in spike_data['named_entities'].dropna():
            if isinstance(entities_str, str) and entities_str.strip():
                entities = [e.strip().lower() for e in entities_str.split(',') if e.strip()]
                all_entities.extend(entities)
        
        # Filter and rank entities
        entity_counts = Counter(all_entities)
        
        # Add high-frequency, meaningful entities
        for entity, count in entity_counts.most_common(10):
            if (len(entity) > 3 and 
                count >= 2 and  # Must appear at least twice
                entity not in ['news', 'media', 'social', 'post', 'tweet', 'user', 'time', 'people']):
                search_terms.append(entity)
        
        # Add hate-related terms if high hate score
        avg_hate_score = spike_data['enhanced_hate_score'].mean()
        if avg_hate_score > 0.3:
            hate_terms = ['protest', 'violence', 'conflict', 'controversy', 'incident']
            search_terms.extend(hate_terms[:2])
        
        # Ensure we have fallback terms
        if not search_terms:
            search_terms = ['breaking news', 'politics', 'social media', 'controversy']
        
        return list(set(search_terms))[:8]  # Limit to 8 unique terms
    
    def get_guardian_news_for_spike(self, spike_date, spike_data):
        """
        Enhanced Guardian API search with better contextual matching
        """
        if not self.guardian_api_key:
            return ["Guardian API key not provided - cannot search for external causes"]
        
        try:
            # Format date for Guardian API
            date_str = spike_date.strftime('%Y-%m-%d')
            
            # Get contextual search terms based on spike characteristics
            search_terms = self.get_contextual_search_terms(spike_data)
            
            print(f"Searching Guardian for: {search_terms} on {date_str}")
            
            all_articles = []
            
            # Search Guardian API for each term with expanded date range
            date_range = [(spike_date - timedelta(days=1)).strftime('%Y-%m-%d'), 
                         (spike_date + timedelta(days=1)).strftime('%Y-%m-%d')]
            
            for search_term in search_terms:
                try:
                    # Guardian API endpoint
                    url = "https://content.guardianapis.com/search"
                    params = {
                        'api-key': self.guardian_api_key,
                        'from-date': date_range[0],
                        'to-date': date_range[1],
                        'q': search_term,
                        'page-size': 15,
                        'show-fields': 'headline,trailText,bodyText',
                        'order-by': 'relevance'
                    }
                    
                    response = requests.get(url, params=params, timeout=15)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data.get('response', {}).get('status') == 'ok':
                            articles = data.get('response', {}).get('results', [])
                            
                            for article in articles:
                                title = article.get('webTitle', '')
                                trail_text = article.get('fields', {}).get('trailText', '')
                                
                                article_info = {
                                    'title': title,
                                    'url': article.get('webUrl', ''),
                                    'search_term': search_term,
                                    'trail_text': trail_text,
                                    'pub_date': article.get('webPublicationDate', '')
                                }
                                all_articles.append(article_info)
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except requests.RequestException as e:
                    print(f"Error searching for '{search_term}': {str(e)}")
                    continue
            
            # Remove duplicates and sort by relevance
            unique_articles = {}
            for article in all_articles:
                key = article['title'].lower()
                if key not in unique_articles:
                    unique_articles[key] = article
            
            return list(unique_articles.values())[:10]  # Return top 10 unique articles
            
        except Exception as e:
            return [f"Error accessing Guardian API: {str(e)}"]
    
    def contextual_correlation_analysis(self, spike_data, guardian_articles):
        """
        Enhanced correlation analysis that checks for contextual relevance
        """
        if not guardian_articles or isinstance(guardian_articles[0], str) and guardian_articles[0].startswith("Error"):
            return {
                'correlation_found': False,
                'message': 'Unable to access external news sources'
            }
        
        # Extract spike characteristics
        spike_category = spike_data['content_category'].mode().iloc[0] if not spike_data['content_category'].empty else 'general'
        avg_hate_score = spike_data['enhanced_hate_score'].mean()
        
        # Extract entities from spike
        spike_entities = []
        for entities_str in spike_data['named_entities'].dropna():
            if isinstance(entities_str, str):
                entities = [e.strip().lower() for e in entities_str.split(',') if e.strip()]
                spike_entities.extend(entities)
        
        entity_counts = Counter(spike_entities)
        top_spike_entities = [entity for entity, count in entity_counts.most_common(10)]
        
        # Analyze each Guardian article for contextual relevance
        correlations = []
        
        for article in guardian_articles:
            title = article.get('title', '').lower()
            trail_text = article.get('trail_text', '').lower()
            search_term = article.get('search_term', '').lower()
            
            article_content = title + ' ' + trail_text
            
            # Check for category-specific relevance
            category_match = False
            if spike_category in self.context_categories:
                category_keywords = self.context_categories[spike_category]
                category_matches = sum(1 for keyword in category_keywords if keyword in article_content)
                if category_matches >= 2:  # At least 2 category keywords must match
                    category_match = True
            
            # Check for entity relevance
            entity_matches = []
            for entity in top_spike_entities:
                if len(entity) > 3 and entity in article_content:
                    entity_matches.append(entity)
            
            # Check for hate/conflict context if spike has high hate score
            hate_context_match = False
            if avg_hate_score > 0.4:
                hate_indicators = ['protest', 'violence', 'conflict', 'controversy', 'incident', 
                                 'clash', 'riot', 'demonstration', 'unrest', 'tension']
                hate_matches = sum(1 for indicator in hate_indicators if indicator in article_content)
                if hate_matches >= 1:
                    hate_context_match = True
            
            # Determine correlation strength based on multiple factors
            correlation_score = 0
            reasons = []
            
            if category_match:
                correlation_score += 3
                reasons.append(f"Category match ({spike_category})")
            
            if entity_matches:
                correlation_score += len(entity_matches)
                reasons.append(f"Entity matches: {entity_matches[:3]}")
            
            if hate_context_match and avg_hate_score > 0.4:
                correlation_score += 2
                reasons.append("Conflict/hate context match")
            
            if search_term in article_content:
                correlation_score += 1
                reasons.append(f"Search term match ({search_term})")
            
            # Only consider it a meaningful correlation if score >= 2
            if correlation_score >= 2:
                strength = 'high' if correlation_score >= 4 else 'medium'
                correlations.append({
                    'article_title': article.get('title', ''),
                    'article_url': article.get('url', ''),
                    'correlation_score': correlation_score,
                    'strength': strength,
                    'reasons': reasons,
                    'spike_category': spike_category,
                    'spike_hate_score': avg_hate_score
                })
        
        # Sort by correlation score
        correlations.sort(key=lambda x: x['correlation_score'], reverse=True)
        
        if correlations:
            return {
                'correlation_found': True,
                'correlations': correlations,
                'spike_category': spike_category,
                'message': f'Found {len(correlations)} contextually relevant correlations'
            }
        else:
            return {
                'correlation_found': False,
                'message': f'Nothing significant - no contextually relevant news found for {spike_category} spike'
            }
    
    def analyze_internal_causes_enhanced(self, spike_day):
        """Enhanced internal analysis using the stored enhanced dataframe"""
        if not hasattr(self, 'enhanced_df'):
            raise ValueError("Enhanced dataframe not available. Run detect_spikes first.")
        
        spike_data = self.enhanced_df[self.enhanced_df['date'] == spike_day].copy()
        
        # Enhanced entity analysis
        all_entities = []
        for entities_str in spike_data['named_entities'].dropna():
            if isinstance(entities_str, str) and entities_str.strip():
                entities = [e.strip() for e in entities_str.split(',')]
                all_entities.extend(entities)
        
        # Top entities with counts
        top_entities = Counter(all_entities).most_common(15)
        
        # Content categorization
        category_dist = spike_data['content_category'].value_counts()
        
        # Enhanced features for spike data
        spike_data['total_engagement'] = (
            spike_data['likes'].fillna(0) + 
            spike_data['shares'].fillna(0) * 3 + 
            spike_data['num_comments'].fillna(0) * 2
        )
        
        # Most viral posts with enhanced scoring
        viral_posts = spike_data.nlargest(5, 'total_engagement')[
            ['text', 'total_engagement', 'speaker_nm', 'enhanced_hate_score', 
             'virality_score', 'content_category']
        ]
        
        # Most active users with their hate scores and categories
        user_activity = spike_data.groupby('speaker_nm').agg({
            'text': 'count',
            'enhanced_hate_score': 'mean',
            'total_engagement': 'sum',
            'content_category': lambda x: x.mode().iloc[0] if not x.empty else 'general'
        }).rename(columns={'text': 'post_count'}).sort_values('post_count', ascending=False)
        
        # Enhanced hate keyword analysis
        hate_analysis = {}
        total_text = ' '.join(spike_data['cleaned_text'].fillna('').astype(str))
        
        for keyword in self.hate_keywords:
            count = total_text.lower().count(keyword)
            if count > 0:
                hate_analysis[keyword] = count
        
        # Sort by frequency
        hate_analysis = dict(sorted(hate_analysis.items(), key=lambda x: x[1], reverse=True))
        
        # Enhanced emotional metrics
        emotional_metrics = {
            'avg_enhanced_hate_score': spike_data['enhanced_hate_score'].mean(),
            'max_enhanced_hate_score': spike_data['enhanced_hate_score'].max(),
            'high_hate_posts': (spike_data['enhanced_hate_score'] > 0.5).sum(),
            'total_virality_score': spike_data['virality_score'].sum(),
            'dominant_category': category_dist.index[0] if not category_dist.empty else 'general'
        }
        
        return {
            'spike_data': spike_data,  # Return full spike data for correlation analysis
            'top_entities': top_entities,
            'viral_posts': viral_posts.to_dict('records'),
            'top_users': user_activity.head(5).to_dict('index'),
            'hate_keywords': hate_analysis,
            'emotional_metrics': emotional_metrics,
            'category_distribution': category_dist.to_dict(),
            'total_posts': len(spike_data),
            'avg_sentiment': spike_data['sentiment'].mean()
        }
    
    def analyze_spike_with_guardian(self, spike_day, daily_metrics):
        """Enhanced spike analysis with improved Guardian integration"""
        print(f"\n{'='*60}")
        print(f"ENHANCED GUARDIAN SPIKE ANALYSIS: {spike_day}")
        print(f"{'='*60}")
        
        # Internal analysis
        internal = self.analyze_internal_causes_enhanced(spike_day)
        spike_data = internal['spike_data']  # Get the actual spike data
        
        # Search Guardian with contextual terms
        print(f"Searching Guardian API with contextual terms...")
        guardian_articles = self.get_guardian_news_for_spike(spike_day, spike_data)
        
        # Enhanced contextual correlation analysis
        correlation_analysis = self.contextual_correlation_analysis(spike_data, guardian_articles)
        
        # Get daily metrics for this spike
        day_metrics = daily_metrics.loc[spike_day] if spike_day in daily_metrics.index else {}
        
        # Print results
        print(f"INTERNAL ANALYSIS:")
        print(f"   • Total posts: {internal['total_posts']}")
        print(f"   • Dominant category: {internal['emotional_metrics']['dominant_category']}")
        print(f"   • Average sentiment: {internal['avg_sentiment']:.3f}")
        print(f"   • Average hate score: {internal['emotional_metrics']['avg_enhanced_hate_score']:.3f}")
        print(f"   • Top entities: {dict(internal['top_entities'][:5])}")
        print(f"   • Category distribution: {internal['category_distribution']}")
        
        print(f"\nGUARDIAN NEWS SEARCH:")
        if isinstance(guardian_articles, list) and guardian_articles:
            for i, article in enumerate(guardian_articles[:5], 1):
                if isinstance(article, dict):
                    print(f"   {i}. {article.get('title', 'No title')}")
                    print(f"      Search term: {article.get('search_term', 'Unknown')}")
                else:
                    print(f"   {i}. {str(article)}")
        else:
            print(f"   No articles found or error occurred")
        
        print(f"\nCONTEXTUAL CORRELATION ANALYSIS:")
        if correlation_analysis['correlation_found']:
            print(f"   ✅ {correlation_analysis['message']}")
            print(f"   Spike category: {correlation_analysis.get('spike_category', 'Unknown')}")
            for i, corr in enumerate(correlation_analysis['correlations'][:3], 1):
                print(f"   {i}. [{corr['strength'].upper()}] {corr['article_title']}")
                print(f"      Score: {corr['correlation_score']} | Reasons: {', '.join(corr['reasons'])}")
        else:
            print(f"   ❌ {correlation_analysis['message']}")
        
        print(f"\nTOP VIRAL POSTS:")
        for i, post in enumerate(internal['viral_posts'][:3], 1):
            print(f"   {i}. {post['text'][:100]}...")
            print(f"      Category: {post.get('content_category', 'Unknown')} | Engagement: {post['total_engagement']} | Hate: {post['enhanced_hate_score']:.3f}")
        
        return {
            'date': spike_day,
            'internal_analysis': internal,
            'guardian_articles': guardian_articles,
            'correlation_analysis': correlation_analysis,
            'daily_metrics': day_metrics,
            'external_cause_found': correlation_analysis['correlation_found'],
            'contextual_relevance': correlation_analysis.get('spike_category', 'general')
        }
    
    def enhance_features_for_tgn(self, spike_days, daily_metrics, spike_analyses=None):
        """Enhanced feature engineering specifically for TGN construction"""
        if not hasattr(self, 'enhanced_df'):
            raise ValueError("Enhanced dataframe not available. Run detect_spikes first.")
        
        df = self.enhanced_df.copy()
        
        # Basic spike features
        df['spike_day'] = df['date'].isin(spike_days).astype(int)
        
        # Enhanced virality and engagement features (already calculated in detect_spikes)
        df['engagement_rank'] = df['virality_score'].rank(pct=True)
        
        # Hate intensity (already calculated as enhanced_hate_score)
        df['hate_intensity'] = df['enhanced_hate_score']
        
        # Initialize new columns properly with proper defaults
        df['hate_keywords_count'] = 0
        df['reply_chain_depth'] = 0
        df['spike_cause'] = 'normal_activity'
        df['spike_top_entities'] = ''
        df['spike_external_cause'] = 'no_spike'
        df['contextual_relevance'] = 'none'
        
        # Create a mapping from spike_day to analysis results
        spike_info_map = {}
        if spike_analyses:
            for analysis in spike_analyses:
                spike_date = analysis['date']
                
                # Build comprehensive external cause string
                if analysis['external_cause_found']:
                    # Get top correlations
                    correlations = analysis['correlation_analysis'].get('correlations', [])
                    if correlations:
                        top_corr = correlations[0]
                        external_cause = f"Guardian correlation: {top_corr['article_title'][:100]} (score: {top_corr['correlation_score']}, strength: {top_corr['strength']})"
                    else:
                        external_cause = f"Guardian correlation: {analysis['correlation_analysis']['message']}"
                    contextual_relevance = analysis.get('contextual_relevance', 'general')
                else:
                    external_cause = analysis['correlation_analysis']['message']
                    contextual_relevance = 'no_correlation'
                
                # Get top entities
                top_entities_str = ', '.join([f"{entity}({count})" 
                                            for entity, count in analysis['internal_analysis']['top_entities'][:5]])
                
                spike_info_map[spike_date] = {
                    'external_cause': external_cause,
                    'contextual_relevance': contextual_relevance,
                    'top_entities': top_entities_str
                }
        
        # Now apply the spike information to all rows with matching dates
        for spike_day in spike_days:
            spike_mask = df['date'] == spike_day
            
            if spike_mask.any():
                # Get spike info if available
                if spike_day in spike_info_map:
                    info = spike_info_map[spike_day]
                    df.loc[spike_mask, 'spike_external_cause'] = info['external_cause']
                    df.loc[spike_mask, 'contextual_relevance'] = info['contextual_relevance']
                    df.loc[spike_mask, 'spike_top_entities'] = info['top_entities']
                
                # Determine primary cause from metrics
                day_metrics = daily_metrics.loc[spike_day] if spike_day in daily_metrics.index else None
                
                if day_metrics is not None:
                    causes = []
                    if day_metrics.get('enhanced_hate_score', 0) > 0.3:
                        causes.append('high_hate_content')
                    if day_metrics.get('volume', 0) > daily_metrics['volume'].mean() + daily_metrics['volume'].std():
                        causes.append('volume_spike')
                    if day_metrics.get('virality_score', 0) > daily_metrics['virality_score'].mean() + daily_metrics['virality_score'].std():
                        causes.append('viral_content')
                    
                    primary_cause = '_'.join(causes) if causes else 'activity_spike'
                    df.loc[spike_mask, 'spike_cause'] = primary_cause
        
        # Hate keyword count (for backward compatibility)
        for idx in df.index:
            row = df.loc[idx]
            text = str(row.get('cleaned_text', '')).lower()
            count = sum(text.count(keyword) for keyword in self.hate_keywords)
            df.loc[idx, 'hate_keywords_count'] = count
        
        # Reply chain analysis
        df['is_reply'] = df['reply_to_nm'].notna().astype(int)
        
        # Calculate actual reply depth where possible
        conversation_depths = {}
        for idx in df.index:
            row = df.loc[idx]
            conv_id = row.get('conversation_id', row.get('id'))
            if pd.notna(row.get('reply_to_nm')):
                conversation_depths[row['id']] = conversation_depths.get(conv_id, 0) + 1
            else:
                conversation_depths[row['id']] = 0
        
        for idx in df.index:
            df.loc[idx, 'reply_chain_depth'] = conversation_depths.get(df.loc[idx, 'id'], 0)
        
        # User influence metrics
        user_metrics = df.groupby('speaker_nm').agg({
            'virality_score': 'mean',
            'hate_intensity': 'mean',
            'likes': 'sum',
            'shares': 'sum',
            'num_comments': 'sum'
        })
        
        user_metrics['influence_score'] = (
            0.4 * user_metrics['virality_score'] + 
            0.3 * user_metrics['likes'] / (user_metrics['likes'].max() + 1) +
            0.3 * (user_metrics['shares'] + user_metrics['num_comments']) / 
                  (user_metrics['shares'] + user_metrics['num_comments']).max()
        )
        
        df['user_influence'] = df['speaker_nm'].map(user_metrics['influence_score']).fillna(0)
        
        # Network centrality approximation
        df['mention_count'] = df['text'].astype(str).str.count('@')
        df['hashtag_count'] = df['text'].astype(str).str.count('#')
        df['url_count'] = df['text'].astype(str).str.count('http')
        
        # Conversation centrality (how connected a post is)
        conv_sizes = df.groupby('conversation_id').size()
        df['conversation_size'] = df['conversation_id'].map(conv_sizes)
        df['conversation_centrality'] = df['conversation_size'] / df['conversation_size'].max()
        
        return df
    
    def run_guardian_spike_analysis(self, df, output_file=None):
        """Main analysis pipeline with enhanced Guardian API integration"""
        print("Starting Enhanced Guardian-Powered Spike Analysis...")
        print("Detecting spikes with improved entity extraction and categorization...")
        
        # Enhanced spike detection (this populates self.enhanced_df)
        spike_days, daily_metrics = self.detect_spikes(df)
        print(f"Detected {len(spike_days)} spike days: {sorted(spike_days)}")
        
        if not spike_days:
            print("No spike days detected. Analysis complete.")
            return df, []
        
        # Analyze each spike day with Guardian API FIRST
        spike_analyses = []
        print(f"Analyzing {len(spike_days)} spike days with enhanced Guardian API integration...")
        
        for i, spike_day in enumerate(sorted(spike_days), 1):
            print(f"\n[{i}/{len(spike_days)}] Analyzing spike: {spike_day}")
            analysis = self.analyze_spike_with_guardian(spike_day, daily_metrics)
            spike_analyses.append(analysis)
        
        # NOW enhance features for TGN with spike analysis results
        print(f"\nEngineering enhanced features for TGN construction with Guardian data...")
        df_enhanced = self.enhance_features_for_tgn(spike_days, daily_metrics, spike_analyses)
        
        # Enhanced summary statistics
        print(f"\nENHANCED GUARDIAN ANALYSIS SUMMARY:")
        print(f"   • Total spike days analyzed: {len(spike_days)}")
        
        correlations_found = sum(1 for a in spike_analyses if a['external_cause_found'])
        nothing_significant = len(spike_analyses) - correlations_found
        
        print(f"   • Spikes with contextual Guardian correlations: {correlations_found}")
        print(f"   • Spikes with 'Nothing significant': {nothing_significant}")
        
        # Category breakdown
        category_breakdown = {}
        for analysis in spike_analyses:
            category = analysis.get('contextual_relevance', 'unknown')
            category_breakdown[category] = category_breakdown.get(category, 0) + 1
        
        print(f"   • Category breakdown: {category_breakdown}")
        print(f"   • Enhanced dataset: {len(df_enhanced)} rows, {len(df_enhanced.columns)} columns")
        print(f"   • Posts during spikes: {df_enhanced['spike_day'].sum()}")
        
        # Verify that spike_external_cause has been populated
        spike_rows = df_enhanced[df_enhanced['spike_day'] == 1]
        non_default_causes = spike_rows[spike_rows['spike_external_cause'] != 'no_spike']
        print(f"   • Spike rows with Guardian data: {len(non_default_causes)}/{len(spike_rows)}")
        
        # Save results
        if output_file:
            # Create serializable version of spike analyses
            serializable_analyses = []
            for analysis in spike_analyses:
                serializable_analysis = analysis.copy()
                # Remove non-serializable DataFrame
                if 'internal_analysis' in serializable_analysis and 'spike_data' in serializable_analysis['internal_analysis']:
                    del serializable_analysis['internal_analysis']['spike_data']
                serializable_analyses.append(serializable_analysis)
            
            with open(output_file, 'w') as f:
                json.dump(serializable_analyses, f, indent=2, default=str)
            print(f"Guardian analysis saved to: {output_file}")
        
        return df_enhanced, spike_analyses

# =============================================================================
# CONFIGURATION SECTION - UPDATE THESE VALUES
# =============================================================================

# 1. GUARDIAN API KEY (Get free key from: https://open-platform.theguardian.com/access/)
GUARDIAN_API_KEY = "..."  # Replace with your Guardian API key

# 2. FILE PATHS
INPUT_CSV_FILE = "cap_processed_tweets.csv"  # Your input CSV file path
OUTPUT_CSV_FILE = "abc.csv"  # Enhanced CSV for TGN
ANALYSIS_REPORT = "abc.json"  # Detailed analysis

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Starting Enhanced Guardian-Powered Hate Speech Spike Analysis")
    print("=" * 70)
    
    # Initialize analyzer with Guardian API
    analyzer = ImprovedHybridSpikeAnalyzer(guardian_api_key=GUARDIAN_API_KEY)
    
    # Check API key
    if GUARDIAN_API_KEY == "your_guardian_api_key_here":
        print("WARNING: Guardian API key not configured!")
        print("Get your free API key from: https://open-platform.theguardian.com/access/")
        print("Will proceed with internal analysis only...")
    
    # Load data
    try:
        print(f"Loading data from: {INPUT_CSV_FILE}")
        df = pd.read_csv(INPUT_CSV_FILE)
        print(f"Successfully loaded {len(df)} rows with {len(df.columns)} columns")
        print(f"   Columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"File {INPUT_CSV_FILE} not found!")
        print("Creating enhanced sample data for demonstration...")
        
        # Create sample data with realistic spike scenarios
        import random
        from datetime import datetime, timedelta
        
        sample_size = 400
        start_date = datetime(2023, 6, 1)
        
        # Create realistic spike scenarios with contextual themes
        spike_scenarios = [
            {
                'date': datetime(2023, 6, 5),  # Political controversy spike
                'category': 'politics',
                'entities': ['Biden', 'Trump', 'Congress', 'Election', 'Policy', 'Government'],
                'hate_level': 0.7,
                'volume_multiplier': 3,
                'posts': [
                    "These politicians are complete liars and corrupt idiots!",
                    "This disgusting government policy will destroy everything!",
                    "Political corruption and lies everywhere, I hate this system",
                    "Congressional hearing reveals shocking government cover-up",
                    "Biden Trump debate highlights deep political divisions",
                ]
            },
            {
                'date': datetime(2023, 6, 12),  # Immigration policy spike
                'category': 'immigration',
                'entities': ['Border', 'Immigration', 'Migrants', 'Policy', 'Homeland', 'Deportation'],
                'hate_level': 0.8,
                'volume_multiplier': 4,
                'posts': [
                    "These illegal aliens are invading our country like animals!",
                    "Border crisis getting worse, migrants flooding in",
                    "Immigration policy debate intensifies in Congress",
                    "Homeland Security announces new deportation measures",
                    "Refugee families seek asylum at southern border",
                ]
            },
            {
                'date': datetime(2023, 6, 18),  # Protest/social unrest spike
                'category': 'protest',
                'entities': ['Protest', 'Police', 'Violence', 'Demonstration', 'Rights', 'Justice'],
                'hate_level': 0.6,
                'volume_multiplier': 3.5,
                'posts': [
                    "These violent riots and thugs are destroying our cities!",
                    "Peaceful protest march demands police reform and justice",
                    "Demonstration turns violent as police clash with crowds",
                    "Rights activists organize peaceful solidarity rally",
                    "Justice department investigates police violence incident",
                ]
            },
            {
                'date': datetime(2023, 6, 25),  # International conflict spike
                'category': 'international',
                'entities': ['Ukraine', 'Russia', 'War', 'Military', 'NATO', 'Sanctions'],
                'hate_level': 0.5,
                'volume_multiplier': 2.5,
                'posts': [
                    "This war is terrible, innocent people suffering",
                    "Ukraine military receives new NATO weapons shipment",
                    "Russia faces additional economic sanctions from Europe",
                    "International community condemns latest military strikes",
                    "War refugees continue fleeing across European borders",
                ]
            }
        ]
        
        all_data = []
        
        # Generate enhanced sample data
        for i in range(sample_size):
            base_date = start_date + timedelta(hours=random.randint(0, 720))
            
            # Check if this falls on a spike day
            spike_info = None
            for scenario in spike_scenarios:
                if abs((base_date.date() - scenario['date'].date()).days) == 0:
                    spike_info = scenario
                    break
            
            if spike_info and random.random() < 0.4:  # 40% chance for spike content
                # Generate spike-related content
                text = random.choice(spike_info['posts'])
                entities = random.sample(spike_info['entities'], random.randint(2, 4))
                sentiment = random.uniform(-0.9, -0.2) if spike_info['hate_level'] > 0.6 else random.uniform(-0.7, 0.1)
                likes = random.randint(100, 1500) * spike_info['volume_multiplier']
                shares = random.randint(20, 300) * spike_info['volume_multiplier']
                comments = random.randint(10, 200) * spike_info['volume_multiplier']
            else:
                # Generate normal content
                normal_posts = [
                    "Beautiful weather today, perfect for outdoor activities",
                    "Enjoying quality time with family and friends this weekend",
                    "Looking forward to upcoming vacation and holiday plans",
                    "Morning commute was surprisingly smooth and pleasant today",
                    "Watching interesting documentary about nature and wildlife",
                    "This new restaurant downtown has incredible food and service",
                    "Regular daily discussion about work and personal life",
                    "Sharing positive thoughts about community events and activities"
                ]
                text = random.choice(normal_posts)
                entities = random.choices(['Weather', 'Family', 'Food', 'Travel', 'Entertainment', 'Community'], k=random.randint(1, 2))
                sentiment = random.uniform(-0.2, 0.9)
                likes = random.randint(0, 150)
                shares = random.randint(0, 30)
                comments = random.randint(0, 50)
            
            # Create cleaned text with more realistic preprocessing
            cleaned_text = re.sub(r'[^\w\s]', '', text.lower())
            cleaned_text = ' '.join([word for word in cleaned_text.split() 
                                   if len(word) > 2 and word not in 
                                   ['the', 'and', 'this', 'that', 'with', 'for', 'are', 'was', 'were', 'been', 'have', 'has']])
            
            all_data.append({
                'id': i + 1,
                'conversation_id': max(1, i + 1 - random.randint(0, 12)),
                'timestamp': base_date,
                'speaker_nm': f"User_{random.randint(1, 100)}",
                'reply_to_nm': f"User_{random.randint(1, 100)}" if random.random() > 0.6 else None,
                'text': text,
                'cleaned_text': cleaned_text,
                'sentiment': sentiment,
                'likes': likes,
                'shares': shares,
                'num_comments': comments,
                'named_entities': ', '.join(entities) if entities else ''
            })
        
        df = pd.DataFrame(all_data)
        print(f"Created enhanced realistic sample dataset with {len(df)} rows")
        print(f"Spike scenarios with contextual themes: {len(spike_scenarios)} potential spike days")
        print(f"Categories: {[s['category'] for s in spike_scenarios]}")
    
    # Run enhanced Guardian-powered analysis
    print(f"\nRunning Enhanced Guardian API Analysis Pipeline...")
    enhanced_df, spike_analyses = analyzer.run_guardian_spike_analysis(df, ANALYSIS_REPORT)
    
    # Detailed results analysis with contextual information
    print(f"\nDETAILED CONTEXTUAL RESULTS ANALYSIS:")
    
    if spike_analyses:
        correlations_found = sum(1 for analysis in spike_analyses if analysis['external_cause_found'])
        nothing_significant = len(spike_analyses) - correlations_found
        
        print(f"   • Total spike days analyzed: {len(spike_analyses)}")
        print(f"   • Guardian correlations found: {correlations_found}")
        print(f"   • 'Nothing significant' cases: {nothing_significant}")
        
        # Contextual analysis
        context_analysis = {}
        for analysis in spike_analyses:
            context = analysis.get('contextual_relevance', 'unknown')
            status = 'CORRELATED' if analysis['external_cause_found'] else 'NO_CORRELATION'
            
            if context not in context_analysis:
                context_analysis[context] = {'correlated': 0, 'no_correlation': 0}
            
            if status == 'CORRELATED':
                context_analysis[context]['correlated'] += 1
            else:
                context_analysis[context]['no_correlation'] += 1
        
        print(f"\nCONTEXTUAL CORRELATION BREAKDOWN:")
        for context, counts in context_analysis.items():
            total = counts['correlated'] + counts['no_correlation']
            correlation_rate = (counts['correlated'] / total * 100) if total > 0 else 0
            print(f"   • {context}: {counts['correlated']}/{total} correlated ({correlation_rate:.1f}%)")
        
        print(f"\nSPIKE-BY-SPIKE DETAILED SUMMARY:")
        for i, analysis in enumerate(spike_analyses, 1):
            status = "CONTEXTUALLY CORRELATED" if analysis['external_cause_found'] else "NOTHING SIGNIFICANT"
            context = analysis.get('contextual_relevance', 'unknown')
            print(f"   {i}. {analysis['date']} [{context}] - {status}")
            
            if analysis['external_cause_found']:
                corr_count = len(analysis['correlation_analysis'].get('correlations', []))
                print(f"      Found {corr_count} contextually relevant Guardian correlations")
                if analysis['correlation_analysis'].get('correlations'):
                    top_corr = analysis['correlation_analysis']['correlations'][0]
                    print(f"      Top match: {top_corr['article_title'][:60]}... (score: {top_corr['correlation_score']})")
            else:
                print(f"      {analysis['correlation_analysis']['message']}")
    else:
        print("   • No spikes detected in the dataset")
    
    # Verify enhanced TGN-ready features
    print(f"\nVERIFYING ENHANCED TGN-READY FEATURES:")
    tgn_features = [
        'hate_intensity', 'virality_score', 'spike_day', 'user_influence',
        'reply_chain_depth', 'conversation_centrality', 'engagement_rank',
        'spike_cause', 'spike_top_entities', 'spike_external_cause',
        'contextual_relevance', 'content_category'
    ]
    
    missing_features = []
    for feature in tgn_features:
        if feature in enhanced_df.columns:
            print(f"   • {feature}: Available")
            if enhanced_df[feature].dtype in ['int64', 'float64']:
                print(f"      Range: {enhanced_df[feature].min():.3f} to {enhanced_df[feature].max():.3f}")
            else:
                unique_vals = enhanced_df[feature].nunique()
                print(f"      Unique values: {unique_vals}")
                if feature in ['contextual_relevance', 'content_category', 'spike_external_cause']:
                    samples = enhanced_df[feature].value_counts().head(3)
                    print(f"      Top values: {dict(samples)}")
        else:
            missing_features.append(feature)
            print(f"   • {feature}: MISSING")
    
    if missing_features:
        print(f"\nMissing features for TGN: {missing_features}")
    else:
        print(f"\nAll enhanced TGN features successfully created!")
    
    # Save enhanced results
    enhanced_df.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"\nFiles saved:")
    print(f"   • Enhanced Guardian CSV: {OUTPUT_CSV_FILE}")
    print(f"   • Contextual Analysis Report: {ANALYSIS_REPORT}")
    
    # Enhanced sample data preview
    print(f"\nSAMPLE ENHANCED DATA WITH CONTEXTUAL CORRELATION:")
    sample_cols = ['date', 'content_category', 'hate_intensity', 'spike_day', 'contextual_relevance', 'spike_external_cause']
    available_cols = [col for col in sample_cols if col in enhanced_df.columns]
    
    if available_cols:
        # Show correlated spike examples first
        correlated_spikes = enhanced_df[
            (enhanced_df['spike_day'] == 1) & 
            (enhanced_df['contextual_relevance'] != 'no_correlation')
        ][available_cols].head(3)
        
        nothing_significant = enhanced_df[
            (enhanced_df['spike_day'] == 1) & 
            (enhanced_df['contextual_relevance'] == 'no_correlation')
        ][available_cols].head(2)
        
        normal_examples = enhanced_df[enhanced_df['spike_day'] == 0][available_cols].head(2)
        
        if not correlated_spikes.empty:
            print("\n   CONTEXTUALLY CORRELATED SPIKE EXAMPLES:")
            print(correlated_spikes.to_string(index=False))
        
        if not nothing_significant.empty:
            print("\n   'NOTHING SIGNIFICANT' SPIKE EXAMPLES:")
            print(nothing_significant.to_string(index=False))
        
        if not normal_examples.empty:
            print("\n   NORMAL DAY EXAMPLES:")
            print(normal_examples.to_string(index=False))
    
    # Enhanced API Usage Summary
    print(f"\nENHANCED GUARDIAN API INTEGRATION SUMMARY:")
    if GUARDIAN_API_KEY and GUARDIAN_API_KEY != "your_guardian_api_key_here":
        print(f"   • API Key: Configured and active")
        print(f"   • Contextual search terms: Generated based on spike characteristics")
        print(f"   • Content categorization: Automatic classification into context categories")
        print(f"   • Correlation analysis: Multi-factor contextual relevance scoring")
        print(f"   • Date range expansion: ±1 day for better news coverage")
        print(f"   • Rate limiting: Implemented (0.1s delays)")
    else:
        print(f"   • API Key: Not configured - using internal analysis only")
    
    print(f"\nENHANCED PIPELINE FEATURES:")
    print(f"   • Robust entity extraction with multiple NLP methods")
    print(f"   • Content categorization into {len(analyzer.context_categories)} categories")
    print(f"   • Contextual Guardian search based on spike characteristics")
    print(f"   • Multi-factor correlation scoring (category, entities, hate context)")
    print(f"   • 'Nothing significant' only when no contextual relevance found")
    print(f"   • Enhanced TGN features with contextual metadata")
    
    print(f"\n" + "="*70)
    print(f"ENHANCED GUARDIAN-POWERED PIPELINE COMPLETE!")
    print(f"   • Contextually aware hate speech spike analysis")
    print(f"   • Intelligent Guardian news correlation with relevance scoring") 
    print(f"   • Ready for advanced TGN construction with contextual features")
    print(f"   • Improved accuracy in identifying meaningful external causes")
    print(f"="*70)
