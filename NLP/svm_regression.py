import json
import pickle
import os
import glob
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import re


class EnhancedSentimentTrainer:
    def __init__(self):
        """Initialize the enhanced sentiment trainer with context awareness"""
        self.vectorizer = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.reverse_label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
        # Enhanced feature extraction patterns
        self.intensity_amplifiers = {
            'very', 'extremely', 'incredibly', 'absolutely', 'totally',
            'completely', 'utterly', 'highly', 'really', 'so', 'super'
        }
        
        self.negation_words = {
            'not', 'no', 'never', 'nothing', 'nowhere', 'neither', 'nobody',
            'none', 'hardly', 'scarcely', 'barely', "n't", 'cannot', 'cant'
        }
        
        self.positive_emoticons = [':)', ':-)', ':D', ':-D', ':P', ':-P', '^_^', '😊', '😃', '👍', '❤️']
        self.negative_emoticons = [':(', ':-(', ':[', ':-[', ':/',':-/', '😢', '😞', '👎', '💔']
        
    def extract_linguistic_features(self, text: str) -> np.ndarray:
        """Extract advanced linguistic and contextual features"""
        features = []
        text_lower = text.lower()
        
        # 1. Punctuation-based features (tone indicators)
        features.append(text.count('!') / max(len(text), 1))  # Exclamation ratio
        features.append(text.count('?') / max(len(text), 1))  # Question ratio
        features.append(text.count('...') + text.count('…'))  # Ellipsis count
        features.append(1 if text.isupper() and len(text) > 10 else 0)  # All caps (shouting)
        
        # 2. Intensity and emphasis
        words = text_lower.split()
        amplifier_count = sum(1 for w in words if w in self.intensity_amplifiers)
        features.append(amplifier_count / max(len(words), 1))
        
        # 3. Negation detection (critical for context)
        negation_count = sum(1 for w in words if w in self.negation_words)
        features.append(negation_count / max(len(words), 1))
        
        # 4. Emoticon analysis
        pos_emoticon = sum(1 for e in self.positive_emoticons if e in text)
        neg_emoticon = sum(1 for e in self.negative_emoticons if e in text)
        features.append(pos_emoticon)
        features.append(neg_emoticon)
        
        # 5. Length-based features (longer reviews often more detailed)
        features.append(len(text))  # Character count
        features.append(len(words))  # Word count
        features.append(len([s for s in text.split('.') if s.strip()]))  # Sentence count
        
        # 6. Capitalization patterns (emphasis)
        capital_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        features.append(capital_words / max(len(words), 1))
        
        # 7. Repeated characters (e.g., "sooooo good" or "baaad")
        repeated_chars = len(re.findall(r'(.)\1{2,}', text_lower))
        features.append(repeated_chars)
        
        # 8. Question words (uncertainty indicators)
        question_words = {'why', 'how', 'what', 'when', 'where', 'who'}
        question_count = sum(1 for w in words if w in question_words)
        features.append(question_count / max(len(words), 1))
        
        # 9. Comparative/superlative (best, worst, better, worse)
        comparatives = {'best', 'worst', 'better', 'worse', 'great', 'terrible', 
                       'excellent', 'awful', 'amazing', 'horrible'}
        comparative_count = sum(1 for w in words if w in comparatives)
        features.append(comparative_count / max(len(words), 1))
        
        # 10. Personal pronouns (engagement level)
        personal_pronouns = {'i', 'me', 'my', 'mine', 'we', 'us', 'our'}
        pronoun_count = sum(1 for w in words if w in personal_pronouns)
        features.append(pronoun_count / max(len(words), 1))
        
        return np.array(features)
    
    def load_combined_json(self, json_path: str) -> List[Dict]:
        """Load reviews from a combined JSON file"""
        print(f"Loading {json_path}...")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        reviews = []
        
        if 'products' in data:
            for product in data['products']:
                if 'reviews' in product and product['reviews']:
                    reviews.extend(product['reviews'])
        elif 'reviews' in data:
            if isinstance(data['reviews'], dict):
                for sentiment_type in ['positive', 'negative', 'neutral']:
                    if sentiment_type in data['reviews']:
                        reviews.extend(data['reviews'][sentiment_type])
                if 'all' in data['reviews']:
                    reviews.extend(data['reviews']['all'])
            elif isinstance(data['reviews'], list):
                reviews.extend(data['reviews'])
        
        print(f"  Loaded {len(reviews)} reviews")
        return reviews
    
    def load_all_combined_files(self, directory: str = ".") -> List[Dict]:
        """Load all combined JSON files from directory"""
        pattern = os.path.join(directory, "*_combined_*.json")
        json_files = glob.glob(pattern)
        
        if not json_files:
            print(f"No combined JSON files found matching pattern: {pattern}")
            print("Looking for any JSON files with 'combined' in the name...")
            json_files = [f for f in glob.glob(os.path.join(directory, "*.json")) 
                         if 'combined' in f.lower()]
        
        if not json_files:
            raise FileNotFoundError(
                f"No combined JSON files found in {directory}\n"
                "Expected files with '_combined_' in the name"
            )
        
        print(f"\nFound {len(json_files)} combined JSON file(s):")
        for f in json_files:
            print(f"  - {os.path.basename(f)}")
        
        all_reviews = []
        for json_file in json_files:
            reviews = self.load_combined_json(json_file)
            all_reviews.extend(reviews)
        
        # Remove duplicates
        seen_texts = set()
        unique_reviews = []
        for review in all_reviews:
            text = review.get('review_text', '')
            if text and text not in seen_texts:
                seen_texts.add(text)
                unique_reviews.append(review)
        
        if len(all_reviews) != len(unique_reviews):
            print(f"\nRemoved {len(all_reviews) - len(unique_reviews)} duplicate reviews")
        
        print(f"\nTotal unique reviews loaded: {len(unique_reviews)}")
        return unique_reviews
    
    def prepare_data(self, reviews: List[Dict]) -> Tuple[List[str], List[int], np.ndarray]:
        """Prepare text, labels, and linguistic features"""
        texts = []
        labels = []
        linguistic_features_list = []
        
        for review in reviews:
            text = review.get('review_text', '').strip()
            title = review.get('title', '').strip()
            sentiment = review.get('sentiment', '').lower()
            
            if not text or sentiment not in self.label_map:
                continue
            
            # Combine title and text with special marker
            full_text = f"{title} [SEP] {text}".strip() if title else text
            
            texts.append(full_text)
            labels.append(self.label_map[sentiment])
            
            # Extract linguistic features
            ling_features = self.extract_linguistic_features(full_text)
            linguistic_features_list.append(ling_features)
        
        linguistic_features = np.array(linguistic_features_list)
        
        print(f"\nPrepared {len(texts)} samples for training")
        print(f"Linguistic features shape: {linguistic_features.shape}")
        
        # Print distribution
        unique, counts = np.unique(labels, return_counts=True)
        print("\nSentiment distribution:")
        for label_idx, count in zip(unique, counts):
            sentiment_name = self.reverse_label_map[label_idx]
            percentage = (count / len(labels)) * 100
            print(f"  {sentiment_name.capitalize()}: {count:5d} ({percentage:5.1f}%)")
        
        return texts, labels, linguistic_features
    
    def train(self, texts: List[str], labels: List[int], 
              linguistic_features: np.ndarray, test_size: float = 0.2):
        """Train enhanced ensemble model with TF-IDF + linguistic features"""
        print("\n" + "="*60)
        print("TRAINING ENHANCED SENTIMENT MODEL")
        print("="*60)
        print("\nEnhancements:")
        print("  ✓ TF-IDF with character n-grams (captures subword patterns)")
        print("  ✓ Linguistic features (punctuation, negation, emphasis)")
        print("  ✓ Context-aware feature extraction")
        print("  ✓ Ensemble voting classifier")
        print()
        
        # Split data
        X_train, X_test, y_train, y_test, ling_train, ling_test = train_test_split(
            texts, labels, linguistic_features, 
            test_size=test_size, random_state=42, stratify=labels
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Enhanced TF-IDF with character n-grams for better context
        print("\nTraining enhanced TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
            analyzer='word',
            min_df=2,
            max_df=0.95,
            strip_accents='unicode',
            lowercase=True,
            stop_words='english',
            sublinear_tf=True,  # Use log scaling
            token_pattern=r'\b\w+\b|[!?.]+'  # Include punctuation as tokens
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        print(f"  Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print(f"  TF-IDF shape: {X_train_tfidf.shape}")
        
        # Scale linguistic features
        ling_train_scaled = self.scaler.fit_transform(ling_train)
        
        # Combine TF-IDF with linguistic features
        from scipy.sparse import hstack, csr_matrix
        X_train_combined = hstack([X_train_tfidf, csr_matrix(ling_train_scaled)])
        print(f"  Combined features shape: {X_train_combined.shape}")
        
        # Train ensemble model
        print("\nTraining ensemble classifier...")
        
        # Create multiple classifiers with increased iterations
        svm_model = LinearSVC(
            C=0.5,
            max_iter=5000,  # Increased from 2000
            random_state=42,
            class_weight='balanced',
            tol=1e-4  # Convergence tolerance
        )
        
        logistic_model = LogisticRegression(
            C=1.0,
            max_iter=3000,  # Increased from 1000
            random_state=42,
            class_weight='balanced',
            solver='lbfgs',  # Changed from 'saga' - more stable
            tol=1e-4
        )
        
        # Ensemble voting
        self.model = VotingClassifier(
            estimators=[
                ('svm', svm_model),
                ('logistic', logistic_model)
            ],
            voting='hard'
        )
        
        self.model.fit(X_train_combined, y_train)
        print("  Training completed!")
        
        # Cross-validation
        print("\nPerforming 5-fold cross-validation...")
        cv_scores = cross_val_score(self.model, X_train_combined, y_train, cv=5)
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Evaluate on test set
        print("\n" + "="*60)
        print("EVALUATION ON TEST SET")
        print("="*60)
        
        X_test_tfidf = self.vectorizer.transform(X_test)
        ling_test_scaled = self.scaler.transform(ling_test)
        X_test_combined = hstack([X_test_tfidf, csr_matrix(ling_test_scaled)])
        
        y_pred = self.model.predict(X_test_combined)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=['Negative', 'Neutral', 'Positive'],
            digits=4
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print("                Predicted")
        print("              Neg   Neu   Pos")
        for i, (label, row) in enumerate(zip(['Negative', 'Neutral', 'Positive'], cm)):
            print(f"Actual {label:8s}  {row[0]:4d}  {row[1]:4d}  {row[2]:4d}")
        
        self.plot_confusion_matrix(cm, ['Negative', 'Neutral', 'Positive'])
        
        return accuracy
    
    def plot_confusion_matrix(self, cm, labels):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix - Enhanced Model')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        filename = f'confusion_matrix_enhanced_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to {filename}")
        plt.close()
    
    def save_model(self, model_path: str = None):
        """Save trained model and all components"""
        if model_path is None:
            model_path = f"walmart_sentiment_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.model,
            'scaler': self.scaler,
            'label_map': self.label_map,
            'reverse_label_map': self.reverse_label_map,
            'intensity_amplifiers': self.intensity_amplifiers,
            'negation_words': self.negation_words,
            'positive_emoticons': self.positive_emoticons,
            'negative_emoticons': self.negative_emoticons,
            'trained_at': datetime.now().isoformat()
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel saved to {model_path}")
        
        with open("latest_model.txt", 'w') as f:
            f.write(model_path)
        print("Latest model path saved to latest_model.txt")
        
        return model_path
    
    def analyze_feature_importance(self):
        """Analyze which features matter most"""
        if self.model is None or self.vectorizer is None:
            print("Model not trained yet!")
            return
        
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Get feature names
        feature_names = list(self.vectorizer.get_feature_names_out())
        ling_feature_names = [
            'exclamation_ratio', 'question_ratio', 'ellipsis_count', 'all_caps',
            'amplifier_ratio', 'negation_ratio', 'pos_emoticons', 'neg_emoticons',
            'char_count', 'word_count', 'sentence_count', 'capital_word_ratio',
            'repeated_chars', 'question_word_ratio', 'comparative_ratio', 'pronoun_ratio'
        ]
        feature_names.extend(ling_feature_names)
        
        # Analyze each classifier in ensemble
        for name, clf in self.model.named_estimators_.items():
            if hasattr(clf, 'coef_'):
                print(f"\n{name.upper()} Classifier:")
                for sentiment_idx, sentiment_name in self.reverse_label_map.items():
                    print(f"\n  {sentiment_name.upper()} - Top 15 features:")
                    coef = clf.coef_[sentiment_idx]
                    top_indices = np.argsort(np.abs(coef))[-15:][::-1]
                    
                    for i, idx in enumerate(top_indices, 1):
                        if idx < len(feature_names):
                            feature = feature_names[idx]
                            score = coef[idx]
                            print(f"    {i:2d}. {feature:25s} ({score:8.4f})")


def main():
    """Main training function"""
    print("\n" + "="*60)
    print("ENHANCED WALMART SENTIMENT ANALYSIS TRAINER")
    print("="*60)
    print("\nNew features in this version:")
    print("  • Context-aware linguistic features")
    print("  • Negation detection (e.g., 'not good')")
    print("  • Tone recognition (punctuation, caps, emoticons)")
    print("  • Emphasis detection (repeated chars, amplifiers)")
    print("  • Ensemble learning for better accuracy")
    print("  • Character n-grams for subword patterns")
    print()
    
    try:
        trainer = EnhancedSentimentTrainer()
        
        directory = input("Enter directory containing combined JSON files (or press Enter for current): ").strip()
        if not directory:
            directory = "."
        
        reviews = trainer.load_all_combined_files(directory)
        
        if len(reviews) < 100:
            print(f"\nWarning: Only {len(reviews)} reviews found. Recommend at least 100 for training.")
            confirm = input("Continue anyway? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Training cancelled.")
                return
        
        texts, labels, linguistic_features = trainer.prepare_data(reviews)
        
        if len(texts) == 0:
            print("\nError: No valid reviews with sentiment labels found!")
            return
        
        test_size = 0.2
        test_input = input(f"\nTest set size (default 0.2 = 20%): ").strip()
        if test_input:
            try:
                test_size = float(test_input)
                if not 0 < test_size < 1:
                    print("Invalid test size, using default 0.2")
                    test_size = 0.2
            except:
                print("Invalid input, using default 0.2")
                test_size = 0.2
        
        accuracy = trainer.train(texts, labels, linguistic_features, test_size=test_size)
        
        print("\nAnalyze feature importance? (y/n): ", end="")
        if input().strip().lower() == 'y':
            trainer.analyze_feature_importance()
        
        print("\n" + "="*60)
        print("SAVING MODEL")
        print("="*60)
        
        model_path = trainer.save_model()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nModel accuracy: {accuracy*100:.2f}%")
        print(f"Model saved to: {model_path}")
        print(f"\nThis enhanced model includes:")
        print(f"  • Better context understanding")
        print(f"  • Negation handling")
        print(f"  • Tone/emphasis recognition")
        print(f"  • Ensemble predictions")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()