import json
import pickle
import os
import glob
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


class WalmartSentimentTrainer:
    def __init__(self):
        """Initialize the TF-IDF + SVM sentiment trainer"""
        self.vectorizer = None
        self.model = None
        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.reverse_label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
    def load_combined_json(self, json_path: str) -> List[Dict]:
        """Load reviews from a combined JSON file"""
        print(f"Loading {json_path}...")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        reviews = []
        
        # Handle different JSON structures
        if 'products' in data:
            # Combined format with multiple products
            for product in data['products']:
                if 'reviews' in product and product['reviews']:
                    reviews.extend(product['reviews'])
        elif 'reviews' in data:
            # Single product or flat reviews structure
            if isinstance(data['reviews'], dict):
                # Reviews grouped by sentiment
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
        
        # Remove duplicates based on review text
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
    
    def prepare_data(self, reviews: List[Dict]) -> Tuple[List[str], List[int], List[str]]:
        """Prepare text and labels for training"""
        texts = []
        labels = []
        sentiment_labels = []
        
        for review in reviews:
            text = review.get('review_text', '').strip()
            title = review.get('title', '').strip()
            sentiment = review.get('sentiment', '').lower()
            
            if not text or sentiment not in self.label_map:
                continue
            
            # Combine title and text for better context
            full_text = f"{title} {text}".strip()
            
            texts.append(full_text)
            labels.append(self.label_map[sentiment])
            sentiment_labels.append(sentiment)
        
        print(f"\nPrepared {len(texts)} samples for training")
        
        # Print distribution
        unique, counts = np.unique(labels, return_counts=True)
        print("\nSentiment distribution:")
        for label_idx, count in zip(unique, counts):
            sentiment_name = self.reverse_label_map[label_idx]
            percentage = (count / len(labels)) * 100
            print(f"  {sentiment_name.capitalize()}: {count:5d} ({percentage:5.1f}%)")
        
        return texts, labels, sentiment_labels
    
    def train(self, texts: List[str], labels: List[int], test_size: float = 0.2):
        """Train TF-IDF + LinearSVC model"""
        print("\n" + "="*60)
        print("TRAINING TF-IDF + SVM MODEL")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train TF-IDF vectorizer
        print("\nTraining TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),  # unigrams and bigrams
            min_df=2,
            max_df=0.95,
            strip_accents='unicode',
            lowercase=True,
            stop_words='english'
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        print(f"  Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print(f"  Feature matrix shape: {X_train_tfidf.shape}")
        
        # Train LinearSVC
        print("\nTraining LinearSVC classifier...")
        self.model = LinearSVC(
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        )
        
        self.model.fit(X_train_tfidf, y_train)
        print("  Training completed!")
        
        # Evaluate on test set
        print("\n" + "="*60)
        print("EVALUATION ON TEST SET")
        print("="*60)
        
        X_test_tfidf = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_tfidf)
        
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
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm, ['Negative', 'Neutral', 'Positive'])
        
        return accuracy
    
    def plot_confusion_matrix(self, cm, labels):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix - TF-IDF + SVM')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        filename = f'confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to {filename}")
        plt.close()
    
    def save_model(self, model_path: str = None):
        """Save trained model and vectorizer"""
        if model_path is None:
            model_path = f"walmart_sentiment_tfidf_svm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.model,
            'label_map': self.label_map,
            'reverse_label_map': self.reverse_label_map,
            'trained_at': datetime.now().isoformat()
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel saved to {model_path}")
        
        # Also save a simple model name reference
        with open("latest_model.txt", 'w') as f:
            f.write(model_path)
        print("Latest model path saved to latest_model.txt")
        
        return model_path
    
    def analyze_top_features(self, n_features: int = 20):
        """Analyze top features for each sentiment class"""
        if self.model is None or self.vectorizer is None:
            print("Model not trained yet!")
            return
        
        print("\n" + "="*60)
        print("TOP FEATURES PER SENTIMENT CLASS")
        print("="*60)
        
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        for sentiment_idx, sentiment_name in self.reverse_label_map.items():
            print(f"\n{sentiment_name.upper()} - Top {n_features} features:")
            
            # Get coefficients for this class
            coef = self.model.coef_[sentiment_idx]
            
            # Get top positive features
            top_indices = np.argsort(coef)[-n_features:][::-1]
            top_features = feature_names[top_indices]
            top_scores = coef[top_indices]
            
            for i, (feature, score) in enumerate(zip(top_features, top_scores), 1):
                print(f"  {i:2d}. {feature:20s} ({score:7.4f})")


def main():
    """Main training function"""
    print("\n" + "="*60)
    print("WALMART SENTIMENT ANALYSIS - TF-IDF + SVM TRAINING")
    print("="*60)
    print("\nThis script trains a sentiment classifier using:")
    print("  • TF-IDF (Term Frequency-Inverse Document Frequency)")
    print("  • LinearSVC (Support Vector Machine)")
    print("  • Training data from combined JSON files")
    print()
    
    try:
        trainer = WalmartSentimentTrainer()
        
        # Load data
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
        
        # Prepare data
        texts, labels, sentiment_labels = trainer.prepare_data(reviews)
        
        if len(texts) == 0:
            print("\nError: No valid reviews with sentiment labels found!")
            return
        
        # Check class distribution
        unique, counts = np.unique(labels, return_counts=True)
        min_samples = min(counts)
        if min_samples < 20:
            print(f"\nWarning: Some classes have very few samples (minimum: {min_samples})")
            print("This may affect model performance.")
        
        # Train model
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
        
        accuracy = trainer.train(texts, labels, test_size=test_size)
        
        # Analyze top features
        print("\nAnalyze top features? (y/n): ", end="")
        if input().strip().lower() == 'y':
            n_features = 20
            try:
                n_input = input(f"How many top features per class? (default {n_features}): ").strip()
                if n_input:
                    n_features = int(n_input)
            except:
                pass
            
            trainer.analyze_top_features(n_features)
        
        # Save model
        print("\n" + "="*60)
        print("SAVING MODEL")
        print("="*60)
        
        model_path = trainer.save_model()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nModel accuracy: {accuracy*100:.2f}%")
        print(f"Model saved to: {model_path}")
        print(f"\nYou can now use this model with the inference script.")
        print("Run the inference script and it will automatically load the latest model.")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()