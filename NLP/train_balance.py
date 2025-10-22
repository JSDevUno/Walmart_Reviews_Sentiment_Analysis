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
from collections import Counter

# Imbalanced-learn imports for balancing strategies
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN


class WalmartSentimentTrainer:
    """Base trainer class - copy from your original MODEL.py"""
    def __init__(self):
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
            json_files = [f for f in glob.glob(os.path.join(directory, "*.json")) 
                         if 'combined' in f.lower()]
        
        if not json_files:
            raise FileNotFoundError(f"No combined JSON files found in {directory}")
        
        print(f"\nFound {len(json_files)} combined JSON file(s)")
        
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
        
        print(f"\nTotal unique reviews loaded: {len(unique_reviews)}")
        return unique_reviews
    
    def prepare_data(self, reviews: List[Dict]) -> Tuple[List[str], List[int], List[str]]:
        """Prepare text and labels for training"""
        texts, labels, sentiment_labels = [], [], []
        
        for review in reviews:
            text = review.get('review_text', '').strip()
            title = review.get('title', '').strip()
            sentiment = review.get('sentiment', '').lower()
            
            if not text or sentiment not in self.label_map:
                continue
            
            full_text = f"{title} {text}".strip()
            texts.append(full_text)
            labels.append(self.label_map[sentiment])
            sentiment_labels.append(sentiment)
        
        print(f"\nPrepared {len(texts)} samples for training")
        unique, counts = np.unique(labels, return_counts=True)
        print("\nSentiment distribution:")
        for label_idx, count in zip(unique, counts):
            sentiment_name = self.reverse_label_map[label_idx]
            percentage = (count / len(labels)) * 100
            print(f"  {sentiment_name.capitalize()}: {count:5d} ({percentage:5.1f}%)")
        
        return texts, labels, sentiment_labels
    
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
        with open("latest_model.txt", 'w') as f:
            f.write(model_path)
        
        return model_path


class BalancedWalmartSentimentTrainer(WalmartSentimentTrainer):
    """Extended trainer with balancing strategies"""
    
    def train_with_balancing(self, texts, labels, test_size=0.2, 
                            balance_method='smote', balance_ratio='auto'):
        """
        Train with various balancing strategies
        
        Args:
            texts: List of review texts
            labels: List of sentiment labels
            test_size: Test set proportion
            balance_method: 'none', 'oversample', 'undersample', 'smote', 'smote_tomek', 'smote_enn'
            balance_ratio: 'auto' or float (e.g., 0.5 means minority=50% of majority)
        """
        print("\n" + "="*60)
        print(f"TRAINING WITH BALANCING METHOD: {balance_method.upper()}")
        print("="*60)
        
        # Split data first
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        print(f"\nOriginal Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Show original distribution
        print("\nOriginal training distribution:")
        counter = Counter(y_train)
        for label_idx in sorted(counter.keys()):
            sentiment = self.reverse_label_map[label_idx]
            count = counter[label_idx]
            pct = (count / len(y_train)) * 100
            print(f"  {sentiment.capitalize()}: {count:5d} ({pct:5.1f}%)")
        
        # Vectorize BEFORE balancing (for SMOTE)
        print("\nTraining TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            strip_accents='unicode',
            lowercase=True,
            stop_words='english'
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        print(f"  Feature matrix shape: {X_train_tfidf.shape}")
        
        # Apply balancing strategy
        if balance_method != 'none':
            print(f"\nApplying {balance_method} balancing...")
            X_train_balanced, y_train_balanced = self._apply_balancing(
                X_train_tfidf, y_train, balance_method, balance_ratio
            )
            
            print(f"Balanced training set: {len(y_train_balanced)} samples")
            print("\nBalanced training distribution:")
            counter_balanced = Counter(y_train_balanced)
            for label_idx in sorted(counter_balanced.keys()):
                sentiment = self.reverse_label_map[label_idx]
                count = counter_balanced[label_idx]
                pct = (count / len(y_train_balanced)) * 100
                print(f"  {sentiment.capitalize()}: {count:5d} ({pct:5.1f}%)")
        else:
            X_train_balanced = X_train_tfidf
            y_train_balanced = y_train
        
        # Train model
        print("\nTraining LinearSVC classifier...")
        self.model = LinearSVC(
            C=1.0,
            max_iter=2000,
            random_state=42,
            class_weight='balanced'  # Keep this for extra protection
        )
        
        self.model.fit(X_train_balanced, y_train_balanced)
        print("  Training completed!")
        
        # Evaluate
        self._evaluate_model(X_test, y_test)
        
        return accuracy_score(y_test, self.model.predict(self.vectorizer.transform(X_test)))
    
    def _apply_balancing(self, X, y, method, ratio):
        """Apply the selected balancing method"""
        
        # Determine sampling strategy
        if ratio == 'auto':
            sampling_strategy = 'auto'  # Balance all to majority class
        else:
            # Custom ratio: make minority = ratio * majority
            counter = Counter(y)
            majority_count = max(counter.values())
            target_count = int(majority_count * ratio)
            sampling_strategy = {label: max(count, target_count) 
                               for label, count in counter.items()}
        
        if method == 'oversample':
            sampler = RandomOverSampler(
                sampling_strategy=sampling_strategy,
                random_state=42
            )
        
        elif method == 'undersample':
            sampler = RandomUnderSampler(
                sampling_strategy=sampling_strategy,
                random_state=42
            )
        
        elif method == 'smote':
            # SMOTE works well with TF-IDF sparse matrices
            sampler = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=42,
                k_neighbors=min(5, Counter(y)[min(Counter(y), key=Counter(y).get)] - 1)
            )
        
        elif method == 'smote_tomek':
            sampler = SMOTETomek(
                sampling_strategy=sampling_strategy,
                random_state=42
            )
        
        elif method == 'smote_enn':
            sampler = SMOTEENN(
                sampling_strategy=sampling_strategy,
                random_state=42
            )
        
        else:
            raise ValueError(f"Unknown balancing method: {method}")
        
        X_balanced, y_balanced = sampler.fit_resample(X, y)
        return X_balanced, y_balanced
    
    def _evaluate_model(self, X_test, y_test):
        """Evaluate model with detailed metrics"""
        print("\n" + "="*60)
        print("EVALUATION ON TEST SET")
        print("="*60)
        
        X_test_tfidf = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_tfidf)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\nDetailed Classification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=['Negative', 'Neutral', 'Positive'],
            digits=4
        ))
        
        # Per-class accuracy
        cm = confusion_matrix(y_test, y_pred)
        print("\nPer-Class Accuracy:")
        for i, sentiment in enumerate(['Negative', 'Neutral', 'Positive']):
            class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
            print(f"  {sentiment}: {class_acc:.4f} ({class_acc*100:.2f}%)")
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        print("                Predicted")
        print("              Neg   Neu   Pos")
        for i, (label, row) in enumerate(zip(['Negative', 'Neutral', 'Positive'], cm)):
            print(f"Actual {label:8s}  {row[0]:4d}  {row[1]:4d}  {row[2]:4d}")
        
        self.plot_confusion_matrix(cm, ['Negative', 'Neutral', 'Positive'])


def main_balanced():
    """Main function with balancing options"""
    print("\n" + "="*60)
    print("WALMART SENTIMENT - BALANCED TRAINING")
    print("="*60)
    
    try:
        trainer = BalancedWalmartSentimentTrainer()
        
        # Load data FIRST - default to script directory
        print("\n📁 STEP 1: Load Training Data")
        print("-" * 60)
        
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
        
        print(f"Default directory: {script_dir}")
        directory = input("Use different directory? (press Enter to use default, or type path): ").strip()
        if not directory:
            directory = script_dir
            print(f"✓ Using script directory: {directory}")
        
        reviews = trainer.load_all_combined_files(directory)
        texts, labels, _ = trainer.prepare_data(reviews)
        
        # THEN choose balancing method
        print("\n⚖️  STEP 2: Choose Balancing Method")
        print("-" * 60)
        print("Balancing Methods Available:")
        print("  1. none         - No balancing (only class_weight)")
        print("  2. oversample   - Duplicate minority samples")
        print("  3. undersample  - Remove majority samples")
        print("  4. smote        - Synthetic Minority Over-sampling (RECOMMENDED)")
        print("  5. smote_tomek  - SMOTE + Tomek links cleaning")
        print("  6. smote_enn    - SMOTE + Edited Nearest Neighbors")
        
        choice = input("\nEnter your choice (1-6, default=4): ").strip()
        method_map = {
            '1': 'none', '2': 'oversample', '3': 'undersample',
            '4': 'smote', '5': 'smote_tomek', '6': 'smote_enn'
        }
        balance_method = method_map.get(choice, 'smote')
        print(f"✓ Selected: {balance_method}")
        
        # Balance ratio
        if balance_method != 'none':
            print("\n📊 STEP 3: Set Balance Ratio")
            print("-" * 60)
            print("Balance ratio options:")
            print("  auto  - Balance all classes equally (100% each)")
            print("  0.5   - Minority = 50% of majority")
            print("  0.7   - Minority = 70% of majority")
            print("  1.0   - Full balancing (same as auto)")
            balance_ratio = input("\nEnter ratio (press Enter for auto): ").strip()
            if balance_ratio and balance_ratio != 'auto':
                try:
                    balance_ratio = float(balance_ratio)
                    print(f"✓ Ratio: {balance_ratio}")
                except:
                    print("⚠ Invalid input, using auto")
                    balance_ratio = 'auto'
            else:
                balance_ratio = 'auto'
                print("✓ Using auto balance")
        else:
            balance_ratio = 'auto'
        
        # Train
        print("\n🚀 STEP 4: Training Model")
        print("-" * 60)
        test_size = 0.2
        accuracy = trainer.train_with_balancing(
            texts, labels, 
            test_size=test_size,
            balance_method=balance_method,
            balance_ratio=balance_ratio
        )
        
        # Save
        print("\n💾 STEP 5: Saving Model")
        print("-" * 60)
        model_path = trainer.save_model()
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\n📈 Final Accuracy: {accuracy*100:.2f}%")
        print(f"💾 Model saved to: {model_path}")
        print(f"\n🎯 You can now use this model for predictions!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Install required package if needed:
    # pip install imbalanced-learn
    
    main_balanced()