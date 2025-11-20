import json
import pickle
import os
import glob
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import re

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (Embedding, LSTM, Bidirectional, Dense, 
                                          Dropout, Input, Concatenate, GlobalMaxPooling1D)
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not installed. Install with: pip install tensorflow")


class BiLSTMContextTrainer:
    def __init__(self):
        """Initialize BiLSTM with context and tone awareness"""
        self.tokenizer = None
        self.model = None
        self.max_len = 150  # Optimal for reviews
        self.vocab_size = 10000
        self.embedding_dim = 100
        
        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.reverse_label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
        # Tone and context markers
        self.intensity_amplifiers = {
            'very', 'extremely', 'incredibly', 'absolutely', 'totally',
            'completely', 'utterly', 'highly', 'really', 'so', 'super'
        }
        
        self.negation_words = {
            'not', 'no', 'never', 'nothing', 'nowhere', 'neither', 'nobody',
            'none', 'hardly', 'scarcely', 'barely', "n't", 'cannot', 'cant', 'won\'t'
        }
        
        self.positive_emoticons = [':)', ':-)', ':D', ':-D', ':P', '^_^', '😊', '😃', '👍', '❤️']
        self.negative_emoticons = [':(', ':-(', ':[', ':-[', ':/',':-/', '😢', '😞', '👎', '💔']
    
    def extract_contextual_features(self, text: str) -> np.ndarray:
        """Extract tone and context features for BiLSTM"""
        features = []
        text_lower = text.lower()
        words = text_lower.split()
        
        # Tone indicators
        features.append(text.count('!') / max(len(text), 1))  # Excitement/anger
        features.append(text.count('?') / max(len(text), 1))  # Uncertainty
        features.append(text.count('...') + text.count('…'))  # Hesitation
        features.append(1 if text.isupper() and len(text) > 10 else 0)  # Shouting
        
        # Intensity (amplifies sentiment)
        amplifier_count = sum(1 for w in words if w in self.intensity_amplifiers)
        features.append(amplifier_count / max(len(words), 1))
        
        # Negation (reverses sentiment)
        negation_count = sum(1 for w in words if w in self.negation_words)
        features.append(negation_count / max(len(words), 1))
        
        # Emoticons (strong tone indicators)
        pos_emoticon = sum(1 for e in self.positive_emoticons if e in text)
        neg_emoticon = sum(1 for e in self.negative_emoticons if e in text)
        features.append(pos_emoticon)
        features.append(neg_emoticon)
        
        # Text length (detailed vs brief)
        features.append(np.log1p(len(text)))
        features.append(np.log1p(len(words)))
        
        # Capitalization emphasis
        capital_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        features.append(capital_words / max(len(words), 1))
        
        # Repeated characters (emphasis: "soooo good")
        repeated_chars = len(re.findall(r'(.)\1{2,}', text_lower))
        features.append(repeated_chars)
        
        # Comparative/superlative (strong opinions)
        comparatives = {'best', 'worst', 'better', 'worse', 'great', 'terrible'}
        comparative_count = sum(1 for w in words if w in comparatives)
        features.append(comparative_count / max(len(words), 1))
        
        # Personal engagement
        personal_pronouns = {'i', 'me', 'my', 'mine'}
        pronoun_count = sum(1 for w in words if w in personal_pronouns)
        features.append(pronoun_count / max(len(words), 1))
        
        return np.array(features, dtype=np.float32)
    
    def preprocess_text(self, text: str) -> str:
        """Enhanced preprocessing preserving context markers"""
        # Preserve negations and contractions
        text = text.replace("n't", " not")
        text = text.replace("won't", "will not")
        text = text.replace("can't", "cannot")
        
        # Mark intensity
        for amplifier in self.intensity_amplifiers:
            text = text.replace(f" {amplifier} ", f" INTENSE_{amplifier} ")
        
        # Mark negations (critical for context)
        for negation in self.negation_words:
            text = text.replace(f" {negation} ", f" NEG_{negation} ")
        
        # Preserve repeated punctuation
        text = re.sub(r'!{2,}', ' MULTIEXCLAIM ', text)
        text = re.sub(r'\?{2,}', ' MULTIQUESTION ', text)
        
        return text.lower()
    
    def load_combined_json(self, json_path: str) -> List[Dict]:
        """Load reviews from JSON"""
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
        """Load all combined JSON files"""
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
        
        print(f"Total unique reviews: {len(unique_reviews)}")
        return unique_reviews
    
    def prepare_data(self, reviews: List[Dict]) -> Tuple:
        """Prepare sequences and contextual features"""
        texts = []
        labels = []
        contextual_features_list = []
        
        for review in reviews:
            text = review.get('review_text', '').strip()
            title = review.get('title', '').strip()
            sentiment = review.get('sentiment', '').lower()
            
            if not text or sentiment not in self.label_map:
                continue
            
            full_text = f"{title} {text}".strip() if title else text
            processed_text = self.preprocess_text(full_text)
            
            texts.append(processed_text)
            labels.append(self.label_map[sentiment])
            
            # Extract contextual features
            context_features = self.extract_contextual_features(full_text)
            contextual_features_list.append(context_features)
        
        contextual_features = np.array(contextual_features_list, dtype=np.float32)
        
        print(f"\nPrepared {len(texts)} samples")
        print(f"Contextual features shape: {contextual_features.shape}")
        
        # Distribution
        unique, counts = np.unique(labels, return_counts=True)
        print("\nSentiment distribution:")
        for label_idx, count in zip(unique, counts):
            sentiment_name = self.reverse_label_map[label_idx]
            percentage = (count / len(labels)) * 100
            print(f"  {sentiment_name.capitalize()}: {count:5d} ({percentage:5.1f}%)")
        
        return texts, np.array(labels), contextual_features
    
    def build_model(self, num_contextual_features: int):
        """Build BiLSTM with attention to context"""
        print("\nBuilding BiLSTM context-aware model...")
        
        # Text input branch (learns sequential context)
        text_input = Input(shape=(self.max_len,), name='text_input')
        
        # Embedding layer
        embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_len,
            mask_zero=True
        )(text_input)
        
        # Bidirectional LSTM layers (captures forward & backward context)
        lstm1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3))(embedding)
        lstm2 = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))(lstm1)
        
        # Global max pooling (captures strongest signals)
        lstm_out = GlobalMaxPooling1D()(lstm2)
        
        # Contextual features input (tone, negation, emphasis)
        context_input = Input(shape=(num_contextual_features,), name='context_input')
        
        # Combine text understanding + contextual awareness
        combined = Concatenate()([lstm_out, context_input])
        
        # Dense layers for classification
        dense1 = Dense(128, activation='relu')(combined)
        dropout1 = Dropout(0.4)(dense1)
        dense2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(0.3)(dense2)
        
        # Output layer
        output = Dense(3, activation='softmax', name='output')(dropout2)
        
        # Build model
        model = Model(inputs=[text_input, context_input], outputs=output)
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nModel Architecture:")
        model.summary()
        
        return model
    
    def train(self, texts: List[str], labels: np.ndarray, 
              contextual_features: np.ndarray, test_size: float = 0.2):
        """Train BiLSTM model"""
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not installed. Run: pip install tensorflow")
        
        print("\n" + "="*60)
        print("TRAINING BiLSTM CONTEXT-AWARE MODEL")
        print("="*60)
        print("\n🧠 Why BiLSTM for context & tone:")
        print("  ✓ Bidirectional: reads text forward AND backward")
        print("  ✓ Understands word order and dependencies")
        print("  ✓ Captures negation context ('not good' vs 'good')")
        print("  ✓ Learns tone patterns from sequence")
        print("  ✓ Combines with explicit tone features")
        print()
        
        # Split data
        X_train, X_test, y_train, y_test, ctx_train, ctx_test = train_test_split(
            texts, labels, contextual_features,
            test_size=test_size, random_state=42, stratify=labels
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Tokenize text
        print("\nTokenizing text...")
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(X_train)
        
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_test_seq = self.tokenizer.texts_to_sequences(X_test)
        
        # Pad sequences
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_len, padding='post')
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.max_len, padding='post')
        
        print(f"  Vocabulary size: {len(self.tokenizer.word_index)}")
        print(f"  Sequence shape: {X_train_pad.shape}")
        
        # Build model
        self.model = self.build_model(contextual_features.shape[1])
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
        
        # Train
        print("\nTraining model (this may take 5-15 minutes)...")
        print("Progress:")
        
        history = self.model.fit(
            {'text_input': X_train_pad, 'context_input': ctx_train},
            y_train,
            validation_split=0.15,
            epochs=30,
            batch_size=32,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        print("\n✓ Training completed!")
        
        # Evaluate
        print("\n" + "="*60)
        print("EVALUATION ON TEST SET")
        print("="*60)
        
        y_pred_probs = self.model.predict(
            {'text_input': X_test_pad, 'context_input': ctx_test},
            verbose=0
        )
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
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
        self.plot_training_history(history)
        
        return accuracy, history
    
    def plot_confusion_matrix(self, cm, labels):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix - BiLSTM')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        filename = f'confusion_matrix_bilstm_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to {filename}")
        plt.close()
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        ax1.plot(history.history['accuracy'], label='Train')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss
        ax2.plot(history.history['loss'], label='Train')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {filename}")
        plt.close()
    
    def save_model(self, model_path: str = None):
        """Save model and tokenizer"""
        if model_path is None:
            model_path = f"walmart_sentiment_bilstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save Keras model
        self.model.save(f"{model_path}.keras")
        print(f"\nKeras model saved to {model_path}.keras")
        
        # Save tokenizer and metadata
        metadata = {
            'tokenizer': self.tokenizer,
            'label_map': self.label_map,
            'reverse_label_map': self.reverse_label_map,
            'max_len': self.max_len,
            'vocab_size': self.vocab_size,
            'intensity_amplifiers': self.intensity_amplifiers,
            'negation_words': self.negation_words,
            'positive_emoticons': self.positive_emoticons,
            'negative_emoticons': self.negative_emoticons,
            'model_type': 'bilstm',
            'trained_at': datetime.now().isoformat()
        }
        
        with open(f"{model_path}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Metadata saved to {model_path}_metadata.pkl")
        
        with open("latest_model.txt", 'w') as f:
            f.write(model_path)
        print("Latest model path saved to latest_model.txt")
        
        return model_path


def main():
    """Main training function"""
    print("\n" + "="*60)
    print("BiLSTM CONTEXT-AWARE SENTIMENT TRAINER")
    print("="*60)
    print("\n🧠 Best for understanding context and tone:")
    print("  • Negation handling: 'not good' vs 'good'")
    print("  • Word order matters: 'good but expensive'")
    print("  • Tone detection: '!!!' vs '...'")
    print("  • Sequential dependencies")
    print()
    
    if not TF_AVAILABLE:
        print("❌ TensorFlow not installed!")
        print("\nInstall with: pip install tensorflow")
        return
    
    try:
        trainer = BiLSTMContextTrainer()
        
        directory = input("Enter directory with JSON files (or Enter for current): ").strip()
        if not directory:
            directory = "."
        
        reviews = trainer.load_all_combined_files(directory)
        
        if len(reviews) < 500:
            print(f"\n⚠️  Warning: Only {len(reviews)} reviews. BiLSTM works best with 5000+")
            confirm = input("Continue anyway? (y/n): ").strip().lower()
            if confirm != 'y':
                return
        
        texts, labels, contextual_features = trainer.prepare_data(reviews)
        
        if len(texts) == 0:
            print("\n❌ No valid reviews found!")
            return
        
        test_size = 0.2
        test_input = input(f"\nTest set size (default 0.2): ").strip()
        if test_input:
            try:
                test_size = float(test_input)
                if not 0 < test_size < 1:
                    test_size = 0.2
            except:
                test_size = 0.2
        
        accuracy, history = trainer.train(texts, labels, contextual_features, test_size)
        
        print("\n" + "="*60)
        print("SAVING MODEL")
        print("="*60)
        
        model_path = trainer.save_model()
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED!")
        print("="*60)
        print(f"\n🎯 Final Accuracy: {accuracy*100:.2f}%")
        print(f"📦 Model: {model_path}.keras")
        print(f"\nBiLSTM advantages:")
        print(f"  ✓ Understands 'not good' ≠ 'good'")
        print(f"  ✓ Detects tone from punctuation")
        print(f"  ✓ Captures word order context")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()