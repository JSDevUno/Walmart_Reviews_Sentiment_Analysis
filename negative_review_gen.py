"""
Synthetic Negative Review Generator using Markov Chains
Loads existing Walmart review datasets and generates new synthetic negative reviews
Maintains the same JSON structure as the scraper output
"""

import json
import random
import markovify
from datetime import datetime
from typing import List, Dict
import os


class SyntheticReviewGenerator:
    def __init__(self):
        self.markov_model = None
        self.training_reviews = []
        
    def load_dataset(self, json_file: str, append: bool = True):
        """Load existing Walmart review dataset"""
        print(f"Loading dataset from {os.path.basename(json_file)}...")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both single product and combined formats
        if 'reviews' in data:
            # Single product format
            reviews = data['reviews']
            print(f"  Found {len(reviews)} reviews from single product")
        elif 'products' in data:
            # Combined format
            reviews = []
            for product in data['products']:
                reviews.extend(product.get('reviews', []))
            print(f"  Found {len(reviews)} reviews from {len(data['products'])} products")
        else:
            raise ValueError("Unknown JSON format. Expected 'reviews' or 'products' key.")
        
        # Filter for negative reviews only (if sentiment data exists)
        negative_reviews = [r for r in reviews if r.get('sentiment') == 'negative']
        
        if negative_reviews:
            print(f"  Using {len(negative_reviews)} negative reviews for training")
            new_reviews = negative_reviews
        else:
            print(f"  No sentiment data found, using all {len(reviews)} reviews")
            new_reviews = reviews
        
        # Append or replace training data
        if append and self.training_reviews:
            self.training_reviews.extend(new_reviews)
        else:
            self.training_reviews = new_reviews
        
        return len(new_reviews)
    
    def train_markov_model(self, state_size: int = 2):
        """Train Markov chain model on review texts"""
        print(f"Training Markov model (state_size={state_size})...")
        
        if not self.training_reviews:
            raise ValueError("No training data loaded. Call load_dataset() first.")
        
        # Combine all review texts
        review_texts = []
        for review in self.training_reviews:
            text = review.get('review_text', '')
            title = review.get('title', '')
            
            # Combine title and text for better context
            if title and text:
                combined = f"{title}. {text}"
            elif text:
                combined = text
            else:
                continue
            
            review_texts.append(combined)
        
        if not review_texts:
            raise ValueError("No valid review texts found in training data")
        
        # Join all texts with newlines for Markov training
        training_text = "\n".join(review_texts)
        
        # Train the model
        self.markov_model = markovify.Text(training_text, state_size=state_size)
        print(f"✓ Model trained on {len(review_texts)} review texts")
    
    def generate_review_text(self, seen_texts: set, seen_titles: set, max_attempts: int = 200) -> tuple:
        """Generate unique synthetic review text and title"""
        if not self.markov_model:
            raise ValueError("Model not trained. Call train_markov_model() first.")
        
        # Generate review text (longer) - ensure uniqueness
        review_text = None
        for attempt in range(max_attempts):
            # Vary the parameters for more diversity
            max_words = random.randint(15, 80)
            sentence = self.markov_model.make_sentence(tries=100, max_words=max_words)
            
            if sentence and len(sentence.split()) >= 10:
                # Check if unique
                if sentence not in seen_texts:
                    # Also check for substantial similarity (fuzzy check)
                    is_unique = True
                    sentence_words = set(sentence.lower().split())
                    for existing in seen_texts:
                        existing_words = set(existing.lower().split())
                        # If more than 70% words overlap, consider it duplicate
                        if len(sentence_words & existing_words) / len(sentence_words | existing_words) > 0.7:
                            is_unique = False
                            break
                    
                    if is_unique:
                        review_text = sentence
                        break
        
        if not review_text:
            # Fallback: try without uniqueness check
            review_text = self.markov_model.make_sentence(tries=100)
        
        # Generate title (shorter) - ensure uniqueness
        title = None
        for attempt in range(max_attempts):
            max_chars = random.randint(40, 100)
            sentence = self.markov_model.make_short_sentence(max_chars=max_chars, tries=100)
            
            if sentence and 3 <= len(sentence.split()) <= 15:
                if sentence not in seen_titles:
                    title = sentence
                    break
        
        if not title:
            title = self.markov_model.make_short_sentence(max_chars=100, tries=100) or ""
        
        return title, review_text
    
    def sample_metadata(self) -> Dict:
        """Sample realistic metadata from training data"""
        if not self.training_reviews:
            return {}
        
        sample = random.choice(self.training_reviews)
        
        return {
            'rating': sample.get('rating'),
            'verified_purchase': sample.get('verified_purchase', False),
            'helpful_count': random.choice([0, 0, 0, 1, 2]) if random.random() > 0.7 else 0,
            'date': sample.get('date', ''),
        }
    
    def generate_reviewer_name(self) -> str:
        """Generate realistic reviewer names"""
        first_names = ['John', 'Sarah', 'Mike', 'Emily', 'David', 'Jessica', 'Chris', 
                      'Amanda', 'Ryan', 'Jennifer', 'Matt', 'Lisa', 'Tom', 'Karen']
        
        if random.random() > 0.3:
            # Full name with initial
            last_initial = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            return f"{random.choice(first_names)} {last_initial}."
        else:
            # Just first name
            return random.choice(first_names)
    
    def generate_synthetic_review(self, seen_texts: set, seen_titles: set, 
                                  product_id: str = None, product_name: str = None, 
                                  product_url: str = None) -> Dict:
        """Generate a single synthetic review matching Walmart format"""
        title, review_text = self.generate_review_text(seen_texts, seen_titles)
        
        if not review_text:
            return None
        
        metadata = self.sample_metadata()
        
        # Negative reviews typically have higher confidence and lower score
        confidence = round(random.uniform(0.65, 0.95), 4)
        score = round(random.uniform(0.10, 0.35), 4)
        
        synthetic_review = {
            'reviewer_name': self.generate_reviewer_name(),
            'rating': metadata.get('rating', None),
            'title': title or '',
            'review_text': review_text,
            'date': metadata.get('date', ''),
            'verified_purchase': metadata.get('verified_purchase', False),
            'helpful_count': metadata.get('helpful_count', 0),
            'sentiment': 'negative',
            'confidence': confidence,
            'score': score,
            'roberta_label': 'negative',
            'method': 'synthetic_markov'
        }
        
        # Add product info if provided
        if product_id:
            synthetic_review['product_id'] = product_id
        if product_name:
            synthetic_review['product_name'] = product_name
        if product_url:
            synthetic_review['product_url'] = product_url
        
        return synthetic_review
    
    def generate_dataset(self, num_reviews: int, num_products: int = 1, 
                        base_product_id: str = None) -> Dict:
        """Generate a complete synthetic dataset with unique reviews"""
        print(f"Generating {num_reviews} synthetic negative reviews...")
        print("Ensuring all reviews are unique (text, structure, and wording)...\n")
        
        # Calculate reviews per product
        reviews_per_product = num_reviews // num_products
        remainder = num_reviews % num_products
        
        products = []
        seen_texts = set()
        seen_titles = set()
        total_generated = 0
        
        for product_idx in range(num_products):
            product_id = f"{base_product_id or 'SYNTHETIC'}_{product_idx + 1:03d}"
            product_name = f"Synthetic Product {product_idx + 1}"
            product_url = f"https://walmart.com/synthetic/{product_id}"
            
            # Calculate reviews for this product
            target_reviews = reviews_per_product + (1 if product_idx < remainder else 0)
            
            product_reviews = []
            attempts = 0
            max_attempts = target_reviews * 10
            failed_attempts = 0
            max_failed = 100
            
            print(f"Product {product_idx + 1}/{num_products}: Generating {target_reviews} reviews...")
            
            while len(product_reviews) < target_reviews and attempts < max_attempts:
                attempts += 1
                
                review = self.generate_synthetic_review(
                    seen_texts, seen_titles, 
                    product_id, product_name, product_url
                )
                
                if review and review['review_text']:
                    review_text = review['review_text']
                    review_title = review['title']
                    
                    # Strict uniqueness check
                    if review_text not in seen_texts:
                        # Check for substantial similarity with existing reviews
                        is_unique = True
                        review_words = set(review_text.lower().split())
                        
                        for existing_text in seen_texts:
                            existing_words = set(existing_text.lower().split())
                            
                            # Calculate Jaccard similarity
                            intersection = len(review_words & existing_words)
                            union = len(review_words | existing_words)
                            
                            if union > 0:
                                similarity = intersection / union
                                
                                # If more than 60% similar, reject
                                if similarity > 0.6:
                                    is_unique = False
                                    break
                        
                        if is_unique:
                            seen_texts.add(review_text)
                            if review_title:
                                seen_titles.add(review_title)
                            
                            product_reviews.append(review)
                            total_generated += 1
                            failed_attempts = 0
                            
                            if len(product_reviews) % 100 == 0:
                                print(f"  Generated {len(product_reviews)}/{target_reviews}...")
                        else:
                            failed_attempts += 1
                    else:
                        failed_attempts += 1
                else:
                    failed_attempts += 1
                
                # Check if we're stuck
                if failed_attempts >= max_failed:
                    print(f"\n⚠ Warning: Difficulty generating more unique reviews for product {product_idx + 1}")
                    print(f"  Generated {len(product_reviews)}/{target_reviews} reviews for this product")
                    break
            
            # Add product to dataset
            if product_reviews:
                products.append({
                    'product_id': product_id,
                    'product_name': product_name,
                    'product_url': product_url,
                    'reviews': product_reviews
                })
        
        print(f"\n✓ Generated {total_generated} unique synthetic reviews across {len(products)} products")
        print(f"  Uniqueness: 100% (no duplicates or highly similar reviews)")
        
        # Calculate averages
        all_reviews = []
        for product in products:
            all_reviews.extend(product['reviews'])
        
        confidences = [r['confidence'] for r in all_reviews]
        scores = [r['score'] for r in all_reviews]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Create dataset with multi-product format
        dataset = {
            "metadata": {
                "filter": "NEGATIVE_ONLY",
                "total_products": len(products),
                "total_negative_reviews": total_generated,
                "average_confidence": round(avg_confidence, 4),
                "average_score": round(avg_score, 4),
                "generation_method": "markov_chain_synthetic",
                "uniqueness": "100% (strict similarity checking)",
                "sentiment_analyzer": "synthetic_negative",
                "device": "N/A",
                "generated_at": datetime.now().isoformat()
            },
            "products": products
        }
        
        return dataset


def find_json_files(directory: str) -> List[str]:
    """Find all JSON files in a directory"""
    json_files = []
    for file in os.listdir(directory):
        if file.endswith('.json'):
            json_files.append(os.path.join(directory, file))
    return json_files


def main():
    print("\n" + "="*60)
    print("SYNTHETIC NEGATIVE REVIEW GENERATOR")
    print("="*60)
    print("Uses Markov Chains trained on your existing datasets")
    print("Generates reviews with the same JSON structure as scraper\n")
    
    # Get input file or directory
    input_path = input("Enter path to JSON file or directory: ").strip()
    
    if not os.path.exists(input_path):
        print(f"Error: Path '{input_path}' not found!")
        return
    
    # Handle directory vs file
    json_files = []
    if os.path.isdir(input_path):
        print(f"\n'{input_path}' is a directory. Looking for JSON files...")
        json_files = find_json_files(input_path)
        
        if not json_files:
            print(f"Error: No JSON files found in '{input_path}'")
            return
        
        print(f"Found {len(json_files)} JSON file(s):")
        for i, f in enumerate(json_files, 1):
            print(f"  {i}. {os.path.basename(f)}")
        
        if len(json_files) == 1:
            input_file = json_files[0]
            print(f"\nUsing: {os.path.basename(input_file)}")
        else:
            print("\nOptions:")
            print("  1. Use a specific file (enter number)")
            print("  2. Combine all files for training")
            choice = input("Enter choice (1 or 2): ").strip()
            
            if choice == "1":
                try:
                    file_num = int(input(f"Enter file number (1-{len(json_files)}): ").strip())
                    if 1 <= file_num <= len(json_files):
                        input_file = json_files[file_num - 1]
                        print(f"Using: {os.path.basename(input_file)}")
                    else:
                        print("Invalid number!")
                        return
                except ValueError:
                    print("Invalid input!")
                    return
            elif choice == "2":
                input_file = json_files  # Pass list of files
                print(f"Will combine all {len(json_files)} files for training")
            else:
                print("Invalid choice!")
                return
    else:
        input_file = input_path
    
    # Get number of reviews to generate
    try:
        num_reviews = int(input("How many synthetic reviews to generate? (default: 50): ").strip() or "50")
    except ValueError:
        num_reviews = 50
        print(f"Using default: {num_reviews} reviews")
    
    # Get number of products
    try:
        num_products = int(input("How many products to distribute reviews across? (default: 1): ").strip() or "1")
    except ValueError:
        num_products = 1
        print(f"Using default: {num_products} product(s)")
    
    if num_products > num_reviews:
        num_products = num_reviews
        print(f"Adjusted to {num_products} products (cannot exceed review count)")
    
    # Initialize generator
    generator = SyntheticReviewGenerator()
    
    try:
        # Load dataset(s)
        if isinstance(input_file, list):
            # Load multiple files
            total_loaded = 0
            for json_file in input_file:
                count = generator.load_dataset(json_file)
                total_loaded += count
            print(f"\n✓ Total loaded from all files: {total_loaded} reviews")
        else:
            # Load single file
            total_loaded = generator.load_dataset(input_file)
        
        if total_loaded == 0:
            print("Error: No reviews found in dataset!")
            return
        
        # Train model
        state_size = 2  # Can be adjusted: 2=more random, 3=more coherent
        generator.train_markov_model(state_size=state_size)
        
        # Generate synthetic reviews
        synthetic_dataset = generator.generate_dataset(
            num_reviews=num_reviews,
            num_products=num_products,
            base_product_id="SYNTHETIC"
        )
        
        # Save to file
        output_file = f"walmart_reviews_synthetic_negative_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(synthetic_dataset, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*60)
        print("GENERATION COMPLETE!")
        print("="*60)
        print(f"Output file: {output_file}")
        print(f"Total products: {synthetic_dataset['metadata']['total_products']}")
        print(f"Total synthetic reviews: {synthetic_dataset['metadata']['total_negative_reviews']}")
        print(f"Average confidence: {synthetic_dataset['metadata']['average_confidence']:.2%}")
        print(f"Average score: {synthetic_dataset['metadata']['average_score']:.4f}")
        
        # Show sample reviews from first product
        print("\n" + "="*60)
        print("SAMPLE SYNTHETIC REVIEWS (First Product)")
        print("="*60)
        
        if synthetic_dataset['products']:
            first_product = synthetic_dataset['products'][0]
            print(f"\nProduct ID: {first_product['product_id']}")
            print(f"Reviews in this product: {len(first_product['reviews'])}")
            
            for i, review in enumerate(first_product['reviews'][:3], 1):
                print(f"\nReview {i}:")
                print(f"  Reviewer: {review['reviewer_name']}")
                print(f"  Rating: {review.get('rating', 'N/A')}")
                print(f"  Title: {review['title']}")
                print(f"  Review: {review['review_text'][:150]}...")
                print(f"  Confidence: {review['confidence']:.2%} | Score: {review['score']:.3f}")
        
        print("\n✓ All synthetic reviews saved to JSON file!")
        print("✓ Structure matches your scraper output exactly")
        print("✓ Format: metadata + products array with reviews")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()