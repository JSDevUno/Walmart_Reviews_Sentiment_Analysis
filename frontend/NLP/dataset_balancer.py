"""
Data Balancer - Balance sentiment distribution in JSON review datasets
Creates balanced datasets by oversampling or undersampling reviews
"""

import json
import os
import glob
from datetime import datetime
from typing import List, Dict
from collections import Counter
import random


class DataBalancer:
    def __init__(self):
        """Initialize the data balancer"""
        self.sentiments = ['positive', 'negative', 'neutral']
    
    def load_combined_json(self, json_path: str) -> List[Dict]:
        """Load reviews from a combined JSON file"""
        print(f"Loading {json_path}...")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        reviews = []
        
        # Handle different JSON structures
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
    
    def analyze_distribution(self, reviews: List[Dict]) -> Dict[str, int]:
        """Analyze sentiment distribution"""
        distribution = Counter()
        
        for review in reviews:
            sentiment = review.get('sentiment', '').lower()
            if sentiment in self.sentiments:
                distribution[sentiment] += 1
        
        return dict(distribution)
    
    def print_distribution(self, distribution: Dict[str, int], title: str = "Distribution"):
        """Pretty print distribution"""
        total = sum(distribution.values())
        
        print(f"\n{title}:")
        print("-" * 60)
        for sentiment in ['positive', 'negative', 'neutral']:
            count = distribution.get(sentiment, 0)
            percentage = (count / total * 100) if total > 0 else 0
            bar_length = int(percentage / 2)  # Scale to 50 chars max
            bar = "█" * bar_length
            print(f"  {sentiment.capitalize():10s}: {count:6d} ({percentage:5.1f}%) {bar}")
        print(f"  {'Total':10s}: {total:6d}")
    
    def balance_oversample(self, reviews: List[Dict], target_count: int = None) -> List[Dict]:
        """Balance by oversampling minority classes"""
        # Group by sentiment
        by_sentiment = {s: [] for s in self.sentiments}
        for review in reviews:
            sentiment = review.get('sentiment', '').lower()
            if sentiment in self.sentiments:
                by_sentiment[sentiment].append(review)
        
        # Determine target count
        if target_count is None:
            target_count = max(len(reviews) for reviews in by_sentiment.values())
        
        print(f"\nTarget count per class: {target_count}")
        
        # Oversample each class
        balanced_reviews = []
        for sentiment in self.sentiments:
            sentiment_reviews = by_sentiment[sentiment]
            current_count = len(sentiment_reviews)
            
            if current_count == 0:
                print(f"⚠️  Warning: No {sentiment} reviews found!")
                continue
            
            # Add all original reviews
            balanced_reviews.extend(sentiment_reviews)
            
            # Add duplicates to reach target
            if current_count < target_count:
                needed = target_count - current_count
                duplicates = random.choices(sentiment_reviews, k=needed)
                balanced_reviews.extend(duplicates)
                print(f"  {sentiment.capitalize()}: Added {needed} duplicates ({current_count} → {target_count})")
            else:
                print(f"  {sentiment.capitalize()}: Already at target ({current_count})")
        
        # Shuffle to mix sentiments
        random.shuffle(balanced_reviews)
        return balanced_reviews
    
    def balance_undersample(self, reviews: List[Dict], target_count: int = None) -> List[Dict]:
        """Balance by undersampling majority classes"""
        # Group by sentiment
        by_sentiment = {s: [] for s in self.sentiments}
        for review in reviews:
            sentiment = review.get('sentiment', '').lower()
            if sentiment in self.sentiments:
                by_sentiment[sentiment].append(review)
        
        # Determine target count
        if target_count is None:
            target_count = min(len(reviews) for reviews in by_sentiment.values() if reviews)
        
        print(f"\nTarget count per class: {target_count}")
        
        # Undersample each class
        balanced_reviews = []
        for sentiment in self.sentiments:
            sentiment_reviews = by_sentiment[sentiment]
            current_count = len(sentiment_reviews)
            
            if current_count == 0:
                print(f"⚠️  Warning: No {sentiment} reviews found!")
                continue
            
            # Randomly sample to target count
            if current_count > target_count:
                sampled = random.sample(sentiment_reviews, target_count)
                balanced_reviews.extend(sampled)
                print(f"  {sentiment.capitalize()}: Reduced from {current_count} → {target_count}")
            else:
                balanced_reviews.extend(sentiment_reviews)
                print(f"  {sentiment.capitalize()}: Kept all {current_count} reviews")
        
        # Shuffle to mix sentiments
        random.shuffle(balanced_reviews)
        return balanced_reviews
    
    def balance_hybrid(self, reviews: List[Dict], target_count: int = None) -> List[Dict]:
        """Balance using hybrid approach (oversample minority, undersample majority)"""
        # Group by sentiment
        by_sentiment = {s: [] for s in self.sentiments}
        for review in reviews:
            sentiment = review.get('sentiment', '').lower()
            if sentiment in self.sentiments:
                by_sentiment[sentiment].append(review)
        
        # Get counts
        counts = {s: len(r) for s, r in by_sentiment.items() if r}
        if not counts:
            return []
        
        # Determine target count (median or custom)
        if target_count is None:
            sorted_counts = sorted(counts.values())
            if len(sorted_counts) >= 2:
                target_count = sorted_counts[len(sorted_counts) // 2]  # Median
            else:
                target_count = sorted_counts[0]
        
        print(f"\nTarget count per class: {target_count}")
        
        # Balance each class
        balanced_reviews = []
        for sentiment in self.sentiments:
            sentiment_reviews = by_sentiment[sentiment]
            current_count = len(sentiment_reviews)
            
            if current_count == 0:
                print(f"⚠️  Warning: No {sentiment} reviews found!")
                continue
            
            if current_count < target_count:
                # Oversample
                balanced_reviews.extend(sentiment_reviews)
                needed = target_count - current_count
                duplicates = random.choices(sentiment_reviews, k=needed)
                balanced_reviews.extend(duplicates)
                print(f"  {sentiment.capitalize()}: Oversampled {current_count} → {target_count}")
            elif current_count > target_count:
                # Undersample
                sampled = random.sample(sentiment_reviews, target_count)
                balanced_reviews.extend(sampled)
                print(f"  {sentiment.capitalize()}: Undersampled {current_count} → {target_count}")
            else:
                balanced_reviews.extend(sentiment_reviews)
                print(f"  {sentiment.capitalize()}: Kept all {current_count} reviews")
        
        # Shuffle
        random.shuffle(balanced_reviews)
        return balanced_reviews
    
    def save_balanced_data(self, reviews: List[Dict], output_path: str):
        """Save balanced reviews to JSON file"""
        data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_reviews': len(reviews),
                'balancing_applied': True
            },
            'reviews': reviews
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Balanced data saved to: {output_path}")


def main():
    """Main balancing function"""
    print("\n" + "="*70)
    print("DATA BALANCER - Balance Sentiment Distribution")
    print("="*70)
    
    try:
        balancer = DataBalancer()
        
        # Load data
        print("\n📁 STEP 1: Load Data")
        print("-" * 70)
        
        script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
        print(f"Default directory: {script_dir}")
        directory = input("Use different directory? (press Enter to use default): ").strip()
        if not directory:
            directory = script_dir
        
        reviews = balancer.load_all_combined_files(directory)
        
        if len(reviews) == 0:
            print("\n❌ No reviews found!")
            return
        
        # Show original distribution
        print("\n📊 STEP 2: Current Distribution")
        print("-" * 70)
        original_dist = balancer.analyze_distribution(reviews)
        balancer.print_distribution(original_dist, "ORIGINAL DISTRIBUTION")
        
        # Choose balancing method
        print("\n⚖️  STEP 3: Choose Balancing Method")
        print("-" * 70)
        print("Balancing methods:")
        print("  1. Oversample  - Duplicate minority classes to match majority")
        print("  2. Undersample - Reduce majority classes to match minority")
        print("  3. Hybrid      - Mix of both (balanced approach)")
        print("  4. Custom      - Specify exact target count")
        
        choice = input("\nEnter choice (1-4, default=3): ").strip() or "3"
        
        # Set seed for reproducibility
        random.seed(42)
        
        # Apply balancing
        print("\n🔄 STEP 4: Applying Balancing")
        print("-" * 70)
        
        if choice == "1":
            print("Method: OVERSAMPLE")
            balanced_reviews = balancer.balance_oversample(reviews)
        elif choice == "2":
            print("Method: UNDERSAMPLE")
            balanced_reviews = balancer.balance_undersample(reviews)
        elif choice == "4":
            print("Method: CUSTOM")
            target = input("Enter target count per class: ").strip()
            try:
                target = int(target)
                balanced_reviews = balancer.balance_hybrid(reviews, target_count=target)
            except ValueError:
                print("Invalid number, using hybrid method")
                balanced_reviews = balancer.balance_hybrid(reviews)
        else:  # Default to hybrid
            print("Method: HYBRID")
            balanced_reviews = balancer.balance_hybrid(reviews)
        
        # Show new distribution
        print("\n📊 STEP 5: New Distribution")
        print("-" * 70)
        new_dist = balancer.analyze_distribution(balanced_reviews)
        balancer.print_distribution(new_dist, "BALANCED DISTRIBUTION")
        
        # Compare
        print("\n📈 Change Summary:")
        print("-" * 70)
        for sentiment in ['positive', 'negative', 'neutral']:
            old = original_dist.get(sentiment, 0)
            new = new_dist.get(sentiment, 0)
            change = new - old
            sign = "+" if change > 0 else ""
            print(f"  {sentiment.capitalize():10s}: {old:5d} → {new:5d} ({sign}{change})")
        
        # Save
        print("\n💾 STEP 6: Save Balanced Data")
        print("-" * 70)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"balanced_reviews_{timestamp}.json"
        
        output_path = input(f"Output filename (default: {default_name}): ").strip()
        if not output_path:
            output_path = default_name
        
        if not output_path.endswith('.json'):
            output_path += '.json'
        
        balancer.save_balanced_data(balanced_reviews, output_path)
        
        print("\n" + "="*70)
        print("✅ BALANCING COMPLETED!")
        print("="*70)
        print(f"\n📊 Original: {len(reviews)} reviews")
        print(f"📊 Balanced: {len(balanced_reviews)} reviews")
        print(f"💾 Saved to: {output_path}")
        print(f"\n🎯 You can now use this balanced dataset for training!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Balancing interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()