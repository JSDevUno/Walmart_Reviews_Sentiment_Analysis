import json
import os
import glob
from datetime import datetime
from typing import List, Dict
from collections import defaultdict


class SentimentCounter:
    def __init__(self):
        """Initialize sentiment counter"""
        self.total_reviews = 0
        self.positive_count = 0
        self.negative_count = 0
        self.neutral_count = 0
        self.product_stats = []
        
    def load_combined_json(self, json_path: str) -> Dict:
        """Load and count sentiments from a combined JSON file"""
        print(f"\nProcessing: {os.path.basename(json_path)}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        reviews = []
        product_info = {
            'filename': os.path.basename(json_path),
            'products': []
        }
        
        # Handle different JSON structures
        if 'products' in data:
            # Combined format with multiple products
            for product in data['products']:
                product_reviews = []
                
                if 'reviews' in product and product['reviews']:
                    product_reviews = product['reviews']
                
                product_id = product.get('product_id', 'Unknown')
                product_name = product.get('product_name', f'Product {product_id}')
                
                # Count sentiments for this product
                pos = sum(1 for r in product_reviews if r.get('sentiment', '').lower() == 'positive')
                neg = sum(1 for r in product_reviews if r.get('sentiment', '').lower() == 'negative')
                neu = sum(1 for r in product_reviews if r.get('sentiment', '').lower() == 'neutral')
                
                product_info['products'].append({
                    'product_id': product_id,
                    'product_name': product_name,
                    'total': len(product_reviews),
                    'positive': pos,
                    'negative': neg,
                    'neutral': neu
                })
                
                reviews.extend(product_reviews)
                
        elif 'reviews' in data:
            # Single product or flat reviews structure
            if isinstance(data['reviews'], dict):
                # Reviews grouped by sentiment
                for sentiment_type in ['positive', 'negative', 'neutral', 'all']:
                    if sentiment_type in data['reviews']:
                        reviews.extend(data['reviews'][sentiment_type])
            elif isinstance(data['reviews'], list):
                reviews.extend(data['reviews'])
            
            # Get product info if available
            product_id = data.get('product_id', 'Unknown')
            product_name = data.get('product_name', f'Product {product_id}')
            
            pos = sum(1 for r in reviews if r.get('sentiment', '').lower() == 'positive')
            neg = sum(1 for r in reviews if r.get('sentiment', '').lower() == 'negative')
            neu = sum(1 for r in reviews if r.get('sentiment', '').lower() == 'neutral')
            
            product_info['products'].append({
                'product_id': product_id,
                'product_name': product_name,
                'total': len(reviews),
                'positive': pos,
                'negative': neg,
                'neutral': neu
            })
        
        # Remove duplicates based on review text
        seen_texts = set()
        unique_reviews = []
        for review in reviews:
            text = review.get('review_text', '')
            if text and text not in seen_texts:
                seen_texts.add(text)
                unique_reviews.append(review)
        
        if len(reviews) != len(unique_reviews):
            print(f"  Removed {len(reviews) - len(unique_reviews)} duplicates")
        
        # Count sentiments
        file_positive = sum(1 for r in unique_reviews if r.get('sentiment', '').lower() == 'positive')
        file_negative = sum(1 for r in unique_reviews if r.get('sentiment', '').lower() == 'negative')
        file_neutral = sum(1 for r in unique_reviews if r.get('sentiment', '').lower() == 'neutral')
        
        print(f"  Reviews: {len(unique_reviews)} total")
        print(f"  Positive: {file_positive} | Negative: {file_negative} | Neutral: {file_neutral}")
        
        # Update totals
        self.total_reviews += len(unique_reviews)
        self.positive_count += file_positive
        self.negative_count += file_negative
        self.neutral_count += file_neutral
        
        product_info['total_reviews'] = len(unique_reviews)
        product_info['positive'] = file_positive
        product_info['negative'] = file_negative
        product_info['neutral'] = file_neutral
        
        self.product_stats.append(product_info)
        
        return product_info
    
    def process_directory(self, directory: str = ".") -> None:
        """Process all combined JSON files in directory"""
        # First, show ALL JSON files in the directory
        all_json_files = glob.glob(os.path.join(directory, "*.json"))
        
        if all_json_files:
            print(f"\nAll JSON files found in {directory}:")
            for f in sorted(all_json_files):
                print(f"  - {os.path.basename(f)}")
        
        # Try multiple patterns
        patterns = [
            "*_combined_*.json",
            "*combined*.json",
            "_hybrid_",
            "*.json"
        ]
        
        json_files = []
        for pattern in patterns:
            full_pattern = os.path.join(directory, pattern)
            found = glob.glob(full_pattern)
            if found:
                json_files = found
                print(f"\nUsing pattern: {pattern}")
                break
        
        if not json_files:
            print(f"\n❌ No JSON files found in {directory}")
            return
        
        # Filter out any non-review files if needed
        # Ask user to confirm if many files found
        if len(json_files) > 5:
            print(f"\nFound {len(json_files)} JSON files.")
            print("Process all of them? (y/n): ", end="")
            confirm = input().strip().lower()
            if confirm != 'y':
                print("\nEnter the specific filenames to process (comma-separated):")
                filenames = input().strip().split(',')
                json_files = [os.path.join(directory, f.strip()) for f in filenames]
        
        print(f"\n{'='*60}")
        print(f"PROCESSING {len(json_files)} JSON FILE(S)")
        print(f"{'='*60}")
        
        for json_file in sorted(json_files):
            try:
                self.load_combined_json(json_file)
            except Exception as e:
                print(f"  ⚠️ Error processing {os.path.basename(json_file)}: {e}")
                continue
        
    def print_summary(self):
        """Print comprehensive summary"""
        if self.total_reviews == 0:
            print("\n❌ No reviews found to count!")
            return
        
        print("\n" + "="*60)
        print("OVERALL SENTIMENT SUMMARY")
        print("="*60)
        
        print(f"\nTotal Reviews: {self.total_reviews}")
        print(f"\nSentiment Distribution:")
        print(f"  Positive: {self.positive_count:5d} ({self.positive_count/self.total_reviews*100:5.1f}%)")
        print(f"  Negative: {self.negative_count:5d} ({self.negative_count/self.total_reviews*100:5.1f}%)")
        print(f"  Neutral:  {self.neutral_count:5d} ({self.neutral_count/self.total_reviews*100:5.1f}%)")
        
        # Visual bar chart
        print(f"\nVisual Distribution:")
        max_count = max(self.positive_count, self.negative_count, self.neutral_count)
        
        pos_bar = "█" * int((self.positive_count / max_count) * 40) if max_count > 0 else ""
        neg_bar = "█" * int((self.negative_count / max_count) * 40) if max_count > 0 else ""
        neu_bar = "█" * int((self.neutral_count / max_count) * 40) if max_count > 0 else ""
        
        print(f"  Positive: {pos_bar} {self.positive_count}")
        print(f"  Negative: {neg_bar} {self.negative_count}")
        print(f"  Neutral:  {neu_bar} {self.neutral_count}")
        
        # Per-file breakdown
        if len(self.product_stats) > 1:
            print("\n" + "="*60)
            print("PER-FILE BREAKDOWN")
            print("="*60)
            
            for i, file_stat in enumerate(self.product_stats, 1):
                print(f"\n{i}. {file_stat['filename']}")
                print(f"   Total: {file_stat['total_reviews']} reviews")
                print(f"   Positive: {file_stat['positive']:4d} ({file_stat['positive']/file_stat['total_reviews']*100:5.1f}%)")
                print(f"   Negative: {file_stat['negative']:4d} ({file_stat['negative']/file_stat['total_reviews']*100:5.1f}%)")
                print(f"   Neutral:  {file_stat['neutral']:4d} ({file_stat['neutral']/file_stat['total_reviews']*100:5.1f}%)")
                
                # Show per-product stats if available
                if len(file_stat['products']) > 1:
                    print(f"   Products in this file:")
                    for product in file_stat['products']:
                        print(f"     - Product {product['product_id']}: {product['total']} reviews "
                              f"(+{product['positive']} -{product['negative']} ={product['neutral']})")
        
        # Sentiment ratio analysis
        print("\n" + "="*60)
        print("SENTIMENT ANALYSIS")
        print("="*60)
        
        if self.positive_count > 0 and self.negative_count > 0:
            pos_neg_ratio = self.positive_count / self.negative_count
            print(f"\nPositive-to-Negative Ratio: {pos_neg_ratio:.2f}:1")
            
            if pos_neg_ratio > 2:
                print("  → Strongly positive overall sentiment")
            elif pos_neg_ratio > 1.5:
                print("  → Moderately positive overall sentiment")
            elif pos_neg_ratio > 0.67:
                print("  → Mixed sentiment (balanced)")
            elif pos_neg_ratio > 0.5:
                print("  → Slightly negative overall sentiment")
            else:
                print("  → Strongly negative overall sentiment")
        
        # Overall sentiment score (0-1 scale)
        sentiment_score = (self.positive_count + (self.neutral_count * 0.5)) / self.total_reviews
        print(f"\nOverall Sentiment Score: {sentiment_score:.3f}")
        print(f"  (0.0 = All Negative, 0.5 = Balanced, 1.0 = All Positive)")
        
        if sentiment_score >= 0.7:
            print("  → Very Positive Product Reception ✓")
        elif sentiment_score >= 0.6:
            print("  → Positive Product Reception")
        elif sentiment_score >= 0.4:
            print("  → Mixed Product Reception")
        elif sentiment_score >= 0.3:
            print("  → Negative Product Reception")
        else:
            print("  → Very Negative Product Reception ✗")
    
    def save_summary(self, output_file: str = None):
        """Save summary to JSON file"""
        if output_file is None:
            output_file = f"sentiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        summary = {
            "generated_at": datetime.now().isoformat(),
            "total_reviews": self.total_reviews,
            "overall_sentiment": {
                "positive": self.positive_count,
                "negative": self.negative_count,
                "neutral": self.neutral_count,
                "positive_percentage": round(self.positive_count / self.total_reviews * 100, 2) if self.total_reviews > 0 else 0,
                "negative_percentage": round(self.negative_count / self.total_reviews * 100, 2) if self.total_reviews > 0 else 0,
                "neutral_percentage": round(self.neutral_count / self.total_reviews * 100, 2) if self.total_reviews > 0 else 0,
                "sentiment_score": round((self.positive_count + (self.neutral_count * 0.5)) / self.total_reviews, 4) if self.total_reviews > 0 else 0
            },
            "files_processed": len(self.product_stats),
            "file_breakdown": self.product_stats
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Summary saved to {output_file}")


def main():
    """Main function"""
    print("\n" + "="*60)
    print("WALMART SENTIMENT COUNTER")
    print("="*60)
    print("\nCount positive, negative, and neutral reviews")
    print("from all combined JSON files")
    print()
    
    try:
        counter = SentimentCounter()
        
        # Get directory
        directory = input("Enter directory containing combined JSON files (or press Enter for current): ").strip()
        if not directory:
            directory = "."
        
        if not os.path.exists(directory):
            print(f"\n❌ Directory not found: {directory}")
            return
        
        # Process all files
        counter.process_directory(directory)
        
        # Print summary
        counter.print_summary()
        
        # Save option
        if counter.total_reviews > 0:
            print("\n" + "="*60)
            save = input("\nSave summary to JSON file? (y/n): ").strip().lower()
            if save == 'y':
                counter.save_summary()
        
        print("\n" + "="*60)
        print("COUNTING COMPLETED!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nCounting interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()