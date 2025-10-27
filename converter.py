import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from datetime import datetime
from typing import Dict, List, Optional
import csv
from tqdm import tqdm


class ReviewConverter:
    def __init__(self):
        """Initialize the converter with RoBERTa sentiment analysis"""
        print("Loading CardiffNLP RoBERTa sentiment model...")
        print("(This may take a moment on first run - downloading model...)")
        
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU (CUDA)" if self.device == 0 else "CPU"
        print(f"Using device: {device_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                truncation=True,
                max_length=512
            )
            
            print(f"✓ RoBERTa sentiment model loaded successfully on {device_name}")
            print("✓ Model: cardiffnlp/twitter-roberta-base-sentiment-latest")
            print("✓ Optimized for social media and review text\n")
            
        except Exception as e:
            print(f"Error loading RoBERTa model: {e}")
            print("Falling back to rating-based sentiment...\n")
            self.sentiment_pipeline = None
    
    def classify_sentiment(self, review_text: str, rating: Optional[float], title: str = "") -> Dict:
        """Classify review sentiment using RoBERTa (same logic as MULTILINK_SCRAPER)"""
        text_to_analyze = f"{title} {review_text}".strip()
        
        if not text_to_analyze:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "score": 0.0,
                "method": "default"
            }
        
        # Fallback to rating-based sentiment if RoBERTa unavailable
        if self.sentiment_pipeline is None:
            if rating is not None:
                if rating >= 4:
                    return {"sentiment": "positive", "confidence": 0.8, "score": 0.8, "method": "rating_fallback"}
                elif rating <= 2:
                    return {"sentiment": "negative", "confidence": 0.8, "score": 0.2, "method": "rating_fallback"}
                else:
                    return {"sentiment": "neutral", "confidence": 0.7, "score": 0.5, "method": "rating_fallback"}
            else:
                return {"sentiment": "neutral", "confidence": 0.5, "score": 0.5, "method": "default"}
        
        try:
            # Truncate if too long
            if len(text_to_analyze) > 2000:
                text_to_analyze = text_to_analyze[:2000]
            
            # Get RoBERTa prediction
            result = self.sentiment_pipeline(text_to_analyze)[0]
            raw_label = result['label'].lower()
            
            # Map labels to sentiment
            if raw_label in ['negative', 'label_0']:
                sentiment = "negative"
            elif raw_label in ['neutral', 'label_1']:
                sentiment = "neutral"
            elif raw_label in ['positive', 'label_2']:
                sentiment = "positive"
            else:
                sentiment = "neutral"
            
            confidence = result['score']
            
            # Calculate normalized score (0-1 scale)
            if sentiment == "negative":
                score = (1 - confidence) * 0.5
            elif sentiment == "neutral":
                score = 0.5
            else:  # positive
                score = 0.5 + (confidence * 0.5)
            
            method = "roberta"
            
            # Align with rating if available (rating validation)
            if rating is not None:
                if (sentiment == "positive" and rating >= 4) or \
                   (sentiment == "negative" and rating <= 2) or \
                   (sentiment == "neutral" and rating == 3):
                    confidence = min(confidence * 1.1, 1.0)
                    method = "roberta_aligned"
                elif confidence < 0.6:
                    # Adjust sentiment based on rating if confidence is low
                    if rating >= 4 and sentiment != "positive":
                        sentiment = "positive"
                        confidence = 0.75
                        score = 0.75
                        method = "roberta_rating_adjusted"
                    elif rating <= 2 and sentiment != "negative":
                        sentiment = "negative"
                        confidence = 0.75
                        score = 0.25
                        method = "roberta_rating_adjusted"
            
            return {
                "sentiment": sentiment,
                "confidence": round(confidence, 4),
                "score": round(score, 4),
                "roberta_label": result['label'],
                "method": method
            }
            
        except Exception as e:
            print(f"RoBERTa error: {e}")
            # Fallback to rating
            if rating is not None:
                if rating >= 4:
                    return {"sentiment": "positive", "confidence": 0.8, "score": 0.8, "method": "rating_fallback"}
                elif rating <= 2:
                    return {"sentiment": "negative", "confidence": 0.8, "score": 0.2, "method": "rating_fallback"}
                else:
                    return {"sentiment": "neutral", "confidence": 0.7, "score": 0.5, "method": "rating_fallback"}
            else:
                return {"sentiment": "neutral", "confidence": 0.5, "score": 0.5, "method": "default"}
    
    def convert_review_with_sentiment(self, amazon_review: Dict, product_id: str, 
                                      sentiment_result: Optional[Dict], rating: Optional[float]) -> Dict:
        """Convert review with pre-computed sentiment"""
        # Extract review text and title
        review_text = amazon_review.get('reviewText', '').strip()
        title = amazon_review.get('summary', '').strip()
        
        # Process sentiment result
        if sentiment_result:
            raw_label = sentiment_result['label'].lower()
            
            # Map labels to sentiment
            if raw_label in ['negative', 'label_0']:
                sentiment = "negative"
            elif raw_label in ['neutral', 'label_1']:
                sentiment = "neutral"
            elif raw_label in ['positive', 'label_2']:
                sentiment = "positive"
            else:
                sentiment = "neutral"
            
            confidence = sentiment_result['score']
            
            # Calculate normalized score
            if sentiment == "negative":
                score = (1 - confidence) * 0.5
            elif sentiment == "neutral":
                score = 0.5
            else:
                score = 0.5 + (confidence * 0.5)
            
            method = "roberta"
            
            # Align with rating if available
            if rating is not None:
                if (sentiment == "positive" and rating >= 4) or \
                   (sentiment == "negative" and rating <= 2) or \
                   (sentiment == "neutral" and rating == 3):
                    confidence = min(confidence * 1.1, 1.0)
                    method = "roberta_aligned"
                elif confidence < 0.6:
                    if rating >= 4 and sentiment != "positive":
                        sentiment = "positive"
                        confidence = 0.75
                        score = 0.75
                        method = "roberta_rating_adjusted"
                    elif rating <= 2 and sentiment != "negative":
                        sentiment = "negative"
                        confidence = 0.75
                        score = 0.25
                        method = "roberta_rating_adjusted"
            
            sentiment_data = {
                "sentiment": sentiment,
                "confidence": round(confidence, 4),
                "score": round(score, 4),
                "roberta_label": sentiment,  # Use sentiment name instead of raw label
                "method": method
            }
        else:
            # Fallback to rating-based
            if rating is not None:
                if rating >= 4:
                    sentiment_data = {"sentiment": "positive", "confidence": 0.8, "score": 0.8, 
                                    "roberta_label": "positive", "method": "rating_fallback"}
                elif rating <= 2:
                    sentiment_data = {"sentiment": "negative", "confidence": 0.8, "score": 0.2,
                                    "roberta_label": "negative", "method": "rating_fallback"}
                else:
                    sentiment_data = {"sentiment": "neutral", "confidence": 0.7, "score": 0.5,
                                    "roberta_label": "neutral", "method": "rating_fallback"}
            else:
                sentiment_data = {"sentiment": "neutral", "confidence": 0.5, "score": 0.5,
                                "roberta_label": "neutral", "method": "default"}
        
        # Convert to exact Walmart format
        converted = {
            'reviewer_name': amazon_review.get('reviewerName', 'Anonymous') or 'Anonymous',
            'rating': rating,
            'title': title,
            'review_text': review_text,
            'date': amazon_review.get('reviewTime', ''),
            'verified_purchase': True,
            'helpful_count': int(amazon_review.get('helpful_yes', 0)),
            'sentiment': sentiment_data['sentiment'],
            'confidence': sentiment_data['confidence'],
            'score': sentiment_data['score'],
            'roberta_label': sentiment_data['roberta_label'],
            'method': sentiment_data['method'],
            'product_id': product_id,
            'product_url': f"https://www.amazon.com/dp/{product_id}",
            'product_name': None
        }
        
        return converted
    
    def convert_review(self, amazon_review: Dict, product_id: str) -> Dict:
        """Convert single Amazon review to exact Walmart scraper format"""
        # Extract rating (convert to float or None)
        try:
            rating = float(amazon_review.get('overall', 0))
            if rating == 0:
                rating = None
        except:
            rating = None
        
        # Extract review text and title
        review_text = amazon_review.get('reviewText', '').strip()
        title = amazon_review.get('summary', '').strip()
        
        # Get sentiment analysis
        sentiment_result = self.classify_sentiment(review_text, rating, title)
        
        # Convert to exact Walmart format - ORDER MATTERS!
        converted = {
            # Original review fields first
            'reviewer_name': amazon_review.get('reviewerName', 'Anonymous') or 'Anonymous',
            'rating': rating,
            'title': title,
            'review_text': review_text,
            'date': amazon_review.get('reviewTime', ''),
            'verified_purchase': True,  # Amazon reviews are typically verified
            'helpful_count': int(amazon_review.get('helpful_yes', 0)),
            
            # Sentiment fields next
            'sentiment': sentiment_result['sentiment'],
            'confidence': sentiment_result['confidence'],
            'score': sentiment_result['score'],
            'roberta_label': sentiment_result.get('roberta_label', sentiment_result['sentiment']),
            'method': sentiment_result['method'],
            
            # Product fields last
            'product_id': product_id,
            'product_url': f"https://www.amazon.com/dp/{product_id}",
            'product_name': None
        }
        
        return converted
    
    def convert_file(self, input_file: str, output_base: str = None):
        """Convert entire JSON file from Amazon format to Walmart format"""
        print(f"Reading {input_file}...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            amazon_reviews = json.load(f)
        
        print(f"Found {len(amazon_reviews)} reviews")
        print("Converting and analyzing sentiment...\n")
        
        # Group by product first to match structure
        products = {}
        for review in amazon_reviews:
            product_id = review.get('asin', 'UNKNOWN')
            if product_id not in products:
                products[product_id] = []
            products[product_id].append(review)
        
        print(f"✓ Found {len(products)} unique products\n")
        
        # Convert all reviews with batch processing
        all_reviews_converted = []
        all_products_data = []
        
        # Calculate total reviews for progress bar
        total_reviews = sum(len(reviews) for reviews in products.values())
        
        with tqdm(total=total_reviews, desc="Processing reviews", unit="review") as pbar:
            for product_id, product_reviews in products.items():
                print(f"\nProduct {product_id} ({len(product_reviews)} reviews)")
                
                # Batch process sentiment analysis
                batch_size = 32
                converted_reviews = []
                
                for i in range(0, len(product_reviews), batch_size):
                    batch = product_reviews[i:i+batch_size]
                    
                    # Prepare texts for batch analysis
                    texts_to_analyze = []
                    ratings = []
                    for review in batch:
                        try:
                            rating = float(review.get('overall', 0))
                            if rating == 0:
                                rating = None
                        except:
                            rating = None
                        ratings.append(rating)
                        
                        title = review.get('summary', '').strip()
                        review_text = review.get('reviewText', '').strip()
                        text = f"{title} {review_text}".strip()
                        texts_to_analyze.append(text[:2000] if len(text) > 2000 else text)
                    
                    # Batch sentiment analysis
                    if self.sentiment_pipeline and texts_to_analyze:
                        try:
                            sentiment_results = self.sentiment_pipeline(texts_to_analyze)
                        except:
                            sentiment_results = [None] * len(batch)
                    else:
                        sentiment_results = [None] * len(batch)
                    
                    # Convert reviews with pre-computed sentiments
                    for j, review in enumerate(batch):
                        converted = self.convert_review_with_sentiment(
                            review, product_id, sentiment_results[j], ratings[j]
                        )
                        converted_reviews.append(converted)
                        all_reviews_converted.append(converted)
                        pbar.update(1)
            
                # Create product data structure
                product_data = {
                    "product_id": product_id,
                    "product_url": f"https://www.amazon.com/dp/{product_id}",
                    "product_name": None,
                    "reviews": converted_reviews,
                    "scraped_at": datetime.now().isoformat()
                }
                all_products_data.append(product_data)
        
        print(f"\n✓ Converted {len(all_reviews_converted)} reviews")
        
        # Calculate statistics
        positive_reviews = [r for r in all_reviews_converted if r.get('sentiment') == 'positive']
        negative_reviews = [r for r in all_reviews_converted if r.get('sentiment') == 'negative']
        neutral_reviews = [r for r in all_reviews_converted if r.get('sentiment') == 'neutral']
        
        all_confidences = [r.get('confidence', 0) for r in all_reviews_converted if r.get('confidence')]
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        
        all_scores = [r.get('score', 0) for r in all_reviews_converted if r.get('score') is not None]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        
        # Create output structure matching MULTILINK_SCRAPER format exactly
        output = {
            "metadata": {
                "source_file": input_file,
                "total_products": len(products),
                "total_reviews": len(all_reviews_converted),
                "positive_count": len(positive_reviews),
                "negative_count": len(negative_reviews),
                "neutral_count": len(neutral_reviews),
                "average_confidence": round(avg_confidence, 4),
                "average_score": round(avg_score, 4),
                "sentiment_analyzer": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "device": "GPU (CUDA)" if self.device == 0 else "CPU",
                "converted_at": datetime.now().isoformat()
            },
            "reviews": {
                "all": all_reviews_converted,
                "positive": positive_reviews,
                "negative": negative_reviews,
                "neutral": neutral_reviews
            },
            "products": all_products_data
        }
        
        # Generate output filename
        if output_base is None:
            output_base = input_file.replace('.json', '_converted')
        
        # Save main combined JSON
        json_file = f"{output_base}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Saved combined file to {json_file}")
        
        # Save separate sentiment JSON files
        positive_file = f"{output_base}_positive.json"
        with open(positive_file, 'w', encoding='utf-8') as f:
            positive_data = {
                "metadata": {
                    "source_file": input_file,
                    "sentiment": "positive",
                    "total_reviews": len(positive_reviews),
                    "average_confidence": round(sum(r['confidence'] for r in positive_reviews) / len(positive_reviews), 4) if positive_reviews else 0,
                    "average_score": round(sum(r['score'] for r in positive_reviews) / len(positive_reviews), 4) if positive_reviews else 0,
                    "sentiment_analyzer": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                    "converted_at": datetime.now().isoformat()
                },
                "reviews": positive_reviews
            }
            json.dump(positive_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved positive reviews to {positive_file}")
        
        negative_file = f"{output_base}_negative.json"
        with open(negative_file, 'w', encoding='utf-8') as f:
            negative_data = {
                "metadata": {
                    "source_file": input_file,
                    "sentiment": "negative",
                    "total_reviews": len(negative_reviews),
                    "average_confidence": round(sum(r['confidence'] for r in negative_reviews) / len(negative_reviews), 4) if negative_reviews else 0,
                    "average_score": round(sum(r['score'] for r in negative_reviews) / len(negative_reviews), 4) if negative_reviews else 0,
                    "sentiment_analyzer": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                    "converted_at": datetime.now().isoformat()
                },
                "reviews": negative_reviews
            }
            json.dump(negative_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved negative reviews to {negative_file}")
        
        neutral_file = f"{output_base}_neutral.json"
        with open(neutral_file, 'w', encoding='utf-8') as f:
            neutral_data = {
                "metadata": {
                    "source_file": input_file,
                    "sentiment": "neutral",
                    "total_reviews": len(neutral_reviews),
                    "average_confidence": round(sum(r['confidence'] for r in neutral_reviews) / len(neutral_reviews), 4) if neutral_reviews else 0,
                    "average_score": round(sum(r['score'] for r in neutral_reviews) / len(neutral_reviews), 4) if neutral_reviews else 0,
                    "sentiment_analyzer": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                    "converted_at": datetime.now().isoformat()
                },
                "reviews": neutral_reviews
            }
            json.dump(neutral_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved neutral reviews to {neutral_file}")
        
        # Save CSV
        csv_file = f"{output_base}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            # Field order matches the converted review structure
            fieldnames = [
                'reviewer_name', 'rating', 'title', 'review_text', 'date', 
                'verified_purchase', 'helpful_count',
                'sentiment', 'confidence', 'score', 'roberta_label', 'method',
                'product_id', 'product_url', 'product_name'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for review in all_reviews_converted:
                writer.writerow(review)
        print(f"✓ Saved to {csv_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("CONVERSION SUMMARY")
        print("="*60)
        print(f"Total Reviews: {len(all_reviews_converted)}")
        print(f"Unique Products: {len(products)}")
        print(f"\nSentiment Distribution:")
        print(f"  Positive: {len(positive_reviews):4d} ({len(positive_reviews)/len(all_reviews_converted)*100:5.1f}%)")
        print(f"  Negative: {len(negative_reviews):4d} ({len(negative_reviews)/len(all_reviews_converted)*100:5.1f}%)")
        print(f"  Neutral:  {len(neutral_reviews):4d} ({len(neutral_reviews)/len(all_reviews_converted)*100:5.1f}%)")
        print(f"\nRoBERTa Analysis:")
        print(f"  Average Confidence: {avg_confidence:.2%}")
        print(f"  Average Score: {avg_score:.4f}")
        
        # Method breakdown
        methods = {}
        for r in all_reviews_converted:
            method = r.get('method', 'unknown')
            methods[method] = methods.get(method, 0) + 1
        
        print(f"\nAnalysis Methods:")
        for method, count in sorted(methods.items(), key=lambda x: x[1], reverse=True):
            print(f"  {method}: {count} ({count/len(all_reviews_converted)*100:.1f}%)")
        
        print("\n" + "="*60)
        print("CONVERSION COMPLETED SUCCESSFULLY!")
        print("="*60)


def main():
    print("\n" + "="*60)
    print("AMAZON TO WALMART REVIEW CONVERTER")
    print("with RoBERTa Sentiment Analysis")
    print("="*60)
    print("\nThis script converts Amazon review JSON to Walmart scraper format")
    print("and adds sentiment analysis using the same RoBERTa model.\n")
    
    # Get input file
    input_file = input("Enter input JSON file path (e.g., data.json): ").strip()
    if not input_file:
        input_file = "data.json"
    
    # Get output filename (optional)
    output_base = input("Enter output filename base (press Enter for auto): ").strip()
    if not output_base:
        output_base = None
    
    try:
        converter = ReviewConverter()
        converter.convert_file(input_file, output_base)
        
    except FileNotFoundError:
        print(f"\n❌ Error: File '{input_file}' not found!")
    except json.JSONDecodeError:
        print(f"\n❌ Error: '{input_file}' is not valid JSON!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()