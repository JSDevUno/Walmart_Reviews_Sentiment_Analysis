import pickle
import os
import time
import random
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import json
import re
import tempfile
import shutil


class TFIDFSentimentAnalyzer:
    def __init__(self, model_path: str = None):
        """Initialize the TF-IDF + SVM sentiment analyzer"""
        if model_path is None:
            # Try to load latest model
            if os.path.exists("latest_model.txt"):
                with open("latest_model.txt", 'r') as f:
                    model_path = f.read().strip()
                print(f"Loading latest model: {model_path}")
            else:
                # Look for any .pkl file
                pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl') and 'sentiment' in f.lower()]
                if pkl_files:
                    model_path = sorted(pkl_files)[-1]  # Get most recent
                    print(f"Found model: {model_path}")
                else:
                    raise FileNotFoundError(
                        "No trained model found! Please run the training script first."
                    )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model from {model_path}...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.model = model_data['model']
        self.label_map = model_data['label_map']
        self.reverse_label_map = model_data['reverse_label_map']
        
        trained_at = model_data.get('trained_at', 'Unknown')
        print(f"✓ Model loaded successfully (trained: {trained_at})")
        print(f"✓ Using TF-IDF + LinearSVC classifier")
        
    def predict_sentiment(self, text: str, title: str = "") -> Dict:
        """Predict sentiment for a single review"""
        full_text = f"{title} {text}".strip()
        
        if not full_text:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "probabilities": {"negative": 0.33, "neutral": 0.34, "positive": 0.33}
            }
        
        # Transform text
        text_tfidf = self.vectorizer.transform([full_text])
        
        # Get prediction
        prediction = self.model.predict(text_tfidf)[0]
        sentiment = self.reverse_label_map[prediction]
        
        # Get decision function scores (not probabilities, but we can normalize them)
        decision_scores = self.model.decision_function(text_tfidf)[0]
        
        # Convert decision scores to pseudo-probabilities using softmax
        exp_scores = np.exp(decision_scores - np.max(decision_scores))
        probabilities = exp_scores / exp_scores.sum()
        
        confidence = probabilities[prediction]
        
        prob_dict = {
            self.reverse_label_map[i]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        return {
            "sentiment": sentiment,
            "confidence": float(confidence),
            "probabilities": prob_dict,
            "decision_scores": {
                self.reverse_label_map[i]: float(score) 
                for i, score in enumerate(decision_scores)
            }
        }


class WalmartReviewInference:
    def __init__(self, sentiment_analyzer: TFIDFSentimentAnalyzer, headless: bool = False):
        """Initialize scraper with sentiment analyzer"""
        print("\nStarting Chrome browser...")
        
        options = uc.ChromeOptions()
        if headless:
            options.add_argument('--headless=new')
        
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-images')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.page_load_strategy = 'eager'
        
        self.temp_profile_dir = tempfile.mkdtemp(prefix='walmart_inference_')
        options.add_argument(f'--user-data-dir={self.temp_profile_dir}')
        
        self.driver = uc.Chrome(options=options, version_main=None)
        self.wait = WebDriverWait(self.driver, 8)
        self.analyzer = sentiment_analyzer
        
        print("✓ Browser started successfully")
    
    def close(self):
        """Close browser and cleanup"""
        if hasattr(self, 'driver'):
            try:
                self.driver.quit()
            except:
                pass
        
        if hasattr(self, 'temp_profile_dir'):
            try:
                if os.path.exists(self.temp_profile_dir):
                    shutil.rmtree(self.temp_profile_dir, ignore_errors=True)
            except:
                pass
    
    def human_delay(self, min_sec: float = 1.0, max_sec: float = 3.0):
        """Random delay"""
        time.sleep(random.uniform(min_sec, max_sec))
    
    def extract_product_id(self, url: str) -> Optional[str]:
        """Extract product ID from URL"""
        patterns = [
            r'/ip/[^/]+/(\d+)',
            r'/(\d+)\?',
            r'/(\d+)$',
            r'itemId=(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def click_element_safely(self, element):
        """Click element safely"""
        try:
            self.driver.execute_script(
                "arguments[0].scrollIntoView({block: 'center', behavior: 'instant'});", 
                element
            )
            self.human_delay(0.2, 0.5)
            self.driver.execute_script("arguments[0].click();", element)
            return True
        except:
            return False
    
    def extract_review_from_element(self, element) -> Optional[Dict]:
        """Extract review data from element"""
        try:
            review_data = {}
            
            # Reviewer name
            name_selectors = [
                ".//*[contains(@class, 'reviewer')]",
                ".//*[contains(@class, 'author')]",
                ".//*[contains(@class, 'name')]"
            ]
            reviewer_name = 'Anonymous'
            for sel in name_selectors:
                try:
                    reviewer = element.find_element(By.XPATH, sel)
                    name = reviewer.text.strip()
                    if name and len(name) < 50:
                        reviewer_name = name
                        break
                except:
                    continue
            review_data['reviewer_name'] = reviewer_name
            
            # Rating
            rating_selectors = [
                ".//*[contains(@aria-label, 'star')]",
                ".//*[contains(@class, 'rating')]"
            ]
            for sel in rating_selectors:
                try:
                    rating_elem = element.find_element(By.XPATH, sel)
                    rating_text = rating_elem.get_attribute('aria-label') or rating_elem.text
                    rating_match = re.search(r'(\d+\.?\d*)', rating_text)
                    if rating_match:
                        review_data['rating'] = float(rating_match.group(1))
                        break
                except:
                    continue
            
            if 'rating' not in review_data:
                review_data['rating'] = None
            
            # Title
            title_selectors = [
                ".//*[contains(@class, 'title')]",
                ".//*[contains(@class, 'headline')]",
                ".//h3", ".//h4"
            ]
            title = ''
            for sel in title_selectors:
                try:
                    title_elem = element.find_element(By.XPATH, sel)
                    title = title_elem.text.strip()
                    if title and len(title) < 200:
                        break
                except:
                    continue
            review_data['title'] = title
            
            # Review text
            text_selectors = [
                ".//*[contains(@class, 'review-text')]",
                ".//*[contains(@class, 'review-body')]",
                ".//*[contains(@class, 'comment')]",
                ".//p"
            ]
            review_text = ''
            for sel in text_selectors:
                try:
                    text_elem = element.find_element(By.XPATH, sel)
                    text = text_elem.text.strip()
                    if text and len(text) > len(review_text):
                        review_text = text
                except:
                    continue
            
            if not review_text:
                review_text = element.text
            
            review_data['review_text'] = review_text
            
            # Date
            date_selectors = [
                ".//*[contains(@class, 'date')]",
                ".//*[contains(@class, 'time')]"
            ]
            date_text = ''
            for sel in date_selectors:
                try:
                    date_elem = element.find_element(By.XPATH, sel)
                    date_text = date_elem.text.strip()
                    if date_text:
                        break
                except:
                    continue
            review_data['date'] = date_text
            
            # Verified purchase
            verified_text = element.text.lower()
            review_data['verified_purchase'] = 'verified' in verified_text
            
            return review_data if review_data.get('review_text') and len(review_data['review_text']) > 10 else None
            
        except Exception as e:
            return None
    
    def scrape_and_analyze(self, url: str, max_reviews: int = 50) -> Dict:
        """Scrape reviews and analyze sentiment"""
        product_id = self.extract_product_id(url)
        
        if not product_id:
            raise ValueError("Invalid Walmart URL")
        
        print(f"\nProduct ID: {product_id}")
        print("Loading product page...")
        
        self.driver.get(url)
        self.human_delay(2.0, 3.5)
        
        # Scroll and click reviews
        print("Navigating to reviews section...")
        
        for _ in range(2):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            self.human_delay(0.5, 1.0)
        
        # Click Reviews tab
        review_tab_selectors = [
            "//button[contains(text(), 'Reviews')]",
            "//a[contains(text(), 'Reviews')]"
        ]
        
        for selector in review_tab_selectors:
            try:
                tab = self.driver.find_element(By.XPATH, selector)
                if tab.is_displayed():
                    print("Clicking Reviews tab...")
                    self.click_element_safely(tab)
                    self.human_delay(1.5, 2.5)
                    break
            except:
                continue
        
        # Click "View all reviews"
        see_all_selectors = [
            "//button[contains(text(), 'View all reviews')]",
            "//a[contains(text(), 'View all reviews')]"
        ]
        
        for selector in see_all_selectors:
            try:
                button = self.driver.find_element(By.XPATH, selector)
                if button.is_displayed():
                    print("Clicking 'View all reviews'...")
                    self.click_element_safely(button)
                    self.human_delay(2.0, 3.5)
                    break
            except:
                continue
        
        # Extract reviews
        print(f"\nExtracting up to {max_reviews} reviews...")
        
        review_selectors = [
            "//*[contains(@data-testid, 'review')]",
            "//*[contains(@class, 'review-')]",
            "//div[contains(@class, 'customer-review')]"
        ]
        
        review_elements = []
        for selector in review_selectors:
            try:
                elements = self.driver.find_elements(By.XPATH, selector)
                filtered = [e for e in elements if len(e.text) > 50]
                if filtered and len(filtered) > len(review_elements):
                    review_elements = filtered
            except:
                continue
        
        reviews = []
        for elem in review_elements[:max_reviews]:
            review_data = self.extract_review_from_element(elem)
            if review_data:
                # Analyze sentiment
                sentiment_result = self.analyzer.predict_sentiment(
                    review_data['review_text'],
                    review_data.get('title', '')
                )
                
                review_data.update(sentiment_result)
                reviews.append(review_data)
        
        print(f"✓ Extracted and analyzed {len(reviews)} reviews")
        
        return {
            "product_id": product_id,
            "product_url": url,
            "reviews": reviews,
            "analyzed_at": datetime.now().isoformat()
        }


def print_sample_reviews(reviews: List[Dict], sentiment_type: str, n_samples: int = 3):
    """Print sample reviews for a sentiment type"""
    filtered = [r for r in reviews if r.get('sentiment') == sentiment_type]
    
    if not filtered:
        print(f"  No {sentiment_type} reviews found")
        return
    
    # Sort by confidence
    filtered.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    
    samples = filtered[:n_samples]
    
    for i, review in enumerate(samples, 1):
        print(f"\n  Sample {i}:")
        print(f"  Rating: {review.get('rating', 'N/A')}/5")
        print(f"  Confidence: {review.get('confidence', 0)*100:.1f}%")
        
        probs = review.get('probabilities', {})
        print(f"  Probabilities: ", end="")
        print(f"Neg={probs.get('negative', 0)*100:.1f}% ", end="")
        print(f"Neu={probs.get('neutral', 0)*100:.1f}% ", end="")
        print(f"Pos={probs.get('positive', 0)*100:.1f}%")
        
        if review.get('title'):
            print(f"  Title: {review['title']}")
        
        text = review.get('review_text', '')
        if len(text) > 200:
            text = text[:200] + "..."
        print(f"  Review: {text}")
        
        if review.get('reviewer_name'):
            print(f"  Reviewer: {review['reviewer_name']}")


def main():
    """Main inference function"""
    print("\n" + "="*60)
    print("WALMART SENTIMENT INFERENCE - TF-IDF + SVM")
    print("="*60)
    print("\nAnalyze sentiment of Walmart product reviews using trained model")
    print()
    
    analyzer = None
    scraper = None
    
    try:
        # Load trained model automatically
        print("="*60)
        print("LOADING TRAINED MODEL")
        print("="*60)
        print()
        
        analyzer = TFIDFSentimentAnalyzer()
        
        # Get product URL
        print("\n" + "="*60)
        print("PRODUCT URL INPUT")
        print("="*60)
        
        url = input("\nEnter Walmart product URL: ").strip()
        
        if not url:
            print("No URL provided. Exiting...")
            return
        
        # Initialize scraper
        print("\n" + "="*60)
        print("INITIALIZING SCRAPER")
        print("="*60)
        
        scraper = WalmartReviewInference(analyzer, headless=False)
        
        # Scrape and analyze
        print("\n" + "="*60)
        print("SCRAPING & ANALYZING REVIEWS")
        print("="*60)
        
        max_reviews = 50
        max_input = input(f"\nMax reviews to analyze (default {max_reviews}): ").strip()
        if max_input:
            try:
                max_reviews = int(max_input)
            except:
                print(f"Invalid input, using default {max_reviews}")
        
        result = scraper.scrape_and_analyze(url, max_reviews)
        reviews = result['reviews']
        
        if not reviews:
            print("\nNo reviews found!")
            return
        
        # Analyze results
        print("\n" + "="*60)
        print("SENTIMENT ANALYSIS RESULTS")
        print("="*60)
        
        positive = [r for r in reviews if r.get('sentiment') == 'positive']
        negative = [r for r in reviews if r.get('sentiment') == 'negative']
        neutral = [r for r in reviews if r.get('sentiment') == 'neutral']
        
        print(f"\nTotal Reviews Analyzed: {len(reviews)}")
        print(f"\nSentiment Distribution:")
        print(f"  Positive: {len(positive):3d} ({len(positive)/len(reviews)*100:5.1f}%)")
        print(f"  Negative: {len(negative):3d} ({len(negative)/len(reviews)*100:5.1f}%)")
        print(f"  Neutral:  {len(neutral):3d} ({len(neutral)/len(reviews)*100:5.1f}%)")
        
        # Calculate average confidence
        confidences = [r.get('confidence', 0) for r in reviews]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        print(f"\nAverage Confidence: {avg_confidence*100:.2f}%")
        
        # Calculate average probabilities
        avg_probs = {
            'negative': np.mean([r.get('probabilities', {}).get('negative', 0) for r in reviews]),
            'neutral': np.mean([r.get('probabilities', {}).get('neutral', 0) for r in reviews]),
            'positive': np.mean([r.get('probabilities', {}).get('positive', 0) for r in reviews])
        }
        
        print(f"\nAverage Probabilities:")
        print(f"  Negative: {avg_probs['negative']*100:.2f}%")
        print(f"  Neutral:  {avg_probs['neutral']*100:.2f}%")
        print(f"  Positive: {avg_probs['positive']*100:.2f}%")
        
        # Rating analysis
        ratings = [r.get('rating') for r in reviews if r.get('rating')]
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            print(f"\nAverage Rating: {avg_rating:.2f}/5.00")
            
            # Sentiment vs rating
            pos_ratings = [r.get('rating') for r in positive if r.get('rating')]
            neg_ratings = [r.get('rating') for r in negative if r.get('rating')]
            neu_ratings = [r.get('rating') for r in neutral if r.get('rating')]
            
            print(f"\nSentiment vs Rating:")
            if pos_ratings:
                print(f"  Positive reviews avg rating: {sum(pos_ratings)/len(pos_ratings):.2f}")
            if neg_ratings:
                print(f"  Negative reviews avg rating: {sum(neg_ratings)/len(neg_ratings):.2f}")
            if neu_ratings:
                print(f"  Neutral reviews avg rating: {sum(neu_ratings)/len(neu_ratings):.2f}")
        
        # Sample reviews
        print("\n" + "="*60)
        print("SAMPLE POSITIVE REVIEWS")
        print("="*60)
        print_sample_reviews(reviews, 'positive', 3)
        
        print("\n" + "="*60)
        print("SAMPLE NEGATIVE REVIEWS")
        print("="*60)
        print_sample_reviews(reviews, 'negative', 3)
        
        print("\n" + "="*60)
        print("SAMPLE NEUTRAL REVIEWS")
        print("="*60)
        print_sample_reviews(reviews, 'neutral', 3)
        
        # Save results
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)
        
        save = input("\nSave results to JSON? (y/n): ").strip().lower()
        if save == 'y':
            filename = f"inference_results_{result['product_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            output = {
                "metadata": {
                    "product_id": result['product_id'],
                    "product_url": result['product_url'],
                    "total_reviews": len(reviews),
                    "positive_count": len(positive),
                    "negative_count": len(negative),
                    "neutral_count": len(neutral),
                    "average_confidence": round(avg_confidence, 4),
                    "average_probabilities": {k: round(v, 4) for k, v in avg_probs.items()},
                    "average_rating": round(avg_rating, 2) if ratings else None,
                    "model_type": "TF-IDF + LinearSVC",
                    "analyzed_at": result['analyzed_at']
                },
                "reviews": {
                    "all": reviews,
                    "positive": positive,
                    "negative": negative,
                    "neutral": neutral
                }
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Results saved to {filename}")
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if scraper:
            print("\nClosing browser...")
            scraper.close()


if __name__ == "__main__":
    main()