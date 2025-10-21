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
from selenium.common.exceptions import TimeoutException, NoSuchElementException
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
    
    def check_for_blocks(self) -> Tuple[bool, str]:
        """Check if page is blocked by anti-bot measures"""
        page_text = self.driver.page_source.lower()
        
        # Check for CAPTCHA
        if 'captcha' in page_text or 'robot' in page_text:
            return True, "CAPTCHA detected"
        
        # Check for "busy" or rate limit messages
        busy_phrases = [
            'sorry, we are busy',
            'too many requests',
            'try again later',
            'temporarily unavailable',
            'access denied',
            'unusual activity'
        ]
        
        for phrase in busy_phrases:
            if phrase in page_text:
                return True, f"Rate limit/Block detected: '{phrase}'"
        
        return False, ""
    
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
    
    def click_next_page(self) -> bool:
        """Try to click next page button"""
        next_button_selectors = [
            "//button[contains(@aria-label, 'next')]",
            "//button[contains(@aria-label, 'Next')]",
            "//a[contains(@aria-label, 'next')]",
            "//a[contains(@aria-label, 'Next')]",
            "//button[contains(text(), 'Next')]",
            "//a[contains(text(), 'Next')]",
            "//*[contains(@class, 'pagination')]//button[not(contains(@disabled, 'true'))]",
        ]
        
        for selector in next_button_selectors:
            try:
                button = self.driver.find_element(By.XPATH, selector)
                if button.is_displayed() and button.is_enabled():
                    print("  Clicking next page...")
                    self.click_element_safely(button)
                    self.human_delay(2.0, 3.5)
                    return True
            except:
                continue
        
        return False
    
    def scrape_and_analyze(self, url: str, max_reviews: int = None) -> Dict:
        """Scrape ALL reviews and analyze sentiment (or up to max_reviews if specified)"""
        product_id = self.extract_product_id(url)
        
        if not product_id:
            raise ValueError("Invalid Walmart URL")
        
        print(f"\nProduct ID: {product_id}")
        print("Loading product page...")
        
        self.driver.get(url)
        self.human_delay(2.0, 3.5)
        
        # Check for blocks
        blocked, reason = self.check_for_blocks()
        if blocked:
            raise Exception(f"Page blocked: {reason}")
        
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
            "//a[contains(text(), 'View all reviews')]",
            "//button[contains(text(), 'See all reviews')]",
            "//a[contains(text(), 'See all reviews')]"
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
        
        # Extract reviews with pagination
        print(f"\nExtracting reviews (limit: {'ALL' if max_reviews is None else max_reviews})...")
        
        review_selectors = [
            "//*[contains(@data-testid, 'review')]",
            "//*[contains(@class, 'review-')]",
            "//div[contains(@class, 'customer-review')]"
        ]
        
        all_reviews = []
        seen_texts = set()  # Deduplicate
        page_num = 1
        no_new_reviews_count = 0
        
        while True:
            # Check for blocks before extracting
            blocked, reason = self.check_for_blocks()
            if blocked:
                print(f"\n⚠ {reason}")
                print(f"Stopping scrape. Total reviews collected: {len(all_reviews)}")
                break
            
            print(f"\n  Page {page_num}:")
            
            # Scroll to load content
            for _ in range(3):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                self.human_delay(0.3, 0.7)
            
            # Find review elements
            review_elements = []
            for selector in review_selectors:
                try:
                    elements = self.driver.find_elements(By.XPATH, selector)
                    filtered = [e for e in elements if len(e.text) > 50]
                    if filtered and len(filtered) > len(review_elements):
                        review_elements = filtered
                except:
                    continue
            
            if not review_elements:
                print("  No review elements found on this page")
                break
            
            # Extract reviews from this page
            page_reviews = 0
            for elem in review_elements:
                # Check if we've reached the limit
                if max_reviews is not None and len(all_reviews) >= max_reviews:
                    print(f"\n✓ Reached maximum review limit ({max_reviews})")
                    break
                
                review_data = self.extract_review_from_element(elem)
                if review_data:
                    # Deduplicate based on review text
                    text_hash = hash(review_data['review_text'])
                    if text_hash not in seen_texts:
                        seen_texts.add(text_hash)
                        
                        # Analyze sentiment
                        sentiment_result = self.analyzer.predict_sentiment(
                            review_data['review_text'],
                            review_data.get('title', '')
                        )
                        
                        review_data.update(sentiment_result)
                        all_reviews.append(review_data)
                        page_reviews += 1
            
            print(f"  Extracted {page_reviews} new reviews (Total: {len(all_reviews)})")
            
            # Check if we've reached the limit
            if max_reviews is not None and len(all_reviews) >= max_reviews:
                break
            
            # If no new reviews on this page, increment counter
            if page_reviews == 0:
                no_new_reviews_count += 1
                if no_new_reviews_count >= 2:
                    print("\n  No new reviews found for 2 consecutive pages. Stopping.")
                    break
            else:
                no_new_reviews_count = 0
            
            # Try to go to next page
            if not self.click_next_page():
                print("\n  No more pages available. Scraping complete.")
                break
            
            page_num += 1
            
            # Safety check - don't scrape forever
            if page_num > 100:
                print("\n⚠ Reached maximum page limit (100). Stopping.")
                break
        
        print(f"\n✓ Extraction complete! Total reviews analyzed: {len(all_reviews)}")
        
        return {
            "product_id": product_id,
            "product_url": url,
            "reviews": all_reviews,
            "total_pages_scraped": page_num,
            "analyzed_at": datetime.now().isoformat()
        }


def generate_conclusion(reviews: List[Dict], avg_rating: float = None) -> str:
    """Generate overall sentiment conclusion"""
    if not reviews:
        return "No reviews available for analysis."
    
    total = len(reviews)
    positive = [r for r in reviews if r.get('sentiment') == 'positive']
    negative = [r for r in reviews if r.get('sentiment') == 'negative']
    neutral = [r for r in reviews if r.get('sentiment') == 'neutral']
    
    pos_pct = len(positive) / total * 100
    neg_pct = len(negative) / total * 100
    neu_pct = len(neutral) / total * 100
    
    # Calculate average confidence
    confidences = [r.get('confidence', 0) for r in reviews]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    # Determine overall sentiment
    if pos_pct > 60:
        overall = "VERY POSITIVE"
        emoji = "🌟"
    elif pos_pct > 45:
        overall = "POSITIVE"
        emoji = "👍"
    elif neg_pct > 50:
        overall = "NEGATIVE"
        emoji = "👎"
    elif neg_pct > 35:
        overall = "MIXED/NEGATIVE"
        emoji = "⚠️"
    elif neu_pct > 50:
        overall = "NEUTRAL"
        emoji = "😐"
    else:
        overall = "MIXED"
        emoji = "🤔"
    
    # Build conclusion
    conclusion = f"""
{'='*60}
OVERALL PRODUCT SENTIMENT CONCLUSION {emoji}
{'='*60}

Based on analysis of {total} customer reviews:

OVERALL SENTIMENT: {overall}

Sentiment Breakdown:
  • Positive Reviews: {len(positive)} ({pos_pct:.1f}%) 
  • Negative Reviews: {len(negative)} ({neg_pct:.1f}%)
  • Neutral Reviews:  {len(neutral)} ({neu_pct:.1f}%)

Model Confidence: {avg_confidence*100:.1f}% average
"""
    
    if avg_rating:
        conclusion += f"Average Star Rating: {avg_rating:.2f}/5.00\n"
    
    conclusion += "\n"
    
    # Add interpretation
    if pos_pct > 60:
        conclusion += "✓ RECOMMENDATION: This product is highly recommended by customers.\n"
        conclusion += "  The overwhelming majority of reviews express satisfaction.\n"
    elif pos_pct > 45:
        conclusion += "✓ RECOMMENDATION: This product is generally well-received.\n"
        conclusion += "  Most customers are satisfied with their purchase.\n"
    elif neg_pct > 50:
        conclusion += "✗ WARNING: This product has significant customer dissatisfaction.\n"
        conclusion += "  More than half of reviewers express negative sentiment.\n"
    elif neg_pct > 35:
        conclusion += "⚠ CAUTION: This product has mixed reviews with notable concerns.\n"
        conclusion += "  A substantial portion of customers report issues.\n"
    elif neu_pct > 50:
        conclusion += "○ NEUTRAL: Customer opinions are largely neutral or ambivalent.\n"
        conclusion += "  The product meets basic expectations without strong feelings.\n"
    else:
        conclusion += "⚠ MIXED: Customer opinions are divided on this product.\n"
        conclusion += "  Consider reading individual reviews to understand specific concerns.\n"
    
    # Add confidence note
    if avg_confidence < 0.5:
        conclusion += "\nNote: Lower model confidence suggests nuanced or ambiguous reviews.\n"
    elif avg_confidence > 0.8:
        conclusion += "\nNote: High model confidence indicates clear sentiment signals.\n"
    
    conclusion += "="*60
    
    return conclusion


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
        
        print("\nBy default, will scrape ALL available reviews.")
        print("This will continue until: no more pages, CAPTCHA, or rate limit.")
        
        limit_input = input("\nSet a maximum limit? (press Enter for ALL, or enter a number): ").strip()
        max_reviews = None
        if limit_input:
            try:
                max_reviews = int(limit_input)
                print(f"Limit set to {max_reviews} reviews")
            except:
                print("Invalid input, will scrape ALL reviews")
        else:
            print("No limit set - will scrape ALL reviews")
        
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
        print(f"Total Pages Scraped: {result.get('total_pages_scraped', 'N/A')}")
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
        avg_rating = None
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
        
        # Generate and print conclusion
        print("\n")
        conclusion = generate_conclusion(reviews, avg_rating)
        print(conclusion)
        
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
                    "total_pages_scraped": result.get('total_pages_scraped'),
                    "positive_count": len(positive),
                    "negative_count": len(negative),
                    "neutral_count": len(neutral),
                    "average_confidence": round(avg_confidence, 4),
                    "average_probabilities": {k: round(v, 4) for k, v in avg_probs.items()},
                    "average_rating": round(avg_rating, 2) if ratings else None,
                    "model_type": "TF-IDF + LinearSVC",
                    "analyzed_at": result['analyzed_at'],
                    "conclusion": conclusion
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