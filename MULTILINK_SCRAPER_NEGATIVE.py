import time
import json
import csv
import re
import random
import os
import tempfile
import shutil
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch


class WalmartReviewScraper:
    def __init__(self, headless: bool = False):
        """Initialize the Walmart Review Scraper with RoBERTa sentiment analysis"""
        print("Starting Chrome browser with anti-detection...")
        
        options = uc.ChromeOptions()
        if headless:
            options.add_argument('--headless=new')
        
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-images')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-plugins')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.page_load_strategy = 'eager'
        
        self.temp_profile_dir = tempfile.mkdtemp(prefix='walmart_scraper_')
        options.add_argument(f'--user-data-dir={self.temp_profile_dir}')
        print(f"Using temporary profile: {self.temp_profile_dir}")
        
        self.driver = uc.Chrome(options=options, version_main=None)
        self.wait = WebDriverWait(self.driver, 8)
        self.short_wait = WebDriverWait(self.driver, 3)
        
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
            print("✓ Optimized for social media and review text")
            print("✓ FILTERING: Only NEGATIVE reviews will be saved")
            
        except Exception as e:
            print(f"Error loading RoBERTa model: {e}")
            print("Falling back to rating-based sentiment...")
            self.sentiment_pipeline = None
        
        print("Browser started successfully")
    
    def close(self):
        """Explicitly close the browser and clean up temporary profile"""
        if hasattr(self, 'driver'):
            try:
                if self.driver:
                    self.driver.quit()
            except:
                pass
        
        if hasattr(self, 'temp_profile_dir'):
            try:
                if os.path.exists(self.temp_profile_dir):
                    shutil.rmtree(self.temp_profile_dir, ignore_errors=True)
                    print(f"✓ Cleaned up temporary profile: {self.temp_profile_dir}")
            except Exception as e:
                print(f"Warning: Could not delete temp profile: {e}")
    
    def __del__(self):
        """Clean up the browser when done"""
        try:
            self.close()
        except:
            pass
    
    def human_delay(self, min_sec: float = 2.0, max_sec: float = 5.0):
        """Random delay to mimic human behavior"""
        time.sleep(random.uniform(min_sec, max_sec))
    
    def simulate_human_movement(self):
        """Simulate random mouse movements"""
        try:
            actions = ActionChains(self.driver)
            for _ in range(random.randint(1, 3)):
                x = random.randint(50, 300)
                y = random.randint(50, 300)
                actions.move_by_offset(x, y)
            actions.perform()
        except:
            pass
    
    def check_captcha_quick(self) -> bool:
        """Quick non-blocking check for CAPTCHA during scraping"""
        captcha_indicators = [
            "//iframe[contains(@src, 'captcha')]",
            "//iframe[contains(@src, 'challenge')]",
            "//*[contains(@id, 'px-captcha')]",
            "//*[contains(text(), 'Press & Hold')]",
            "//*[contains(text(), 'press and hold')]",
            "//*[contains(@class, 'captcha') and not(contains(@style, 'display: none'))]",
            "//div[@data-testid='captcha-modal']",
            "//div[contains(@class, 'Modal') and contains(., 'Press')]"
        ]
        
        for indicator in captcha_indicators:
            try:
                elements = self.driver.find_elements(By.XPATH, indicator)
                for element in elements:
                    try:
                        if element.is_displayed():
                            return True
                    except:
                        pass
            except:
                continue
        
        return False
    
    def classify_sentiment(self, review_text: str, rating: Optional[float], title: str = "") -> Dict:
        """Classify review sentiment using RoBERTa"""
        text_to_analyze = f"{title} {review_text}".strip()
        
        if not text_to_analyze:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "score": 0.0,
                "method": "default"
            }
        
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
            if len(text_to_analyze) > 2000:
                text_to_analyze = text_to_analyze[:2000]
            
            result = self.sentiment_pipeline(text_to_analyze)[0]
            raw_label = result['label'].lower()
            
            if raw_label in ['negative', 'label_0']:
                sentiment = "negative"
            elif raw_label in ['neutral', 'label_1']:
                sentiment = "neutral"
            elif raw_label in ['positive', 'label_2']:
                sentiment = "positive"
            else:
                sentiment = "neutral"
            
            confidence = result['score']
            
            if sentiment == "negative":
                score = (1 - confidence) * 0.5
            elif sentiment == "neutral":
                score = 0.5
            else:
                score = 0.5 + (confidence * 0.5)
            
            method = "roberta"
            
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
            
            return {
                "sentiment": sentiment,
                "confidence": round(confidence, 4),
                "score": round(score, 4),
                "roberta_label": result['label'],
                "method": method
            }
            
        except Exception as e:
            print(f"RoBERTa error: {e}")
            if rating is not None:
                if rating >= 4:
                    return {"sentiment": "positive", "confidence": 0.8, "score": 0.8, "method": "rating_fallback"}
                elif rating <= 2:
                    return {"sentiment": "negative", "confidence": 0.8, "score": 0.2, "method": "rating_fallback"}
                else:
                    return {"sentiment": "neutral", "confidence": 0.7, "score": 0.5, "method": "rating_fallback"}
            else:
                return {"sentiment": "neutral", "confidence": 0.5, "score": 0.5, "method": "default"}
    
    def extract_product_id(self, url: str) -> Optional[str]:
        """Extract product ID from Walmart URL"""
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
    
    def check_for_errors(self) -> Optional[str]:
        """Check if page has loading errors or is unavailable"""
        try:
            error_indicators = [
                "//*[contains(text(), 'sorry') and contains(text(), 'busy')]",
                "//*[contains(text(), 'Something went wrong')]",
                "//*[contains(text(), 'Page not found')]",
                "//*[contains(text(), 'temporarily unavailable')]",
                "//*[contains(text(), 'Try again later')]",
                "//*[contains(text(), '404')]",
                "//*[contains(text(), 'Error')]",
                "//*[contains(@class, 'error-page')]",
                "//*[contains(@class, 'error-message')]",
                "//h1[contains(text(), 'Oops')]"
            ]
            
            for indicator in error_indicators:
                try:
                    elements = self.driver.find_elements(By.XPATH, indicator)
                    for element in elements:
                        if element.is_displayed():
                            error_text = element.text[:100]
                            return error_text
                except:
                    continue
            
            return None
        except:
            return None
    
    def find_element_smart(self, selectors: List[str], parent=None, timeout=3) -> Optional[any]:
        """Try multiple selectors and return first visible match"""
        search_context = parent or self.driver
        
        for selector in selectors:
            try:
                element = search_context.find_element(By.XPATH, selector)
                if element.is_displayed():
                    return element
            except:
                continue
        return None
    
    def click_element_safely(self, element):
        """Click element using JavaScript for reliability"""
        try:
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center', behavior: 'instant'});", element)
            self.human_delay(0.2, 0.5)
            self.driver.execute_script("arguments[0].click();", element)
            return True
        except:
            return False
    
    def click_next_page(self, current_page: int) -> Tuple[bool, bool]:
        """
        Optimized page navigation.
        Returns (success: bool, captcha_detected: bool)
        """
        if self.check_captcha_quick():
            print("\n⚠️ CAPTCHA DETECTED before pagination click!")
            return False, True
        
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        self.human_delay(0.3, 0.7)
        
        next_page_num = current_page + 1
        print(f"Looking for page {next_page_num}...")
        
        all_selectors = [
            f"//button[@aria-label='page {next_page_num}']",
            f"//a[@aria-label='page {next_page_num}']",
            f"//button[normalize-space(text())='{next_page_num}']",
            f"//a[normalize-space(text())='{next_page_num}']",
            "//button[contains(@aria-label, 'next page') and not(@disabled)]",
            "//a[contains(@aria-label, 'next page')]",
            "//button[contains(., '›') and not(@disabled)]",
            "//a[contains(., '›')]"
        ]
        
        for selector in all_selectors:
            try:
                button = self.driver.find_element(By.XPATH, selector)
                if button.is_displayed() and button.is_enabled():
                    disabled = button.get_attribute('disabled')
                    aria_disabled = button.get_attribute('aria-disabled')
                    
                    if disabled or aria_disabled == 'true':
                        continue
                    
                    print(f"Found navigation button, clicking...")
                    
                    if self.check_captcha_quick():
                        print("\n⚠️ CAPTCHA DETECTED - stopping pagination!")
                        return False, True
                    
                    if self.click_element_safely(button):
                        self.wait.until(
                            lambda d: d.execute_script("return document.readyState") == "complete"
                        )
                        self.human_delay(0.5, 1.2)
                        
                        if self.check_captcha_quick():
                            print("\n⚠️ CAPTCHA DETECTED after page load!")
                            return False, True
                        
                        return True, False
            except:
                continue
        
        print(f"No more pages found")
        return False, False
    
    def extract_reviews_from_current_page(self, seen_texts: set) -> Tuple[List[Dict], bool]:
        """
        Extract NEGATIVE reviews only from the current page.
        Returns (reviews_list, captcha_detected)
        """
        if self.check_captcha_quick():
            print("\n⚠️ CAPTCHA DETECTED during extraction!")
            return [], True
        
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        self.human_delay(0.5, 1.0)
        
        if self.check_captcha_quick():
            print("\n⚠️ CAPTCHA DETECTED after scrolling!")
            return [], True
        
        review_selectors = [
            "//*[contains(@data-testid, 'review')]",
            "//*[contains(@class, 'review-') and not(contains(@class, 'button'))]",
            "//div[contains(@class, 'customer-review')]",
            "//article[contains(@class, 'review')]",
            "//*[@itemprop='review']"
        ]
        
        review_elements = []
        for selector in review_selectors:
            try:
                elements = self.driver.find_elements(By.XPATH, selector)
                filtered_elements = [e for e in elements if len(e.text) > 50]
                
                if filtered_elements and len(filtered_elements) > len(review_elements):
                    review_elements = filtered_elements
            except:
                continue
        
        if not review_elements:
            return [], False
        
        negative_reviews = []
        filtered_count = 0
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.extract_review_from_element, elem) 
                      for elem in review_elements]
            results = [f.result() for f in futures]
        
        for review_data in results:
            if review_data and review_data.get('review_text'):
                review_text = review_data['review_text']
                if review_text not in seen_texts and len(review_text) > 10:
                    sentiment_result = self.classify_sentiment(
                        review_data['review_text'],
                        review_data.get('rating'),
                        review_data.get('title', '')
                    )
                    
                    # FILTER: Only keep negative reviews
                    if sentiment_result.get('sentiment') == 'negative':
                        review_data.update(sentiment_result)
                        negative_reviews.append(review_data)
                        seen_texts.add(review_text)
                    else:
                        filtered_count += 1
        
        if filtered_count > 0:
            print(f"  Filtered out {filtered_count} non-negative reviews")
        
        return negative_reviews, False
    
    def extract_review_from_element(self, element) -> Optional[Dict]:
        """Extract review data from a Selenium WebElement"""
        try:
            review_data = {}
            element_text = element.text
            
            name_selectors = [
                ".//*[contains(@class, 'reviewer')]",
                ".//*[contains(@class, 'author')]",
                ".//*[contains(@class, 'name')]",
                ".//*[contains(@class, 'user')]"
            ]
            reviewer_name = 'Anonymous'
            for sel in name_selectors:
                try:
                    reviewer = element.find_element(By.XPATH, sel)
                    name = reviewer.text.strip()
                    if name and 0 < len(name) < 50:
                        reviewer_name = name
                        break
                except:
                    continue
            review_data['reviewer_name'] = reviewer_name
            
            rating_selectors = [
                ".//*[contains(@aria-label, 'star')]",
                ".//*[contains(@class, 'rating')]",
                ".//*[contains(@class, 'stars')]"
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
            
            title_selectors = [
                ".//*[contains(@class, 'title')]",
                ".//*[contains(@class, 'headline')]",
                ".//h3",
                ".//h4"
            ]
            title = ''
            for sel in title_selectors:
                try:
                    title_elem = element.find_element(By.XPATH, sel)
                    title = title_elem.text.strip()
                    if title and 0 < len(title) < 200:
                        break
                except:
                    continue
            review_data['title'] = title
            
            text_selectors = [
                ".//*[contains(@class, 'review-text')]",
                ".//*[contains(@class, 'review-body')]",
                ".//*[contains(@class, 'comment')]",
                ".//*[contains(@class, 'content')]",
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
                review_text = element_text
            
            review_data['review_text'] = review_text
            
            date_selectors = [
                ".//*[contains(@class, 'date')]",
                ".//*[contains(@class, 'time')]",
                ".//*[contains(@class, 'timestamp')]"
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
            
            verified_text = element_text.lower()
            review_data['verified_purchase'] = 'verified' in verified_text
            
            helpful_match = re.search(r'(\d+)\s*helpful', element_text, re.IGNORECASE)
            if helpful_match:
                review_data['helpful_count'] = int(helpful_match.group(1))
            else:
                review_data['helpful_count'] = 0
            
            return review_data if review_data.get('review_text') and len(review_data['review_text']) > 10 else None
            
        except Exception as e:
            return None
    
    def save_to_json(self, reviews: List[Dict], filename: str = None):
        """Save NEGATIVE reviews to JSON file"""
        if filename is None:
            filename = f"walmart_reviews_negative_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # All reviews should be negative, but double-check
        negative_reviews = [r for r in reviews if r.get('sentiment') == 'negative']
        
        all_confidences = [r.get('confidence', 0) for r in negative_reviews if r.get('confidence')]
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        
        all_scores = [r.get('score', 0) for r in negative_reviews if r.get('score') is not None]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        
        output = {
            "metadata": {
                "filter": "NEGATIVE_ONLY",
                "total_reviews": len(negative_reviews),
                "average_confidence": round(avg_confidence, 4),
                "average_score": round(avg_score, 4),
                "sentiment_analyzer": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "device": "GPU (CUDA)" if self.device == 0 else "CPU",
                "scraped_at": datetime.now().isoformat()
            },
            "reviews": negative_reviews
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"Negative reviews saved to {filename}")
        print(f"Total Negative Reviews: {len(negative_reviews)}")
        print(f"Average Confidence: {avg_confidence:.2%} | Avg Score: {avg_score:.3f}")
    
    def save_to_csv(self, reviews: List[Dict], filename: str = None):
        """Save NEGATIVE reviews to CSV file"""
        if filename is None:
            filename = f"walmart_reviews_negative_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        if not reviews:
            print("No negative reviews to save")
            return
        
        # Filter to only negative (should already be filtered)
        negative_reviews = [r for r in reviews if r.get('sentiment') == 'negative']
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['product_id', 'product_name', 'reviewer_name', 'rating', 'sentiment', 
                         'confidence', 'score', 'roberta_label', 'method', 'title', 'review_text', 
                         'date', 'verified_purchase', 'helpful_count']
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            
            writer.writeheader()
            for review in negative_reviews:
                writer.writerow(review)
        
        print(f"Negative reviews saved to {filename}")
    
    def scrape_reviews(self, url: str, product_name: str = None, auto_skip_captcha: bool = True) -> Dict:
        """
        Scrape NEGATIVE reviews only from a Walmart product page.
        Handles runtime CAPTCHA detection during extraction.
        """
        product_id = self.extract_product_id(url)
        
        if not product_id:
            raise ValueError("Invalid Walmart URL. Could not extract product ID.")
        
        print(f"\nProduct ID: {product_id}")
        if product_name:
            print(f"Product Name: {product_name}")
        print("Filter: NEGATIVE REVIEWS ONLY")
        
        print("\nLoading product page...")
        self.driver.get(url)
        self.human_delay(2.0, 3.5)
        
        if self.check_captcha_quick():
            print("CAPTCHA detected on initial load - skipping...")
            return {
                "product_id": product_id,
                "product_url": url,
                "reviews": [],
                "error": "CAPTCHA detected",
                "skipped": True,
                "captcha_detected": True
            }
        
        error_message = self.check_for_errors()
        if error_message:
            print(f"⚠ Page Error Detected: {error_message}")
            return {
                "product_id": product_id, 
                "product_url": url, 
                "reviews": [], 
                "error": f"Page error: {error_message}",
                "skipped": True
            }
        
        self.simulate_human_movement()
        
        print("Scrolling page to load content...")
        for i in range(2):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")
            self.human_delay(0.5, 1.0)
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            self.human_delay(0.5, 1.0)
            
            if self.check_captcha_quick():
                print("CAPTCHA detected during scroll - skipping...")
                return {
                    "product_id": product_id,
                    "product_url": url,
                    "reviews": [],
                    "error": "CAPTCHA detected during scroll",
                    "skipped": True,
                    "captcha_detected": True
                }
        
        error_message = self.check_for_errors()
        if error_message:
            print(f"⚠ Page Error Detected After Scroll: {error_message}")
            return {
                "product_id": product_id, 
                "product_url": url, 
                "reviews": [], 
                "error": f"Page error: {error_message}",
                "skipped": True
            }
        
        print("Looking for Reviews tab...")
        review_tab_selectors = [
            "//button[contains(text(), 'Reviews')]",
            "//a[contains(text(), 'Reviews')]",
            "//*[@role='tab'][contains(text(), 'Reviews')]"
        ]
        
        tab = self.find_element_smart(review_tab_selectors)
        if tab:
            print("Found Reviews tab, clicking...")
            self.click_element_safely(tab)
            self.human_delay(1.5, 2.5)
        
        print("Looking for 'View all reviews' button...")
        see_all_selectors = [
            "//button[contains(text(), 'View all reviews')]",
            "//a[contains(text(), 'View all reviews')]",
            "//button[contains(text(), 'See all reviews')]",
            "//a[contains(text(), 'See all reviews')]"
        ]
        
        see_all_button = self.find_element_smart(see_all_selectors)
        if see_all_button:
            print(f"Found button: '{see_all_button.text}', clicking...")
            self.click_element_safely(see_all_button)
            self.human_delay(2.0, 3.5)
            
            if self.check_captcha_quick():
                print("CAPTCHA detected after clicking reviews - skipping...")
                return {
                    "product_id": product_id,
                    "product_url": url,
                    "reviews": [],
                    "error": "CAPTCHA detected after reviews click",
                    "skipped": True,
                    "captcha_detected": True
                }
        
        print("Scrolling to pagination area...")
        for _ in range(2):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            self.human_delay(0.5, 1.0)
        
        print("\nCollecting NEGATIVE reviews from all pages...")
        all_negative_reviews = []
        seen_texts = set()
        current_page = 1
        max_pages = 50
        
        start_time = time.time()
        
        while current_page <= max_pages:
            page_start = time.time()
            print(f"\nPage {current_page}: Extracting negative reviews...")
            
            page_reviews, captcha_detected = self.extract_reviews_from_current_page(seen_texts)
            
            if captcha_detected:
                print(f"\nStopping extraction - CAPTCHA detected on page {current_page}")
                return {
                    "product_id": product_id,
                    "product_url": url,
                    "reviews": all_negative_reviews,
                    "error": f"CAPTCHA detected on page {current_page}",
                    "skipped": True,
                    "captcha_detected": True,
                    "partial_reviews": len(all_negative_reviews)
                }
            
            all_negative_reviews.extend(page_reviews)
            
            page_time = time.time() - page_start
            print(f"Got {len(page_reviews)} NEGATIVE reviews from page {current_page} (Total: {len(all_negative_reviews)}) [{page_time:.1f}s]")
            
            next_page_result, captcha_detected = self.click_next_page(current_page)
            
            if captcha_detected:
                print(f"\nStopping pagination - CAPTCHA detected")
                return {
                    "product_id": product_id,
                    "product_url": url,
                    "reviews": all_negative_reviews,
                    "error": "CAPTCHA detected during pagination",
                    "skipped": True,
                    "captcha_detected": True,
                    "partial_reviews": len(all_negative_reviews)
                }
            
            if not next_page_result:
                print(f"Finished at page {current_page}")
                break
            
            current_page += 1
            self.human_delay(0.8, 1.5)
        
        elapsed = time.time() - start_time
        print(f"\nSuccessfully extracted {len(all_negative_reviews)} NEGATIVE reviews in {elapsed:.1f}s")
        if all_negative_reviews:
            print(f"Average: {elapsed/len(all_negative_reviews):.2f}s per review\n")
        
        return {
            "product_id": product_id,
            "product_url": url,
            "product_name": product_name,
            "reviews": all_negative_reviews,
            "filter": "NEGATIVE_ONLY",
            "scraped_at": datetime.now().isoformat()
        }


def read_urls_from_file(txt_file: str) -> List[str]:
    """Read URLs from a text file, one URL per line"""
    urls = []
    
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"File not found: {txt_file}")
    
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                urls.append(line)
    
    return urls


def mark_url_as_completed(txt_file: str, completed_url: str, product_id: str = None):
    """Remove completed URL from original file and add it to completed file"""
    try:
        if not os.path.exists(txt_file):
            return
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        remaining_lines = []
        for line in lines:
            if line.strip() and completed_url not in line:
                remaining_lines.append(line)
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.writelines(remaining_lines)
        
        completed_file = txt_file.replace('.txt', '_completed.txt')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(completed_file, 'a', encoding='utf-8') as f:
            if product_id:
                f.write(f"{completed_url} # Completed at {timestamp} | Product ID: {product_id}\n")
            else:
                f.write(f"{completed_url} # Completed at {timestamp}\n")
        
        print(f"✓ Marked as completed and moved to {completed_file}")
        
    except Exception as e:
        print(f"Warning: Could not update URL files: {e}")


def main():
    """Main scraping function - NEGATIVE REVIEWS ONLY"""
    
    print("\n" + "="*60)
    print("WALMART REVIEW SCRAPER - NEGATIVE REVIEWS ONLY")
    print("="*60)
    print("\nUsing CardiffNLP Twitter-RoBERTa Model")
    print("✓ State-of-the-art transformer-based sentiment analysis")
    print("✓ Trained on 124M tweets for social media/review text")
    print("✓ GPU-accelerated (if CUDA available)")
    print("✓ FILTERING: Only NEGATIVE sentiment reviews saved")
    print("✓ Positive and Neutral reviews are filtered out")
    print("✓ Auto-resume: Completed URLs are moved to *_completed.txt")
    print("✓ Runtime CAPTCHA detection during scraping")
    print("✓ Auto-skip on CAPTCHA or bot detection\n")
    
    scraper = None
    input_file = None
    
    try:
        print("Choose input method:")
        print("1. Enter a single URL manually")
        print("2. Load URLs from a text file")
        choice = input("Enter choice (1 or 2): ").strip()
        
        urls = []
        
        if choice == "1":
            product_url = input("\nEnter Walmart product URL: ").strip()
            if not product_url:
                print("No URL provided. Exiting...")
                return
            urls = [product_url]
        
        elif choice == "2":
            txt_file = input("\nEnter path to text file (e.g., links.txt): ").strip()
            if not txt_file:
                txt_file = "links.txt"
            
            try:
                urls = read_urls_from_file(txt_file)
                input_file = txt_file
                
                if not urls:
                    print(f"\n{txt_file} is empty or all URLs have been processed!")
                    completed_file = txt_file.replace('.txt', '_completed.txt')
                    if os.path.exists(completed_file):
                        print(f"Check {completed_file} for completed URLs.")
                    return
                
                print(f"\nFound {len(urls)} URLs remaining in {txt_file}")
                
                completed_file = txt_file.replace('.txt', '_completed.txt')
                if os.path.exists(completed_file):
                    with open(completed_file, 'r', encoding='utf-8') as f:
                        completed_count = sum(1 for line in f if line.strip() and not line.strip().startswith('#'))
                    print(f"(Already completed: {completed_count} URLs - see {completed_file})")
                
                for i, url in enumerate(urls, 1):
                    print(f"  {i}. {url[:80]}{'...' if len(url) > 80 else ''}")
                
                confirm = input("\nProceed with scraping? (y/n): ").strip().lower()
                if confirm != 'y':
                    print("Scraping cancelled.")
                    return
                    
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return
        else:
            print("Invalid choice. Exiting...")
            return
        
        print("\n" + "="*60)
        print("NEGATIVE REVIEWS ONLY MODE")
        print("="*60)
        print("Only reviews classified as NEGATIVE will be saved")
        print("Positive and neutral reviews will be filtered out\n")
        
        all_products_data = []
        all_negative_reviews_combined = []
        skipped_urls = []
        captcha_skipped_count = 0
        
        for idx, url in enumerate(urls, 1):
            print("\n" + "="*60)
            print(f"SCRAPING PRODUCT {idx}/{len(urls)}")
            print("="*60)
            
            print(f"\nStarting fresh browser session for product {idx}...")
            scraper = WalmartReviewScraper(headless=False)
            
            try:
                result = scraper.scrape_reviews(url, auto_skip_captcha=True)
                
                if result.get('captcha_detected'):
                    print(f"\nCAPTCHA detected - URL auto-skipped")
                    captcha_skipped_count += 1
                    skipped_urls.append({"url": url, "reason": "CAPTCHA", "product_id": result.get('product_id', 'UNKNOWN')})
                    if input_file:
                        mark_url_as_completed(input_file, url, f"CAPTCHA_{result.get('product_id', 'UNKNOWN')}")
                    continue
                
                if result.get('skipped'):
                    print(f"\nProduct skipped due to page error")
                    skipped_urls.append({"url": url, "reason": "Page Error", "product_id": result.get('product_id', 'UNKNOWN')})
                    if input_file:
                        mark_url_as_completed(input_file, url, f"SKIPPED_{result.get('product_id', 'UNKNOWN')}")
                    continue
                
                if result and result.get('reviews'):
                    reviews = result['reviews']
                    
                    # All reviews should already be negative, but ensure
                    unique_negative_reviews = []
                    seen = set()
                    
                    for review in reviews:
                        if review.get('sentiment') == 'negative':
                            key = review.get('review_text', '')
                            if key and key not in seen and len(key) > 10:
                                seen.add(key)
                                review['product_id'] = result['product_id']
                                review['product_url'] = result['product_url']
                                review['product_name'] = result.get('product_name', f"Product {result['product_id']}")
                                unique_negative_reviews.append(review)
                    
                    if len(reviews) != len(unique_negative_reviews):
                        print(f"Removed {len(reviews) - len(unique_negative_reviews)} duplicates")
                    
                    result['reviews'] = unique_negative_reviews
                    all_products_data.append(result)
                    all_negative_reviews_combined.extend(unique_negative_reviews)
                    
                    product_id = result['product_id']
                    base_filename = f"walmart_reviews_negative_{product_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    with open(f"{base_filename}.json", 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    print(f"\nSaved negative reviews to {base_filename}.json")
                    
                    print(f"\nProduct {idx} Summary:")
                    print(f"  Total NEGATIVE Reviews: {len(unique_negative_reviews)}")
                    
                    confidences = [r.get('confidence', 0) for r in unique_negative_reviews]
                    scores = [r.get('score', 0) for r in unique_negative_reviews if r.get('score') is not None]
                    avg_conf = sum(confidences) / len(confidences) if confidences else 0
                    avg_score = sum(scores) / len(scores) if scores else 0
                    print(f"  Avg Confidence: {avg_conf:.2%} | Avg Score: {avg_score:.3f}")
                    
                    if input_file:
                        mark_url_as_completed(input_file, url, product_id)
                    
                else:
                    print(f"\nNo negative reviews found for product {idx}")
                    skipped_urls.append({"url": url, "reason": "No Negative Reviews", "product_id": result.get('product_id', 'UNKNOWN') if result else 'UNKNOWN'})
                    if input_file:
                        result_id = result.get('product_id', 'UNKNOWN') if result else 'UNKNOWN'
                        mark_url_as_completed(input_file, url, f"NO_NEGATIVE_{result_id}")
                    
            except Exception as e:
                print(f"\nError scraping product {idx}: {e}")
                import traceback
                traceback.print_exc()
                
                skipped_urls.append({"url": url, "reason": f"Error: {str(e)[:50]}", "product_id": "ERROR"})
                
                if input_file:
                    print("\nMarking URL as completed despite error...")
                    mark_url_as_completed(input_file, url, "ERROR")
            
            finally:
                print(f"Closing browser session for product {idx}...")
                if scraper:
                    scraper.close()
                    scraper = None
                time.sleep(1)
            
            if idx < len(urls):
                delay = random.uniform(5, 8)
                print(f"\nWaiting {delay:.1f}s before starting next product...")
                time.sleep(delay)
        
        if all_negative_reviews_combined:
            print("\n" + "="*60)
            print("SAVING COMBINED NEGATIVE REVIEWS")
            print("="*60)
            
            if input_file:
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                combined_filename = f"walmart_reviews_negative_{base_name}_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                combined_filename = f"walmart_reviews_negative_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            all_confidences = [r.get('confidence', 0) for r in all_negative_reviews_combined if r.get('confidence')]
            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
            
            all_scores = [r.get('score', 0) for r in all_negative_reviews_combined if r.get('score') is not None]
            avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
            
            combined_data = {
                "metadata": {
                    "filter": "NEGATIVE_ONLY",
                    "source_file": input_file if input_file else "manual_input",
                    "total_products": len(all_products_data),
                    "total_negative_reviews": len(all_negative_reviews_combined),
                    "average_confidence": round(avg_confidence, 4),
                    "average_score": round(avg_score, 4),
                    "sentiment_analyzer": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                    "device": "GPU (CUDA)" if torch.cuda.is_available() else "CPU",
                    "scraped_at": datetime.now().isoformat()
                },
                "products": all_products_data
            }
            
            with open(f"{combined_filename}.json", 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, indent=2, ensure_ascii=False)
            print(f"\nSaved combined negative reviews to {combined_filename}.json")
            
            with open(f"{combined_filename}.csv", 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['product_id', 'product_name', 'product_url', 'reviewer_name', 'rating', 
                             'sentiment', 'confidence', 'score', 'roberta_label', 'method', 'title', 
                             'review_text', 'date', 'verified_purchase', 'helpful_count']
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                for review in all_negative_reviews_combined:
                    writer.writerow(review)
            print(f"Saved combined negative reviews to {combined_filename}.csv")
            
            print("\n" + "="*60)
            print("OVERALL SUMMARY - NEGATIVE REVIEWS ONLY")
            print("="*60)
            print(f"Products Scraped: {len(all_products_data)}")
            print(f"Total NEGATIVE Reviews: {len(all_negative_reviews_combined)}")
            
            print(f"\nRoBERTa Sentiment Analysis:")
            print(f"  Average Confidence: {avg_confidence:.2%}")
            print(f"  Average Score: {avg_score:.4f} (negative < 0.5)")
            
            methods = {}
            for r in all_negative_reviews_combined:
                method = r.get('method', 'unknown')
                methods[method] = methods.get(method, 0) + 1
            
            print(f"\nSentiment Analysis Methods:")
            for method, count in sorted(methods.items(), key=lambda x: x[1], reverse=True):
                print(f"  {method}: {count} ({count/len(all_negative_reviews_combined)*100:.1f}%)")
            
            ratings = [r['rating'] for r in all_negative_reviews_combined if r.get('rating')]
            if ratings:
                avg_rating = sum(ratings) / len(ratings)
                print(f"\nAverage Rating (Negative Reviews): {avg_rating:.2f} / 5.00")
                
                verified = sum(1 for r in all_negative_reviews_combined if r.get('verified_purchase'))
                print(f"Total Verified Purchases: {verified} ({verified/len(all_negative_reviews_combined)*100:.1f}%)")
                
                print(f"\nRating Distribution (Negative Reviews):")
                for star in range(5, 0, -1):
                    count = sum(1 for r in ratings if r == star)
                    percentage = (count / len(ratings) * 100) if ratings else 0
                    bar = "█" * int(percentage / 2)
                    print(f"  {star} stars: {count:4d} ({percentage:5.1f}%) {bar}")
            
            print("\n" + "="*60)
            print("PER-PRODUCT SUMMARY")
            print("="*60)
            for i, product in enumerate(all_products_data, 1):
                reviews = product['reviews']
                product_id = product['product_id']
                print(f"\nProduct {i} (ID: {product_id}):")
                print(f"  URL: {product['product_url'][:60]}...")
                print(f"  NEGATIVE Reviews: {len(reviews)}")
                
                if reviews:
                    prod_ratings = [r['rating'] for r in reviews if r.get('rating')]
                    if prod_ratings:
                        prod_avg = sum(prod_ratings) / len(prod_ratings)
                        print(f"  Avg Rating: {prod_avg:.2f}/5.00")
                    
                    prod_confidences = [r.get('confidence', 0) for r in reviews if r.get('confidence')]
                    prod_scores = [r.get('score', 0) for r in reviews if r.get('score') is not None]
                    
                    if prod_confidences:
                        prod_avg_conf = sum(prod_confidences) / len(prod_confidences)
                        print(f"  Avg Confidence: {prod_avg_conf:.2%}")
                    
                    if prod_scores:
                        prod_avg_score = sum(prod_scores) / len(prod_scores)
                        print(f"  Avg Score: {prod_avg_score:.3f}")
            
            print("\n" + "="*60)
            print("SCRAPING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"\nRoBERTa analyzed {len(all_negative_reviews_combined)} NEGATIVE reviews")
            print(f"Overall sentiment score: {avg_score:.4f} (negative)")
            print("✓ Only NEGATIVE reviews saved to files")
            print("✓ Positive and neutral reviews were filtered out")
            
            if input_file:
                completed_file = input_file.replace('.txt', '_completed.txt')
                print(f"\nAll URLs processed!")
                print(f"  Original file: {input_file}")
                print(f"  Completed URLs: {completed_file}")
            
            if skipped_urls:
                print(f"\nSKIPPED URLs SUMMARY ({len(skipped_urls)} total)")
                print("="*60)
                if captcha_skipped_count > 0:
                    print(f"CAPTCHA Auto-Skipped: {captcha_skipped_count}")
                
                for u in skipped_urls[:10]:
                    print(f"\n  {u['url'][:70]}...")
                    print(f"  Reason: {u['reason']} | Product ID: {u['product_id']}")
                
                if len(skipped_urls) > 10:
                    print(f"\n  ... and {len(skipped_urls) - 10} more")
        
        else:
            print("\nNo negative reviews were collected from any products.")
            if input_file:
                print(f"\nURLs have been marked as completed in {input_file.replace('.txt', '_completed.txt')}")
            
            if skipped_urls:
                print(f"\nALL URLs WERE SKIPPED ({len(skipped_urls)} total)")
                print("="*60)
                if captcha_skipped_count > 0:
                    print(f"CAPTCHA Auto-Skipped: {captcha_skipped_count}")
                    print("\nTip: Try again later or increase delays between requests")
                
                for u in skipped_urls[:5]:
                    print(f"\n  {u['url'][:70]}...")
                    print(f"  Reason: {u['reason']} | Product ID: {u['product_id']}")
    
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("SCRAPING INTERRUPTED BY USER")
        print("="*60)
        if input_file:
            completed_file = input_file.replace('.txt', '_completed.txt')
            print(f"\nProgress saved!")
            print(f"  Remaining URLs: {input_file}")
            print(f"  Completed URLs: {completed_file}")
            print(f"\nResume by running the script again with the same file.")
        print()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        
        if input_file:
            print(f"\nProgress saved before crash!")
            print(f"  Remaining URLs: {input_file}")
            print(f"  Completed URLs: {input_file.replace('.txt', '_completed.txt')}")
            print(f"\nRestart the script to continue from where it stopped.")
    
    finally:
        if scraper:
            print("\nClosing final browser session...")
            scraper.close()
            time.sleep(0.5)


if __name__ == "__main__":
    main()