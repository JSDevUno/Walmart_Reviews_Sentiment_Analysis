import time
import json
import csv
import re
import random
import os
from datetime import datetime
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class WalmartReviewScraper:
    def __init__(self, headless: bool = False):
        """Initialize the Walmart Review Scraper with VADER sentiment analysis"""
        print("Starting Chrome browser with anti-detection...")
        
        options = uc.ChromeOptions()
        if headless:
            options.add_argument('--headless=new')
        
        # Performance optimizations
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-images')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-plugins')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.page_load_strategy = 'eager'
        
        # Reuse profile for session persistence
        profile_path = os.path.expanduser("~/.walmart_scraper_profile")
        options.add_argument(f'--user-data-dir={profile_path}')
        
        self.driver = uc.Chrome(options=options, version_main=None)
        self.wait = WebDriverWait(self.driver, 8)
        self.short_wait = WebDriverWait(self.driver, 3)
        
        # Initialize VADER sentiment analyzer
        print("Loading VADER sentiment analyzer...")
        self.vader_analyzer = SentimentIntensityAnalyzer()
        print("✓ VADER sentiment analyzer loaded (lightweight, CPU-optimized)")
        
        print("Browser started successfully")
    
    def close(self):
        """Explicitly close the browser"""
        if hasattr(self, 'driver'):
            try:
                if self.driver:
                    self.driver.quit()
            except:
                pass
        
    def __del__(self):
        """Clean up the browser when done"""
        try:
            self.close()
        except:
            pass
    
    def human_delay(self, min_sec: float = 0.3, max_sec: float = 1.0):
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
    
    def classify_sentiment(self, review_text: str, rating: Optional[float], title: str = "") -> Dict:
        """
        Classify review sentiment using VADER
        Returns dict with sentiment label, score, and confidence
        """
        # Combine text for analysis
        text_to_analyze = f"{title} {review_text}".strip()
        
        if not text_to_analyze:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "compound_score": 0.0,
                "method": "default"
            }
        
        try:
            # VADER analysis
            scores = self.vader_analyzer.polarity_scores(text_to_analyze)
            compound = scores['compound']
            positive = scores['pos']
            negative = scores['neg']
            neutral = scores['neu']
            
            # VADER compound score ranges from -1 (most negative) to +1 (most positive)
            # Thresholds: >= 0.05 is positive, <= -0.05 is negative, else neutral
            if compound >= 0.05:
                sentiment = "positive"
                confidence = min((compound + 1) / 2, 1.0)  # Normalize to 0-1
            elif compound <= -0.05:
                sentiment = "negative"
                confidence = min((1 - compound) / 2, 1.0)  # Normalize to 0-1
            else:
                sentiment = "neutral"
                confidence = 1 - abs(compound) * 2
            
            # Boost confidence if rating aligns with sentiment
            if rating is not None:
                if (sentiment == "positive" and rating >= 4) or \
                   (sentiment == "negative" and rating <= 2) or \
                   (sentiment == "neutral" and rating == 3):
                    confidence = min(confidence * 1.2, 1.0)
                    method = "vader"
                # If VADER is uncertain (|compound| < 0.3), use rating to decide
                elif abs(compound) < 0.3:
                    if rating >= 4:
                        sentiment = "positive"
                        confidence = 0.75
                    elif rating <= 2:
                        sentiment = "negative"
                        confidence = 0.75
                    else:
                        sentiment = "neutral"
                        confidence = 0.65
                    method = "vader_rating_adjusted"
                else:
                    method = "vader"
            else:
                method = "vader"
            
            return {
                "sentiment": sentiment,
                "confidence": round(confidence, 4),
                "compound_score": round(compound, 4),
                "positive_score": round(positive, 4),
                "negative_score": round(negative, 4),
                "neutral_score": round(neutral, 4),
                "method": method
            }
            
        except Exception as e:
            print(f"VADER error: {e}")
            # Fallback to rating-based
            if rating is not None:
                if rating >= 4:
                    return {"sentiment": "positive", "confidence": 0.8, "compound_score": 0.6, "method": "rating_fallback"}
                elif rating <= 2:
                    return {"sentiment": "negative", "confidence": 0.8, "compound_score": -0.6, "method": "rating_fallback"}
                else:
                    return {"sentiment": "neutral", "confidence": 0.7, "compound_score": 0.0, "method": "rating_fallback"}
            else:
                return {"sentiment": "neutral", "confidence": 0.5, "compound_score": 0.0, "method": "default"}
    
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
                            error_text = element.text[:100]  # First 100 chars
                            return error_text
                except:
                    continue
            
            return None
        except:
            return None
    
    def wait_for_captcha(self) -> bool:
        """Wait for user to complete CAPTCHA if it appears"""
        try:
            captcha_indicators = [
                "//iframe[contains(@src, 'captcha')]",
                "//*[contains(text(), 'Press & Hold')]",
                "//*[contains(text(), 'press and hold')]",
                "//*[contains(@id, 'px-captcha')]",
                "//*[contains(@class, 'captcha')]"
            ]
            
            for indicator in captcha_indicators:
                try:
                    element = self.driver.find_element(By.XPATH, indicator)
                    if element.is_displayed():
                        print("\nBOT DETECTION FOUND!")
                        print("Please complete the CAPTCHA/Press & Hold challenge manually...")
                        print("You have 60 seconds...")
                        
                        for i in range(60, 0, -5):
                            try:
                                if not element.is_displayed():
                                    print("Challenge completed! Continuing...")
                                    self.human_delay(0.5, 1.5)
                                    return True
                            except:
                                print("Challenge completed! Continuing...")
                                self.human_delay(0.5, 1.5)
                                return True
                            
                            print(f"Waiting... {i} seconds remaining")
                            time.sleep(5)
                        
                        print("Timeout waiting for challenge completion")
                        return False
                except:
                    continue
            
            return True
        except:
            return True
    
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
    
    def click_next_page(self, current_page: int) -> bool:
        """Optimized page navigation"""
        next_page_num = current_page + 1
        
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        self.human_delay(0.3, 0.7)
        
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
                    if self.click_element_safely(button):
                        self.wait.until(
                            lambda d: d.execute_script("return document.readyState") == "complete"
                        )
                        self.human_delay(0.5, 1.2)
                        return True
            except:
                continue
        
        print(f"No more pages found")
        return False
    
    def extract_reviews_from_current_page(self, seen_texts: set) -> List[Dict]:
        """Extract reviews from the current page with parallel processing"""
        reviews = []
        
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        self.human_delay(0.5, 1.0)
        
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
            return reviews
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.extract_review_from_element, elem) 
                      for elem in review_elements]
            results = [f.result() for f in futures]
        
        for review_data in results:
            if review_data and review_data.get('review_text'):
                review_text = review_data['review_text']
                if review_text not in seen_texts and len(review_text) > 10:
                    # Apply VADER sentiment analysis
                    sentiment_result = self.classify_sentiment(
                        review_data['review_text'],
                        review_data.get('rating'),
                        review_data.get('title', '')
                    )
                    
                    review_data.update(sentiment_result)
                    
                    reviews.append(review_data)
                    seen_texts.add(review_text)
        
        return reviews
    
    def scrape_reviews(self, url: str, product_name: str = None) -> Dict:
        """Scrape all reviews from a Walmart product page"""
        product_id = self.extract_product_id(url)
        
        if not product_id:
            raise ValueError("Invalid Walmart URL. Could not extract product ID.")
        
        print(f"\nProduct ID: {product_id}")
        if product_name:
            print(f"Product Name: {product_name}")
        
        print("\nLoading product page...")
        self.driver.get(url)
        self.human_delay(2.0, 3.5)
        
        # Check for page errors first
        error_message = self.check_for_errors()
        if error_message:
            print(f"⚠ Page Error Detected: {error_message}")
            print("Skipping this product...")
            return {
                "product_id": product_id, 
                "product_url": url, 
                "reviews": [], 
                "error": f"Page error: {error_message}",
                "skipped": True
            }
        
        if not self.wait_for_captcha():
            print("Failed to complete CAPTCHA. Exiting...")
            return {"product_id": product_id, "product_url": url, "reviews": [], "error": "CAPTCHA failed"}
        
        self.human_delay(1.0, 2.0)
        self.simulate_human_movement()
        
        print("Scrolling page to load content...")
        for i in range(2):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")
            self.human_delay(0.5, 1.0)
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            self.human_delay(0.5, 1.0)
        
        # Check for errors again after scrolling
        error_message = self.check_for_errors()
        if error_message:
            print(f"⚠ Page Error Detected After Scroll: {error_message}")
            print("Skipping this product...")
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
            print("Waiting for all reviews page to load...")
            
            # Check for errors after clicking
            error_message = self.check_for_errors()
            if error_message:
                print(f"⚠ Page Error After Loading Reviews: {error_message}")
                print("Skipping this product...")
                return {
                    "product_id": product_id, 
                    "product_url": url, 
                    "reviews": [], 
                    "error": f"Page error: {error_message}",
                    "skipped": True
                }
        
        print("Scrolling to pagination area...")
        for _ in range(2):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            self.human_delay(0.5, 1.0)
        
        print("\nCollecting reviews from all pages...")
        all_reviews = []
        seen_texts = set()
        current_page = 1
        max_pages = 50
        
        start_time = time.time()
        
        while True:
            page_start = time.time()
            print(f"\nPage {current_page}: Extracting reviews...")
            
            page_reviews = self.extract_reviews_from_current_page(seen_texts)
            all_reviews.extend(page_reviews)
            
            page_time = time.time() - page_start
            print(f"Got {len(page_reviews)} reviews from page {current_page} (Total: {len(all_reviews)}) [{page_time:.1f}s]")
            
            if not self.click_next_page(current_page):
                print(f"Finished at page {current_page}")
                break
            
            current_page += 1
            self.human_delay(0.8, 1.5)
        
        elapsed = time.time() - start_time
        print(f"\nSuccessfully extracted {len(all_reviews)} reviews in {elapsed:.1f}s")
        if all_reviews:
            print(f"Average: {elapsed/len(all_reviews):.2f}s per review\n")
        
        return {
            "product_id": product_id,
            "product_url": url,
            "product_name": product_name,
            "reviews": all_reviews,
            "scraped_at": datetime.now().isoformat()
        }
    
    def extract_review_from_element(self, element) -> Optional[Dict]:
        """Extract review data from a Selenium WebElement"""
        try:
            review_data = {}
            element_text = element.text
            
            # Extract reviewer name
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
            
            # Extract rating
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
            
            # Extract title
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
            
            # Extract review text
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
            
            # Extract date
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
            
            # Check verified purchase
            verified_text = element_text.lower()
            review_data['verified_purchase'] = 'verified' in verified_text
            
            # Extract helpful count
            helpful_match = re.search(r'(\d+)\s*helpful', element_text, re.IGNORECASE)
            if helpful_match:
                review_data['helpful_count'] = int(helpful_match.group(1))
            else:
                review_data['helpful_count'] = 0
            
            return review_data if review_data.get('review_text') and len(review_data['review_text']) > 10 else None
            
        except Exception as e:
            return None
    
    def save_to_json(self, reviews: List[Dict], filename: str = None):
        """Save reviews to JSON file with sentiment grouping"""
        if filename is None:
            filename = f"walmart_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        positive_reviews = [r for r in reviews if r.get('sentiment') == 'positive']
        negative_reviews = [r for r in reviews if r.get('sentiment') == 'negative']
        neutral_reviews = [r for r in reviews if r.get('sentiment') == 'neutral']
        
        # Calculate average confidence and compound scores
        all_confidences = [r.get('confidence', 0) for r in reviews if r.get('confidence')]
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        
        all_compounds = [r.get('compound_score', 0) for r in reviews if r.get('compound_score') is not None]
        avg_compound = sum(all_compounds) / len(all_compounds) if all_compounds else 0
        
        output = {
            "metadata": {
                "total_reviews": len(reviews),
                "positive_count": len(positive_reviews),
                "negative_count": len(negative_reviews),
                "neutral_count": len(neutral_reviews),
                "average_confidence": round(avg_confidence, 4),
                "average_compound_score": round(avg_compound, 4),
                "sentiment_analyzer": "VADER",
                "scraped_at": datetime.now().isoformat()
            },
            "reviews": {
                "all": reviews,
                "positive": positive_reviews,
                "negative": negative_reviews,
                "neutral": neutral_reviews
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"Reviews saved to {filename}")
        print(f"Positive: {len(positive_reviews)} | Negative: {len(negative_reviews)} | Neutral: {len(neutral_reviews)}")
        print(f"Average Confidence: {avg_confidence:.2%} | Avg Compound: {avg_compound:+.3f}")
    
    def save_to_csv(self, reviews: List[Dict], filename: str = None):
        """Save reviews to CSV file with sentiment"""
        if filename is None:
            filename = f"walmart_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        if not reviews:
            print("No reviews to save")
            return
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['product_id', 'product_name', 'reviewer_name', 'rating', 'sentiment', 
                         'confidence', 'compound_score', 'positive_score', 'negative_score', 
                         'neutral_score', 'method', 'title', 'review_text', 
                         'date', 'verified_purchase', 'helpful_count']
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            
            writer.writeheader()
            for review in reviews:
                writer.writerow(review)
        
        print(f"Reviews saved to {filename}")


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
    """
    Remove completed URL from original file and add it to completed file.
    This allows resuming if the script crashes.
    """
    try:
        # Read all remaining URLs
        if not os.path.exists(txt_file):
            return
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Filter out the completed URL
        remaining_lines = []
        for line in lines:
            if line.strip() and completed_url not in line:
                remaining_lines.append(line)
        
        # Overwrite original file with remaining URLs
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.writelines(remaining_lines)
        
        # Append to completed file
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
    """Example usage with multiple URLs from file"""
    
    print("\n" + "="*60)
    print("WALMART REVIEW SCRAPER - VADER SENTIMENT ANALYSIS")
    print("="*60)
    print("\nUsing VADER: Valence Aware Dictionary and sEntiment Reasoner")
    print("✓ Lightweight, fast, CPU-optimized")
    print("✓ Specifically designed for social media and review text")
    print("✓ Handles slang, emojis, and intensity modifiers")
    print("✓ Auto-resume: Completed URLs are moved to *_completed.txt\n")
    
    scraper = None
    input_file = None  # Track the input file for URL management
    
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
                input_file = txt_file  # Save for URL management
                
                if not urls:
                    print(f"\n{txt_file} is empty or all URLs have been processed!")
                    completed_file = txt_file.replace('.txt', '_completed.txt')
                    if os.path.exists(completed_file):
                        print(f"Check {completed_file} for completed URLs.")
                    return
                
                print(f"\nFound {len(urls)} URLs remaining in {txt_file}")
                
                # Check if completed file exists
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
        
        scraper = WalmartReviewScraper(headless=False)
        
        all_products_data = []
        all_reviews_combined = []
        
        for idx, url in enumerate(urls, 1):
            print("\n" + "="*60)
            print(f"SCRAPING PRODUCT {idx}/{len(urls)}")
            print("="*60)
            
            try:
                result = scraper.scrape_reviews(url)
                
                # Check if product was skipped due to error
                if result.get('skipped'):
                    print(f"\n⚠ Product skipped due to page error")
                    # Mark as completed with error status
                    if input_file:
                        mark_url_as_completed(input_file, url, f"SKIPPED_{result.get('product_id', 'UNKNOWN')}")
                    continue  # Skip to next URL
                
                if result and result.get('reviews'):
                    reviews = result['reviews']
                    
                    unique_reviews = []
                    seen = set()
                    
                    for review in reviews:
                        key = review.get('review_text', '')
                        if key and key not in seen and len(key) > 10:
                            seen.add(key)
                            review['product_id'] = result['product_id']
                            review['product_url'] = result['product_url']
                            review['product_name'] = result.get('product_name', f"Product {result['product_id']}")
                            unique_reviews.append(review)
                    
                    if len(reviews) != len(unique_reviews):
                        print(f"Removed {len(reviews) - len(unique_reviews)} duplicates")
                    
                    result['reviews'] = unique_reviews
                    all_products_data.append(result)
                    all_reviews_combined.extend(unique_reviews)
                    
                    product_id = result['product_id']
                    base_filename = f"walmart_reviews_{product_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    with open(f"{base_filename}.json", 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    print(f"\nSaved product data to {base_filename}.json")
                    
                    print(f"\nProduct {idx} Summary:")
                    print(f"  Total Reviews: {len(unique_reviews)}")
                    positive = sum(1 for r in unique_reviews if r.get('sentiment') == 'positive')
                    negative = sum(1 for r in unique_reviews if r.get('sentiment') == 'negative')
                    neutral = sum(1 for r in unique_reviews if r.get('sentiment') == 'neutral')
                    print(f"  Positive: {positive} | Negative: {negative} | Neutral: {neutral}")
                    
                    # Show VADER stats
                    confidences = [r.get('confidence', 0) for r in unique_reviews]
                    compounds = [r.get('compound_score', 0) for r in unique_reviews if r.get('compound_score') is not None]
                    avg_conf = sum(confidences) / len(confidences) if confidences else 0
                    avg_comp = sum(compounds) / len(compounds) if compounds else 0
                    print(f"  Avg Confidence: {avg_conf:.2%} | Avg Compound: {avg_comp:+.3f}")
                    
                    # Mark URL as completed (remove from original file, add to completed file)
                    if input_file:
                        mark_url_as_completed(input_file, url, product_id)
                    
                else:
                    print(f"\nNo reviews found for product {idx}")
                    # Still mark as completed even if no reviews found
                    if input_file:
                        result_id = result.get('product_id', 'UNKNOWN') if result else 'UNKNOWN'
                        mark_url_as_completed(input_file, url, f"NO_REVIEWS_{result_id}")
                    
            except Exception as e:
                print(f"\nError scraping product {idx}: {e}")
                import traceback
                traceback.print_exc()
                
                # Mark as completed even on error to avoid getting stuck
                if input_file:
                    print("\nMarking URL as completed despite error (to avoid infinite retry)...")
                    mark_url_as_completed(input_file, url, "ERROR")
            
            if idx < len(urls):
                delay = random.uniform(3, 6)
                print(f"\nWaiting {delay:.1f}s before next product...")
                time.sleep(delay)
        
        if all_reviews_combined:
            print("\n" + "="*60)
            print("SAVING COMBINED RESULTS")
            print("="*60)
            
            # Create unique filename based on input file name
            if input_file:
                # Use the input filename (without extension) in the output
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                combined_filename = f"walmart_reviews_{base_name}_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                combined_filename = f"walmart_reviews_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Calculate overall sentiment statistics
            positive = sum(1 for r in all_reviews_combined if r.get('sentiment') == 'positive')
            negative = sum(1 for r in all_reviews_combined if r.get('sentiment') == 'negative')
            neutral = sum(1 for r in all_reviews_combined if r.get('sentiment') == 'neutral')
            
            all_confidences = [r.get('confidence', 0) for r in all_reviews_combined if r.get('confidence')]
            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
            
            all_compounds = [r.get('compound_score', 0) for r in all_reviews_combined if r.get('compound_score') is not None]
            avg_compound = sum(all_compounds) / len(all_compounds) if all_compounds else 0
            
            combined_data = {
                "metadata": {
                    "source_file": input_file if input_file else "manual_input",
                    "total_products": len(all_products_data),
                    "total_reviews": len(all_reviews_combined),
                    "positive_count": positive,
                    "negative_count": negative,
                    "neutral_count": neutral,
                    "average_confidence": round(avg_confidence, 4),
                    "average_compound_score": round(avg_compound, 4),
                    "sentiment_analyzer": "VADER",
                    "scraped_at": datetime.now().isoformat()
                },
                "products": all_products_data
            }
            
            with open(f"{combined_filename}.json", 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, indent=2, ensure_ascii=False)
            print(f"\nSaved combined data to {combined_filename}.json")
            
            # Save combined CSV
            with open(f"{combined_filename}.csv", 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['product_id', 'product_name', 'product_url', 'reviewer_name', 'rating', 
                             'sentiment', 'confidence', 'compound_score', 'positive_score', 
                             'negative_score', 'neutral_score', 'method', 'title', 
                             'review_text', 'date', 'verified_purchase', 'helpful_count']
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                for review in all_reviews_combined:
                    writer.writerow(review)
            print(f"Saved combined data to {combined_filename}.csv")
            
            # Display overall summary
            print("\n" + "="*60)
            print("OVERALL SUMMARY")
            print("="*60)
            print(f"Products Scraped: {len(all_products_data)}")
            print(f"Total Reviews: {len(all_reviews_combined)}")
            
            print(f"\nOverall Sentiment Distribution:")
            print(f"  Positive: {positive:4d} ({positive/len(all_reviews_combined)*100:5.1f}%)")
            print(f"  Negative: {negative:4d} ({negative/len(all_reviews_combined)*100:5.1f}%)")
            print(f"  Neutral:  {neutral:4d} ({neutral/len(all_reviews_combined)*100:5.1f}%)")
            
            # VADER-specific statistics
            print(f"\nVADER Sentiment Scores:")
            print(f"  Average Confidence: {avg_confidence:.2%}")
            print(f"  Average Compound Score: {avg_compound:+.4f} (range: -1 to +1)")
            
            if all_compounds:
                max_positive = max(all_compounds)
                max_negative = min(all_compounds)
                print(f"  Most Positive Score: {max_positive:+.4f}")
                print(f"  Most Negative Score: {max_negative:+.4f}")
            
            # Show method distribution
            methods = {}
            for r in all_reviews_combined:
                method = r.get('method', 'unknown')
                methods[method] = methods.get(method, 0) + 1
            
            print(f"\nSentiment Analysis Methods:")
            for method, count in sorted(methods.items(), key=lambda x: x[1], reverse=True):
                print(f"  {method}: {count} ({count/len(all_reviews_combined)*100:.1f}%)")
            
            ratings = [r['rating'] for r in all_reviews_combined if r.get('rating')]
            if ratings:
                avg_rating = sum(ratings) / len(ratings)
                print(f"\nOverall Average Rating: {avg_rating:.2f} / 5.00")
                
                verified = sum(1 for r in all_reviews_combined if r.get('verified_purchase'))
                print(f"Total Verified Purchases: {verified} ({verified/len(all_reviews_combined)*100:.1f}%)")
                
                print(f"\nOverall Rating Distribution:")
                for star in range(5, 0, -1):
                    count = sum(1 for r in ratings if r == star)
                    percentage = (count / len(ratings) * 100) if ratings else 0
                    bar = "█" * int(percentage / 2)
                    print(f"  {star} stars: {count:4d} ({percentage:5.1f}%) {bar}")
            
            # Sentiment vs Rating correlation
            if ratings:
                print(f"\nSentiment vs Rating Analysis:")
                pos_ratings = [r['rating'] for r in all_reviews_combined if r.get('sentiment') == 'positive' and r.get('rating')]
                neg_ratings = [r['rating'] for r in all_reviews_combined if r.get('sentiment') == 'negative' and r.get('rating')]
                neu_ratings = [r['rating'] for r in all_reviews_combined if r.get('sentiment') == 'neutral' and r.get('rating')]
                
                if pos_ratings:
                    print(f"  Positive sentiment avg rating: {sum(pos_ratings)/len(pos_ratings):.2f}")
                if neg_ratings:
                    print(f"  Negative sentiment avg rating: {sum(neg_ratings)/len(neg_ratings):.2f}")
                if neu_ratings:
                    print(f"  Neutral sentiment avg rating: {sum(neu_ratings)/len(neu_ratings):.2f}")
            
            # Per-product summary
            print("\n" + "="*60)
            print("PER-PRODUCT SUMMARY")
            print("="*60)
            for i, product in enumerate(all_products_data, 1):
                reviews = product['reviews']
                product_id = product['product_id']
                print(f"\nProduct {i} (ID: {product_id}):")
                print(f"  URL: {product['product_url'][:60]}...")
                print(f"  Reviews: {len(reviews)}")
                
                if reviews:
                    pos = sum(1 for r in reviews if r.get('sentiment') == 'positive')
                    neg = sum(1 for r in reviews if r.get('sentiment') == 'negative')
                    neu = sum(1 for r in reviews if r.get('sentiment') == 'neutral')
                    print(f"  Sentiment: +{pos} / -{neg} / ={neu}")
                    
                    prod_ratings = [r['rating'] for r in reviews if r.get('rating')]
                    if prod_ratings:
                        prod_avg = sum(prod_ratings) / len(prod_ratings)
                        print(f"  Avg Rating: {prod_avg:.2f}/5.00")
                    
                    prod_confidences = [r.get('confidence', 0) for r in reviews if r.get('confidence')]
                    prod_compounds = [r.get('compound_score', 0) for r in reviews if r.get('compound_score') is not None]
                    
                    if prod_confidences:
                        prod_avg_conf = sum(prod_confidences) / len(prod_confidences)
                        print(f"  Avg Confidence: {prod_avg_conf:.2%}")
                    
                    if prod_compounds:
                        prod_avg_comp = sum(prod_compounds) / len(prod_compounds)
                        print(f"  Avg Compound: {prod_avg_comp:+.3f}")
            
            print("\n" + "="*60)
            print("SCRAPING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"\nVADER analyzed {len(all_reviews_combined)} reviews")
            print(f"Overall sentiment: {avg_compound:+.4f} ", end="")
            if avg_compound >= 0.05:
                print("(Positive leaning)")
            elif avg_compound <= -0.05:
                print("(Negative leaning)")
            else:
                print("(Neutral/Mixed)")
            
            # Show URL tracking info
            if input_file:
                completed_file = input_file.replace('.txt', '_completed.txt')
                print(f"\n✓ All URLs processed!")
                print(f"  Original file: {input_file} (now empty or has remaining URLs)")
                print(f"  Completed URLs: {completed_file}")
                
        else:
            print("\nNo reviews were collected from any products.")
            if input_file:
                print(f"\nNote: URLs have been moved to {input_file.replace('.txt', '_completed.txt')}")
                
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("SCRAPING INTERRUPTED BY USER")
        print("="*60)
        if input_file:
            completed_file = input_file.replace('.txt', '_completed.txt')
            print(f"\n✓ Progress saved!")
            print(f"  Remaining URLs: {input_file}")
            print(f"  Completed URLs: {completed_file}")
            print(f"\nYou can resume by running the script again with the same file.")
            print(f"Only unprocessed URLs will be scraped.")
        print()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        
        if input_file:
            print(f"\n✓ Progress saved before crash!")
            print(f"  Remaining URLs: {input_file}")
            print(f"  Completed URLs: {input_file.replace('.txt', '_completed.txt')}")
            print(f"\nRestart the script to continue from where it stopped.")
    
    finally:
        if scraper:
            print("\nClosing browser...")
            scraper.close()
            time.sleep(0.5)


if __name__ == "__main__":
    main()