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


class WalmartReviewScraper:
    def __init__(self, headless: bool = False):
        """Initialize the Walmart Review Scraper with optimized settings"""
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
        options.page_load_strategy = 'eager'  # Don't wait for all resources
        
        # Reuse profile for session persistence
        profile_path = os.path.expanduser("~/.walmart_scraper_profile")
        options.add_argument(f'--user-data-dir={profile_path}')
        
        self.driver = uc.Chrome(options=options, version_main=None)
        self.wait = WebDriverWait(self.driver, 8)
        self.short_wait = WebDriverWait(self.driver, 3)
        
        # Sentiment analysis keywords
        self.positive_keywords = [
            'excellent', 'amazing', 'great', 'love', 'perfect', 'best', 'awesome',
            'wonderful', 'fantastic', 'good', 'nice', 'happy', 'pleased', 'satisfied',
            'recommend', 'quality', 'comfortable', 'beautiful', 'worth', 'glad'
        ]
        
        self.negative_keywords = [
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'poor', 'disappointing',
            'disappointed', 'waste', 'useless', 'broken', 'defective', 'cheap',
            'returning', 'return', 'refund', 'never', 'not worth', 'avoid', 'do not buy'
        ]
        
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
    
    def classify_sentiment(self, review_text: str, rating: Optional[float], title: str = "") -> str:
        """Classify review sentiment as 'positive', 'negative', or 'neutral'"""
        text_to_analyze = (review_text + " " + title).lower()
        
        if rating is not None:
            if rating >= 4:
                sentiment = "positive"
            elif rating <= 2:
                sentiment = "negative"
            else:
                sentiment = "neutral"
        else:
            sentiment = "neutral"
        
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_to_analyze)
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in text_to_analyze)
        
        if rating is None or rating == 3:
            if negative_count > positive_count and negative_count >= 2:
                sentiment = "negative"
            elif positive_count > negative_count and positive_count >= 2:
                sentiment = "positive"
            elif positive_count == 0 and negative_count == 0:
                sentiment = "neutral"
        
        if rating and rating >= 4 and negative_count >= 3 and negative_count > positive_count:
            sentiment = "negative"
        
        if rating and rating <= 2 and positive_count >= 3 and positive_count > negative_count:
            sentiment = "positive"
        
        return sentiment
    
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
        
        # Single scroll to pagination
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        self.human_delay(0.3, 0.7)
        
        print(f"Looking for page {next_page_num}...")
        
        # Combined selector list - try page number first, then next button
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
                    # Check if disabled
                    disabled = button.get_attribute('disabled')
                    aria_disabled = button.get_attribute('aria-disabled')
                    
                    if disabled or aria_disabled == 'true':
                        continue
                    
                    print(f"Found navigation button, clicking...")
                    if self.click_element_safely(button):
                        # Wait for new content
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
        
        # Quick scroll to load lazy content
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        self.human_delay(0.5, 1.0)
        
        # Find review elements
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
        
        # Parallel extraction for speed
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.extract_review_from_element, elem) 
                      for elem in review_elements]
            results = [f.result() for f in futures]
        
        # Filter and process results
        for review_data in results:
            if review_data and review_data.get('review_text'):
                review_text = review_data['review_text']
                if review_text not in seen_texts and len(review_text) > 10:
                    sentiment = self.classify_sentiment(
                        review_data['review_text'],
                        review_data.get('rating'),
                        review_data.get('title', '')
                    )
                    review_data['sentiment'] = sentiment
                    
                    reviews.append(review_data)
                    seen_texts.add(review_text)
        
        return reviews
    
    def scrape_reviews(self, url: str) -> List[Dict]:
        """Scrape all reviews from a Walmart product page"""
        product_id = self.extract_product_id(url)
        
        if not product_id:
            raise ValueError("Invalid Walmart URL. Could not extract product ID.")
        
        print(f"\nProduct ID: {product_id}")
        
        print("\nLoading product page...")
        self.driver.get(url)
        self.human_delay(2.0, 3.5)
        
        if not self.wait_for_captcha():
            print("Failed to complete CAPTCHA. Exiting...")
            return []
        
        self.human_delay(1.0, 2.0)
        
        # Simulate human behavior
        self.simulate_human_movement()
        
        print("Scrolling page to load content...")
        for i in range(2):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")
            self.human_delay(0.5, 1.0)
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            self.human_delay(0.5, 1.0)
        
        # Find and click Reviews tab
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
        
        # Find and click 'View all reviews' button
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
        
        # Scroll to pagination
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
            # Adaptive delay based on speed
            self.human_delay(0.8, 1.5)
        
        elapsed = time.time() - start_time
        print(f"\nSuccessfully extracted {len(all_reviews)} reviews in {elapsed:.1f}s")
        print(f"Average: {elapsed/len(all_reviews):.2f}s per review\n")
        
        return all_reviews
    
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
        
        output = {
            "metadata": {
                "total_reviews": len(reviews),
                "positive_count": len(positive_reviews),
                "negative_count": len(negative_reviews),
                "neutral_count": len(neutral_reviews),
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
    
    def save_to_csv(self, reviews: List[Dict], filename: str = None):
        """Save reviews to CSV file with sentiment"""
        if filename is None:
            filename = f"walmart_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        if not reviews:
            print("No reviews to save")
            return
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['reviewer_name', 'rating', 'sentiment', 'title', 'review_text', 'date', 'verified_purchase', 'helpful_count']
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            
            writer.writeheader()
            for review in reviews:
                writer.writerow(review)
        
        print(f"Reviews saved to {filename}")
    
    def save_sentiment_separate_files(self, reviews: List[Dict], base_filename: str = None):
        """Save reviews in separate JSON files by sentiment"""
        if base_filename is None:
            base_filename = f"walmart_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        positive = [r for r in reviews if r.get('sentiment') == 'positive']
        negative = [r for r in reviews if r.get('sentiment') == 'negative']
        neutral = [r for r in reviews if r.get('sentiment') == 'neutral']
        
        for sentiment_type, sentiment_reviews in [('positive', positive), ('negative', negative), ('neutral', neutral)]:
            if sentiment_reviews:
                filename = f"{base_filename}_{sentiment_type}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(sentiment_reviews, f, indent=2, ensure_ascii=False)
                print(f"{sentiment_type.capitalize()}: {len(sentiment_reviews)} reviews saved to {filename}")


def main():
    """Example usage"""
    
    print("\n" + "="*60)
    print("WALMART REVIEW SCRAPER - OPTIMIZED VERSION")
    print("="*60)
    
    scraper = None
    
    try:
        product_url = input("\nEnter Walmart product URL: ").strip()
        
        if not product_url:
            print("No URL provided. Exiting...")
            return
        
        scraper = WalmartReviewScraper(headless=False)
        
        reviews = scraper.scrape_reviews(product_url)
        
        if reviews:
            # Remove duplicates
            unique_reviews = []
            seen = set()
            
            for review in reviews:
                key = review.get('review_text', '')
                if key and key not in seen and len(key) > 10:
                    seen.add(key)
                    unique_reviews.append(review)
            
            if len(reviews) != len(unique_reviews):
                print(f"Removed {len(reviews) - len(unique_reviews)} duplicates\n")
                reviews = unique_reviews
            
            # Save files
            scraper.save_to_json(reviews)
            scraper.save_to_csv(reviews)
            
            print("\nSaving separate files by sentiment...")
            scraper.save_sentiment_separate_files(reviews)
            
            # Display summary
            print("\n" + "="*60)
            print("SUMMARY")
            print("="*60)
            print(f"Total Reviews: {len(reviews)}")
            
            positive = sum(1 for r in reviews if r.get('sentiment') == 'positive')
            negative = sum(1 for r in reviews if r.get('sentiment') == 'negative')
            neutral = sum(1 for r in reviews if r.get('sentiment') == 'neutral')
            
            print(f"\nSentiment Distribution:")
            print(f"  Positive: {positive:4d} ({positive/len(reviews)*100:5.1f}%)")
            print(f"  Negative: {negative:4d} ({negative/len(reviews)*100:5.1f}%)")
            print(f"  Neutral:  {neutral:4d} ({neutral/len(reviews)*100:5.1f}%)")
            
            ratings = [r['rating'] for r in reviews if r.get('rating')]
            if ratings:
                avg_rating = sum(ratings) / len(ratings)
                print(f"\nAverage Rating: {avg_rating:.2f} / 5.00")
                
                verified = sum(1 for r in reviews if r.get('verified_purchase'))
                print(f"Verified Purchases: {verified}")
                
                print(f"\nRating Distribution:")
                for star in range(5, 0, -1):
                    count = sum(1 for r in ratings if r == star)
                    percentage = (count / len(ratings) * 100) if ratings else 0
                    bar = "#" * int(percentage / 2)
                    print(f"  {star} stars: {count:3d} ({percentage:5.1f}%) {bar}")
            
            print("\n" + "="*60)
            print("Scraping completed successfully!")
            print("="*60)
        else:
            print("\nNo reviews found.")
            
    except KeyboardInterrupt:
        print("\n\nScraping interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if scraper:
            print("\nClosing browser...")
            scraper.close()
            time.sleep(0.5)


if __name__ == "__main__":
    main()