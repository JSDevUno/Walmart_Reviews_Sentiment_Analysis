import time
import json
import csv
import re
from datetime import datetime
from typing import List, Dict, Optional
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

class WalmartReviewScraper:
    def __init__(self, headless: bool = False):
        """Initialize the Walmart Review Scraper using undetected-chromedriver"""
        print("🚀 Starting Chrome browser with anti-detection...")
        
        options = uc.ChromeOptions()
        if headless:
            options.add_argument('--headless=new')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--window-size=1920,1080')
        
        self.driver = uc.Chrome(options=options, version_main=None)
        self.wait = WebDriverWait(self.driver, 20)
        
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
        
        print("✅ Browser started successfully!")
        
    def __del__(self):
        """Clean up the browser when done"""
        if hasattr(self, 'driver'):
            try:
                self.driver.quit()
            except:
                pass
    
    def classify_sentiment(self, review_text: str, rating: Optional[float], title: str = "") -> str:
        """
        Classify review sentiment as 'positive', 'negative', or 'neutral'
        Uses both rating and text analysis
        """
        text_to_analyze = (review_text + " " + title).lower()
        
        # Primary classification: Use rating if available
        if rating is not None:
            if rating >= 4:
                sentiment = "positive"
            elif rating <= 2:
                sentiment = "negative"
            else:
                sentiment = "neutral"
        else:
            # No rating - classify by text only
            sentiment = "neutral"
        
        # Secondary classification: Analyze text for keywords
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_to_analyze)
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in text_to_analyze)
        
        # Adjust sentiment based on keyword analysis
        if rating is None or rating == 3:
            # For neutral ratings or missing ratings, rely more on keywords
            if negative_count > positive_count and negative_count >= 2:
                sentiment = "negative"
            elif positive_count > negative_count and positive_count >= 2:
                sentiment = "positive"
            elif positive_count == 0 and negative_count == 0:
                sentiment = "neutral"
        
        # Override: Strong negative keywords can override positive rating
        if rating and rating >= 4 and negative_count >= 3 and negative_count > positive_count:
            sentiment = "negative"
        
        # Override: Strong positive keywords can override negative rating
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
    
    def wait_for_captcha(self):
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
                        print("\n" + "="*70)
                        print("🤖 BOT DETECTION FOUND!")
                        print("="*70)
                        print("⏳ Please complete the CAPTCHA/Press & Hold challenge manually...")
                        print("   You have 60 seconds...")
                        print("="*70 + "\n")
                        
                        for i in range(60, 0, -5):
                            try:
                                if not element.is_displayed():
                                    print("✅ Challenge completed! Continuing...")
                                    time.sleep(2)
                                    return True
                            except:
                                print("✅ Challenge completed! Continuing...")
                                time.sleep(2)
                                return True
                            
                            print(f"   ⏱️  Waiting... {i} seconds remaining")
                            time.sleep(5)
                        
                        print("⚠️  Timeout waiting for challenge completion")
                        return False
                except:
                    continue
            
            return True
        except:
            return True
    
    def navigate_to_page(self, page_num: int) -> bool:
        """Navigate to a specific page number"""
        try:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            
            page_button_selectors = [
                f"//button[@aria-label='page {page_num}']",
                f"//a[@aria-label='page {page_num}']",
                f"//button[text()='{page_num}' and not(@disabled)]",
                f"//a[text()='{page_num}']",
                f"//nav//button[.='{page_num}']"
            ]
            
            for selector in page_button_selectors:
                try:
                    page_button = self.driver.find_element(By.XPATH, selector)
                    if page_button.is_displayed():
                        self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", page_button)
                        time.sleep(1)
                        self.driver.execute_script("arguments[0].click();", page_button)
                        time.sleep(3)
                        return True
                except:
                    continue
            
            return False
        except:
            return False
    
    def click_next_page(self, current_page: int) -> bool:
        """Click next page button"""
        try:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            next_page_num = current_page + 1
            
            # Method 1: Click the next page number directly
            print(f"      🔍 Looking for page {next_page_num} button...")
            page_number_selectors = [
                f"//button[normalize-space(text())='{next_page_num}']",
                f"//a[normalize-space(text())='{next_page_num}']",
                f"//button[@aria-label='page {next_page_num}']",
                f"//a[@aria-label='page {next_page_num}']",
                f"//button[@aria-label='Go to page {next_page_num}']",
                f"//*[@role='button'][normalize-space(text())='{next_page_num}']",
                f"//nav//*[normalize-space(text())='{next_page_num}']"
            ]
            
            for selector in page_number_selectors:
                try:
                    page_button = self.driver.find_element(By.XPATH, selector)
                    if page_button.is_displayed():
                        print(f"      ✅ Found page {next_page_num} button!")
                        self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", page_button)
                        time.sleep(1)
                        # Highlight for debugging
                        self.driver.execute_script("arguments[0].style.border='3px solid red'", page_button)
                        time.sleep(0.5)
                        self.driver.execute_script("arguments[0].click();", page_button)
                        time.sleep(4)
                        return True
                except Exception as e:
                    continue
            
            # Method 2: Click the "next" arrow/button
            print(f"      🔍 Looking for 'Next' button...")
            next_selectors = [
                "//button[contains(@aria-label, 'next page')]",
                "//a[contains(@aria-label, 'next page')]",
                "//button[contains(@aria-label, 'Go to next page')]",
                "//button[contains(., '›')]",
                "//a[contains(., '›')]",
                "//*[@aria-label='next page']",
                "//button[contains(@class, 'pagination')][contains(., 'Next')]"
            ]
            
            for selector in next_selectors:
                try:
                    next_button = self.driver.find_element(By.XPATH, selector)
                    if next_button.is_displayed() and next_button.is_enabled():
                        disabled = next_button.get_attribute('disabled')
                        aria_disabled = next_button.get_attribute('aria-disabled')
                        
                        if disabled or aria_disabled == 'true':
                            print(f"      ⚠️  Next button is disabled")
                            return False
                        
                        print(f"      ✅ Found 'Next' button!")
                        self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_button)
                        time.sleep(1)
                        self.driver.execute_script("arguments[0].click();", next_button)
                        time.sleep(4)
                        return True
                except Exception as e:
                    continue
            
            print(f"      ❌ Could not find page {next_page_num} or next button")
            
            # Debug: Print all buttons on page
            try:
                all_buttons = self.driver.find_elements(By.XPATH, "//button | //a[@role='button']")
                print(f"      🐛 DEBUG: Found {len(all_buttons)} total buttons/links")
                nav_elements = self.driver.find_elements(By.XPATH, "//nav//*")
                print(f"      🐛 DEBUG: Found {len(nav_elements)} elements in nav")
            except:
                pass
            
            return False
        except Exception as e:
            print(f"      ❌ Exception in click_next_page: {e}")
            return False
    
    def extract_reviews_from_current_page(self, seen_texts: set) -> List[Dict]:
        """Extract reviews from the current page only"""
        reviews = []
        
        # Scroll to ensure content is loaded
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        
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
        
        for element in review_elements:
            try:
                review_data = self.extract_review_from_element(element)
                if review_data and review_data.get('review_text'):
                    review_text = review_data['review_text']
                    if review_text not in seen_texts and len(review_text) > 10:
                        # Add sentiment classification
                        sentiment = self.classify_sentiment(
                            review_data['review_text'],
                            review_data.get('rating'),
                            review_data.get('title', '')
                        )
                        review_data['sentiment'] = sentiment
                        
                        reviews.append(review_data)
                        seen_texts.add(review_text)
            except:
                continue
        
        return reviews
    
    def scrape_reviews(self, url: str, max_reviews: int = 5000) -> List[Dict]:
        """Scrape all reviews from a Walmart product page"""
        product_id = self.extract_product_id(url)
        
        if not product_id:
            raise ValueError("Invalid Walmart URL. Could not extract product ID.")
        
        print(f"\n{'='*70}")
        print(f"🛒 Product ID: {product_id}")
        print(f"{'='*70}")
        
        # Load the page
        print("\n🌐 Loading product page...")
        self.driver.get(url)
        time.sleep(5)
        
        if not self.wait_for_captcha():
            print("❌ Failed to complete CAPTCHA. Exiting...")
            return []
        
        time.sleep(3)
        
        # Scroll to load content
        print("📜 Scrolling page to load content...")
        for _ in range(3):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
        
        # Look for Reviews tab
        try:
            print("🔍 Looking for Reviews tab...")
            review_tab_selectors = [
                "//button[contains(text(), 'Reviews')]",
                "//a[contains(text(), 'Reviews')]",
                "//*[@role='tab'][contains(text(), 'Reviews')]"
            ]
            
            for selector in review_tab_selectors:
                try:
                    tab = self.driver.find_element(By.XPATH, selector)
                    if tab.is_displayed():
                        print("   ✅ Found Reviews tab, clicking...")
                        self.driver.execute_script("arguments[0].click();", tab)
                        time.sleep(3)
                        break
                except:
                    continue
        except:
            pass
        
        # Click "View all reviews" button
        try:
            print("🔘 Looking for 'View all reviews' button...")
            see_all_selectors = [
                "//button[contains(text(), 'View all reviews')]",
                "//a[contains(text(), 'View all reviews')]",
                "//button[contains(text(), 'See all reviews')]",
                "//a[contains(text(), 'See all reviews')]"
            ]
            
            for selector in see_all_selectors:
                try:
                    buttons = self.driver.find_elements(By.XPATH, selector)
                    for button in buttons:
                        if button.is_displayed():
                            print(f"   ✅ Found button: '{button.text}', clicking...")
                            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
                            time.sleep(1)
                            self.driver.execute_script("arguments[0].click();", button)
                            time.sleep(5)
                            print("   ⏳ Waiting for all reviews page to load...")
                            break
                    break
                except:
                    continue
        except:
            pass
        
        # Scroll to ensure pagination loads
        print("📜 Scrolling to pagination area...")
        for _ in range(5):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
        
        # Debug: Check if pagination exists
        try:
            pagination_elements = self.driver.find_elements(By.XPATH, "//nav | //*[contains(@class, 'pagination')]")
            print(f"   🐛 DEBUG: Found {len(pagination_elements)} pagination elements")
            
            page_buttons = self.driver.find_elements(By.XPATH, "//button[number(text())>0] | //a[number(text())>0]")
            print(f"   🐛 DEBUG: Found {len(page_buttons)} numeric page buttons")
            
            if page_buttons:
                for i, btn in enumerate(page_buttons[:5]):
                    print(f"      Button {i+1}: '{btn.text}' (visible: {btn.is_displayed()})")
            
            # Save page source for debugging
            with open('walmart_pagination_debug.html', 'w', encoding='utf-8') as f:
                f.write(self.driver.page_source)
            print(f"   💾 Saved page source to 'walmart_pagination_debug.html' for inspection")
        except Exception as e:
            print(f"   ⚠️  Debug error: {e}")
        
        # Now collect reviews from all pages
        print("\n📄 Collecting reviews from all pages...")
        all_reviews = []
        seen_texts = set()
        current_page = 1
        max_pages = 50
        
        for page_num in range(1, max_pages + 1):
            print(f"\n   📄 Page {page_num}: Extracting reviews...")
            
            # Extract reviews from current page
            page_reviews = self.extract_reviews_from_current_page(seen_texts)
            all_reviews.extend(page_reviews)
            
            print(f"   ✅ Got {len(page_reviews)} reviews from page {page_num} (Total: {len(all_reviews)})")
            
            if len(all_reviews) >= max_reviews:
                print(f"   ⚠️  Reached maximum review limit ({max_reviews})")
                break
            
            # Try to go to next page
            if not self.click_next_page(page_num):
                print(f"   ✅ No more pages found. Finished at page {page_num}")
                break
            
            time.sleep(2)
        
        print(f"\n{'='*70}")
        print(f"✅ Successfully extracted {len(all_reviews)} reviews")
        print(f"{'='*70}\n")
        
        return all_reviews
    
    def extract_review_from_element(self, element) -> Optional[Dict]:
        """Extract review data from a Selenium WebElement"""
        try:
            review_data = {}
            element_text = element.text
            
            # Get reviewer name
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
                    if name and len(name) > 0 and len(name) < 50:
                        reviewer_name = name
                        break
                except:
                    continue
            review_data['reviewer_name'] = reviewer_name
            
            # Get rating
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
            
            # Get title
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
                    if title and len(title) > 0 and len(title) < 200:
                        break
                except:
                    continue
            review_data['title'] = title
            
            # Get review text
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
            
            # Get date
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
            
            # Check if verified purchase
            verified_text = element_text.lower()
            review_data['verified_purchase'] = 'verified' in verified_text
            
            # Get helpful count
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
        
        # Group reviews by sentiment
        positive_reviews = [r for r in reviews if r.get('sentiment') == 'positive']
        negative_reviews = [r for r in reviews if r.get('sentiment') == 'negative']
        neutral_reviews = [r for r in reviews if r.get('sentiment') == 'neutral']
        
        # Create structured output
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
        
        print(f"💾 Reviews saved to {filename}")
        print(f"   📊 Positive: {len(positive_reviews)} | Negative: {len(negative_reviews)} | Neutral: {len(neutral_reviews)}")
    
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
        
        print(f"💾 Reviews saved to {filename}")
    
    def save_sentiment_separate_files(self, reviews: List[Dict], base_filename: str = None):
        """Save reviews in separate JSON files by sentiment"""
        if base_filename is None:
            base_filename = f"walmart_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Group by sentiment
        positive = [r for r in reviews if r.get('sentiment') == 'positive']
        negative = [r for r in reviews if r.get('sentiment') == 'negative']
        neutral = [r for r in reviews if r.get('sentiment') == 'neutral']
        
        # Save each sentiment to separate file
        for sentiment_type, sentiment_reviews in [('positive', positive), ('negative', negative), ('neutral', neutral)]:
            if sentiment_reviews:
                filename = f"{base_filename}_{sentiment_type}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(sentiment_reviews, f, indent=2, ensure_ascii=False)
                print(f"   💾 {sentiment_type.capitalize()}: {len(sentiment_reviews)} reviews → {filename}")


def main():
    """Example usage"""
    
    print("\n" + "=" * 70)
    print(" " * 15 + "WALMART REVIEW SCRAPER")
    print(" " * 12 + "(Pagination Support)")
    print("=" * 70)
    
    scraper = None
    
    try:
        # Get product URL
        product_url = input("\n🔗 Enter Walmart product URL: ").strip()
        
        if not product_url:
            print("❌ No URL provided. Exiting...")
            return
        
        # Initialize scraper
        scraper = WalmartReviewScraper(headless=False)
        
        # Scrape reviews
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
                print(f"🔄 Removed {len(reviews) - len(unique_reviews)} duplicates\n")
                reviews = unique_reviews
            
            # Save files
            scraper.save_to_json(reviews)
            scraper.save_to_csv(reviews)
            
            # Also save separate files by sentiment for easy dataset creation
            print("\n📁 Saving separate files by sentiment...")
            scraper.save_sentiment_separate_files(reviews)
            
            # Print summary
            print("\n" + "=" * 70)
            print(" " * 25 + "📊 SUMMARY")
            print("=" * 70)
            
            print(f"\n   📝 Total Reviews: {len(reviews)}")
            
            # Sentiment distribution
            positive = sum(1 for r in reviews if r.get('sentiment') == 'positive')
            negative = sum(1 for r in reviews if r.get('sentiment') == 'negative')
            neutral = sum(1 for r in reviews if r.get('sentiment') == 'neutral')
            
            print(f"\n   😊 Sentiment Distribution:")
            print(f"   ✅ Positive: {positive} ({positive/len(reviews)*100:.1f}%)")
            print(f"   ❌ Negative: {negative} ({negative/len(reviews)*100:.1f}%)")
            print(f"   ⚪ Neutral:  {neutral} ({neutral/len(reviews)*100:.1f}%)")
            
            ratings = [r['rating'] for r in reviews if r.get('rating')]
            if ratings:
                avg_rating = sum(ratings) / len(ratings)
                print(f"\n   ⭐ Average Rating: {avg_rating:.2f} / 5")
                
                verified = sum(1 for r in reviews if r.get('verified_purchase'))
                print(f"   ✓ Verified Purchases: {verified}")
                
                print(f"\n   Rating Distribution:")
                for star in range(5, 0, -1):
                    count = sum(1 for r in ratings if r == star)
                    percentage = (count / len(ratings) * 100) if ratings else 0
                    bar = "█" * int(percentage / 2)
                    print(f"   {star} ⭐: {count:3d} ({percentage:5.1f}%) {bar}")
            
            # Show sample reviews from each sentiment
            print(f"\n" + "=" * 70)
            print(" " * 20 + "📝 SAMPLE REVIEWS BY SENTIMENT")
            print("=" * 70)
            
            for sentiment_type in ['positive', 'negative', 'neutral']:
                sentiment_reviews = [r for r in reviews if r.get('sentiment') == sentiment_type]
                if sentiment_reviews:
                    print(f"\n   {'='*66}")
                    emoji = "✅" if sentiment_type == 'positive' else "❌" if sentiment_type == 'negative' else "⚪"
                    print(f"   {emoji} {sentiment_type.upper()} REVIEWS ({len(sentiment_reviews)} total)")
                    print(f"   {'='*66}")
                    
                    for i, review in enumerate(sentiment_reviews[:2], 1):
                        print(f"\n   Review #{i}:")
                        print(f"   👤 {review.get('reviewer_name', 'Anonymous')}")
                        if review.get('rating'):
                            print(f"   ⭐ Rating: {review.get('rating')}/5")
                        if review.get('title'):
                            print(f"   📌 {review.get('title')}")
                        text = review.get('review_text', '')
                        if text:
                            display_text = text[:150] + "..." if len(text) > 150 else text
                            print(f"   💬 {display_text}")
                        if review.get('date'):
                            print(f"   📅 {review.get('date')}")
            
            print("\n" + "=" * 70)
            print("✅ Scraping completed successfully!")
            print("=" * 70)
        else:
            print("\n❌ No reviews found.")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if scraper:
            print("\n🧹 Closing browser...")
            time.sleep(2)
            del scraper


if __name__ == "__main__":
    main()