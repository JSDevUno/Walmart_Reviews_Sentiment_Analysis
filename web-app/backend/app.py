from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
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
from selenium.common.exceptions import WebDriverException, NoSuchWindowException
import re
import tempfile
import shutil
import threading

app = Flask(__name__)
CORS(app)

# Global variable to track analysis status
analysis_status = {}

class TFIDFSentimentAnalyzer:
    def __init__(self, model_path: str = None):
        """Initialize the TF-IDF + SVM sentiment analyzer"""
        if model_path is None:
            if os.path.exists("latest_model.txt"):
                with open("latest_model.txt", 'r') as f:
                    model_path = f.read().strip()
            else:
                pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl') and 'sentiment' in f.lower()]
                if pkl_files:
                    model_path = sorted(pkl_files)[-1]
                else:
                    raise FileNotFoundError("No trained model found!")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.model = model_data['model']
        self.label_map = model_data['label_map']
        self.reverse_label_map = model_data['reverse_label_map']
        
    def predict_sentiment(self, text: str, title: str = "") -> Dict:
        """Predict sentiment for a single review"""
        full_text = f"{title} {text}".strip()
        
        if not full_text:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "probabilities": {"negative": 0.33, "neutral": 0.34, "positive": 0.33}
            }
        
        text_tfidf = self.vectorizer.transform([full_text])
        prediction = self.model.predict(text_tfidf)[0]
        sentiment = self.reverse_label_map[prediction]
        
        decision_scores = self.model.decision_function(text_tfidf)[0]
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
            "probabilities": prob_dict
        }


class WalmartReviewInference:
    def __init__(self, sentiment_analyzer: TFIDFSentimentAnalyzer, headless: bool = False):
        """Initialize scraper with sentiment analyzer"""
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
        self.browser_closed = False
    
    def is_browser_alive(self) -> bool:
        """Check if browser is still running"""
        if self.browser_closed:
            return False
        try:
            _ = self.driver.current_url
            return True
        except (WebDriverException, NoSuchWindowException):
            self.browser_closed = True
            return False
    
    def close(self):
        """Close browser and cleanup"""
        self.browser_closed = True
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
        if not self.is_browser_alive():
            return False
        try:
            self.driver.execute_script(
                "arguments[0].scrollIntoView({block: 'center', behavior: 'instant'});", 
                element
            )
            self.human_delay(0.2, 0.5)
            self.driver.execute_script("arguments[0].click();", element)
            return True
        except (WebDriverException, NoSuchWindowException):
            self.browser_closed = True
            return False
    
    def check_captcha_quick(self) -> bool:
        """Quick check for CAPTCHA"""
        if not self.is_browser_alive():
            return False
            
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
    
    def click_next_page(self, current_page: int) -> Tuple[bool, bool]:
        """
        Navigate to next page.
        Returns (success: bool, captcha_detected: bool)
        """
        if not self.is_browser_alive():
            return False, False
            
        if self.check_captcha_quick():
            return False, True
        
        try:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            self.human_delay(0.3, 0.7)
        except (WebDriverException, NoSuchWindowException):
            self.browser_closed = True
            return False, False
        
        next_page_num = current_page + 1
        
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
            if not self.is_browser_alive():
                return False, False
                
            try:
                button = self.driver.find_element(By.XPATH, selector)
                if button.is_displayed() and button.is_enabled():
                    disabled = button.get_attribute('disabled')
                    aria_disabled = button.get_attribute('aria-disabled')
                    
                    if disabled or aria_disabled == 'true':
                        continue
                    
                    if self.check_captcha_quick():
                        return False, True
                    
                    if self.click_element_safely(button):
                        try:
                            self.wait.until(
                                lambda d: d.execute_script("return document.readyState") == "complete"
                            )
                            self.human_delay(0.5, 1.2)
                        except:
                            pass
                        
                        if self.check_captcha_quick():
                            return False, True
                        
                        return True, False
            except:
                continue
        
        return False, False
    
    def extract_review_from_element(self, element) -> Optional[Dict]:
        """Extract review data from element"""
        try:
            review_data = {}
            
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
            
            verified_text = element.text.lower()
            review_data['verified_purchase'] = 'verified' in verified_text
            
            return review_data if review_data.get('review_text') and len(review_data['review_text']) > 10 else None
            
        except Exception as e:
            return None
    
    def extract_reviews_from_current_page(self, seen_texts: set) -> Tuple[List[Dict], bool]:
        """Extract reviews from current page"""
        if not self.is_browser_alive():
            return [], False
            
        if self.check_captcha_quick():
            return [], True
        
        try:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            self.human_delay(0.5, 1.0)
        except (WebDriverException, NoSuchWindowException):
            self.browser_closed = True
            return [], False
        
        if self.check_captcha_quick():
            return [], True
        
        review_selectors = [
            "//*[contains(@data-testid, 'review')]",
            "//*[contains(@class, 'review-')]",
            "//div[contains(@class, 'customer-review')]"
        ]
        
        review_elements = []
        for selector in review_selectors:
            if not self.is_browser_alive():
                return [], False
            try:
                elements = self.driver.find_elements(By.XPATH, selector)
                filtered = [e for e in elements if len(e.text) > 50]
                if filtered and len(filtered) > len(review_elements):
                    review_elements = filtered
            except:
                continue
        
        reviews = []
        for elem in review_elements:
            review_data = self.extract_review_from_element(elem)
            if review_data and review_data.get('review_text'):
                review_text = review_data['review_text']
                if review_text not in seen_texts and len(review_text) > 10:
                    sentiment_result = self.analyzer.predict_sentiment(
                        review_data['review_text'],
                        review_data.get('title', '')
                    )
                    
                    review_data.update(sentiment_result)
                    reviews.append(review_data)
                    seen_texts.add(review_text)
        
        return reviews, False
    
    def scrape_and_analyze(self, url: str, max_reviews: int = 50, session_id: str = None) -> Dict:
        """Scrape reviews and analyze sentiment with pagination"""
        if session_id:
            analysis_status[session_id] = {"status": "loading", "message": "Extracting product ID..."}
        
        product_id = self.extract_product_id(url)
        
        if not product_id:
            raise ValueError("Invalid Walmart URL")
        
        if session_id:
            analysis_status[session_id] = {"status": "loading", "message": "Loading product page..."}
        
        try:
            self.driver.get(url)
            self.human_delay(2.0, 3.5)
        except (WebDriverException, NoSuchWindowException):
            raise ValueError("Browser was closed unexpectedly")
        
        if not self.is_browser_alive():
            raise ValueError("Browser was closed unexpectedly")
        
        if session_id:
            analysis_status[session_id] = {"status": "loading", "message": "Navigating to reviews..."}
        
        for _ in range(2):
            if not self.is_browser_alive():
                raise ValueError("Browser was closed unexpectedly")
            try:
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                self.human_delay(0.5, 1.0)
            except:
                break
        
        review_tab_selectors = [
            "//button[contains(text(), 'Reviews')]",
            "//a[contains(text(), 'Reviews')]"
        ]
        
        for selector in review_tab_selectors:
            if not self.is_browser_alive():
                raise ValueError("Browser was closed unexpectedly")
            try:
                tab = self.driver.find_element(By.XPATH, selector)
                if tab.is_displayed():
                    self.click_element_safely(tab)
                    self.human_delay(1.5, 2.5)
                    break
            except:
                continue
        
        see_all_selectors = [
            "//button[contains(text(), 'View all reviews')]",
            "//a[contains(text(), 'View all reviews')]"
        ]
        
        for selector in see_all_selectors:
            if not self.is_browser_alive():
                raise ValueError("Browser was closed unexpectedly")
            try:
                button = self.driver.find_element(By.XPATH, selector)
                if button.is_displayed():
                    self.click_element_safely(button)
                    self.human_delay(2.0, 3.5)
                    break
            except:
                continue
        
        if session_id:
            analysis_status[session_id] = {"status": "loading", "message": "Extracting reviews from multiple pages..."}
        
        all_reviews = []
        seen_texts = set()
        current_page = 1
        max_pages = 10
        
        while current_page <= max_pages and len(all_reviews) < max_reviews:
            if not self.is_browser_alive():
                raise ValueError("Browser was closed unexpectedly")
                
            if session_id:
                analysis_status[session_id] = {
                    "status": "loading", 
                    "message": f"Extracting reviews from page {current_page}... (Found: {len(all_reviews)})"
                }
            
            page_reviews, captcha_detected = self.extract_reviews_from_current_page(seen_texts)
            
            if captcha_detected:
                break
            
            all_reviews.extend(page_reviews)
            
            if len(all_reviews) >= max_reviews:
                break
            
            next_success, captcha_detected = self.click_next_page(current_page)
            
            if captcha_detected or not next_success:
                break
            
            current_page += 1
            self.human_delay(0.8, 1.5)
        
        reviews = all_reviews[:max_reviews]
        
        if session_id:
            analysis_status[session_id] = {"status": "loading", "message": "Finalizing analysis..."}
        
        return {
            "product_id": product_id,
            "product_url": url,
            "reviews": reviews,
            "total_pages_scraped": current_page,
            "analyzed_at": datetime.now().isoformat()
        }


# Initialize analyzer once at startup
try:
    analyzer = TFIDFSentimentAnalyzer()
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    analyzer = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    if not analyzer:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.json
    url = data.get('url', '').strip()
    max_reviews = data.get('max_reviews', 50)
    
    if not url:
        return jsonify({"error": "URL is required"}), 400
    
    if not re.search(r'walmart\.com', url, re.IGNORECASE):
        return jsonify({"error": "Invalid Walmart URL"}), 400
    
    session_id = str(time.time())
    
    # Initialize status immediately
    analysis_status[session_id] = {
        "status": "loading",
        "message": "Initializing browser..."
    }
    
    def run_analysis():
        scraper = None
        try:
            analysis_status[session_id] = {
                "status": "loading",
                "message": "Starting browser..."
            }
            
            scraper = WalmartReviewInference(analyzer, headless=False)
            
            analysis_status[session_id] = {
                "status": "loading",
                "message": "Browser started, beginning analysis..."
            }
            
            result = scraper.scrape_and_analyze(url, max_reviews, session_id)
            
            reviews = result['reviews']
            
            if not reviews:
                analysis_status[session_id] = {
                    "status": "error",
                    "message": "No reviews found for this product"
                }
                return
            
            positive = [r for r in reviews if r.get('sentiment') == 'positive']
            negative = [r for r in reviews if r.get('sentiment') == 'negative']
            neutral = [r for r in reviews if r.get('sentiment') == 'neutral']
            
            confidences = [r.get('confidence', 0) for r in reviews]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            avg_probs = {
                'negative': float(np.mean([r.get('probabilities', {}).get('negative', 0) for r in reviews])),
                'neutral': float(np.mean([r.get('probabilities', {}).get('neutral', 0) for r in reviews])),
                'positive': float(np.mean([r.get('probabilities', {}).get('positive', 0) for r in reviews]))
            }
            
            ratings = [r.get('rating') for r in reviews if r.get('rating')]
            avg_rating = sum(ratings) / len(ratings) if ratings else None
            
            positive.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            negative.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            neutral.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            analysis_status[session_id] = {
                "status": "complete",
                "data": {
                    "metadata": {
                        "product_id": result['product_id'],
                        "product_url": result['product_url'],
                        "total_reviews": len(reviews),
                        "total_pages_scraped": result.get('total_pages_scraped', 1),
                        "positive_count": len(positive),
                        "negative_count": len(negative),
                        "neutral_count": len(neutral),
                        "average_confidence": round(avg_confidence, 4),
                        "average_probabilities": avg_probs,
                        "average_rating": round(avg_rating, 2) if avg_rating else None,
                        "analyzed_at": result['analyzed_at']
                    },
                    "samples": {
                        "positive": positive[:3],
                        "negative": negative[:3],
                        "neutral": neutral[:3]
                    }
                }
            }
            
        except ValueError as e:
            analysis_status[session_id] = {
                "status": "error",
                "message": str(e)
            }
        except Exception as e:
            error_msg = str(e)
            if "no such window" in error_msg.lower() or "target window already closed" in error_msg.lower():
                error_msg = "Browser was closed unexpectedly. Please keep the browser window open during analysis."
            
            analysis_status[session_id] = {
                "status": "error",
                "message": error_msg
            }
        finally:
            if scraper:
                scraper.close()
    
    thread = threading.Thread(target=run_analysis, daemon=True)
    thread.start()
    
    return jsonify({"session_id": session_id}), 202


@app.route('/api/status/<session_id>')
def get_status(session_id):
    status = analysis_status.get(session_id, {"status": "not_found"})
    return jsonify(status)


if __name__ == '__main__':
    app.run(debug=True, port=5000)