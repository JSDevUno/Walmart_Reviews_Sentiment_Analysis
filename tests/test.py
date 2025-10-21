import requests
import json
import csv
import re
import time
from datetime import datetime
from typing import List, Dict, Optional
from bs4 import BeautifulSoup

class WalmartReviewScraper:
    def __init__(self):
        """Initialize the Walmart Review Scraper (no API keys needed)"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        
    def extract_product_id(self, url: str) -> Optional[str]:
        """Extract product ID from Walmart URL"""
        patterns = [
            r'/ip/[^/]+/(\d+)',  # /ip/product-name/12345
            r'/(\d+)\?',         # /12345?params
            r'/(\d+)$',          # ends with /12345
            r'itemId=(\d+)'      # ?itemId=12345
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def scrape_from_html(self, url: str, max_pages: int = 50) -> List[Dict]:
        """
        Scrape reviews directly from HTML page with pagination
        """
        product_id = self.extract_product_id(url)
        
        if not product_id:
            raise ValueError("Invalid Walmart URL. Could not extract product ID.")
        
        print(f"Extracting reviews for product ID: {product_id}\n")
        
        all_reviews = []
        page = 1
        
        while page <= max_pages:
            try:
                print(f"📄 Fetching page {page}...")
                
                # Construct URL with pagination
                if page == 1:
                    page_url = url
                else:
                    # Add page parameter
                    if '?' in url:
                        page_url = f"{url}&page={page}"
                    else:
                        page_url = f"{url}?page={page}"
                
                response = self.session.get(page_url, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for JSON data embedded in the page
                scripts = soup.find_all('script')
                page_reviews = []
                
                for script in scripts:
                    if script.string and 'reviewText' in script.string:
                        try:
                            # Try to find JSON within the script
                            script_text = script.string
                            
                            # Look for common JSON patterns
                            json_matches = re.findall(r'\{[^{}]*"reviewText"[^{}]*\}', script_text)
                            
                            for json_str in json_matches:
                                try:
                                    # Try to expand to full JSON object
                                    start = script_text.find(json_str)
                                    bracket_count = 0
                                    end = start
                                    
                                    for i, char in enumerate(script_text[start:], start):
                                        if char == '{':
                                            bracket_count += 1
                                        elif char == '}':
                                            bracket_count -= 1
                                            if bracket_count == 0:
                                                end = i + 1
                                                break
                                    
                                    full_json = script_text[start:end]
                                    data = json.loads(full_json)
                                    
                                    review_data = self.parse_review(data)
                                    if review_data and review_data not in page_reviews:
                                        page_reviews.append(review_data)
                                except:
                                    continue
                            
                            # Also try parsing the entire script as JSON
                            try:
                                data = json.loads(script_text)
                                reviews = self.extract_reviews_from_json(data)
                                for review in reviews:
                                    if review not in page_reviews:
                                        page_reviews.append(review)
                            except:
                                pass
                                
                        except Exception as e:
                            continue
                
                if not page_reviews:
                    print(f"   No reviews found on page {page}. Stopping.")
                    break
                
                print(f"   ✅ Found {len(page_reviews)} reviews")
                all_reviews.extend(page_reviews)
                
                # Check if there's a next page
                # Look for pagination elements
                next_button = soup.find('button', {'aria-label': 'next page'}) or \
                             soup.find('a', string=re.compile(r'Next', re.I))
                
                if not next_button or 'disabled' in next_button.get('class', []):
                    print(f"\n   📌 Reached last page")
                    break
                
                page += 1
                time.sleep(1)  # Be polite to the server
                
            except Exception as e:
                print(f"   ⚠️  Error on page {page}: {e}")
                break
        
        print(f"\n{'='*60}")
        print(f"✅ Total reviews scraped: {len(all_reviews)}")
        print(f"{'='*60}\n")
        return all_reviews
    
    def extract_reviews_from_json(self, data: Dict, reviews: List = None) -> List[Dict]:
        """Recursively search for reviews in nested JSON data"""
        if reviews is None:
            reviews = []
        
        if isinstance(data, dict):
            # Check if this dict contains review data
            if 'reviewText' in data or 'review' in data:
                review_data = self.parse_review(data)
                if review_data:
                    # Check for duplicates
                    is_duplicate = any(
                        r.get('review_text') == review_data.get('review_text') and
                        r.get('reviewer_name') == review_data.get('reviewer_name')
                        for r in reviews
                    )
                    if not is_duplicate:
                        reviews.append(review_data)
            
            # Check for arrays of reviews
            for key in ['reviews', 'customerReviews', 'reviewList']:
                if key in data and isinstance(data[key], list):
                    for review in data[key]:
                        if isinstance(review, dict):
                            review_data = self.parse_review(review)
                            if review_data:
                                is_duplicate = any(
                                    r.get('review_text') == review_data.get('review_text') and
                                    r.get('reviewer_name') == review_data.get('reviewer_name')
                                    for r in reviews
                                )
                                if not is_duplicate:
                                    reviews.append(review_data)
            
            # Recurse into nested dicts
            for value in data.values():
                if isinstance(value, (dict, list)):
                    self.extract_reviews_from_json(value, reviews)
        
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    self.extract_reviews_from_json(item, reviews)
        
        return reviews
    
    def parse_review(self, review: Dict) -> Optional[Dict]:
        """Parse review data from various possible formats"""
        try:
            # Try different possible field names
            reviewer_name = (
                review.get('userNickname') or 
                review.get('reviewerName') or 
                review.get('nickname') or
                review.get('authorName') or
                'Anonymous'
            )
            
            rating = (
                review.get('rating') or 
                review.get('overallRating', {}).get('rating') if isinstance(review.get('overallRating'), dict) else review.get('overallRating') or
                review.get('stars')
            )
            
            title = (
                review.get('reviewTitle') or 
                review.get('title') or 
                review.get('headline') or
                ''
            )
            
            review_text = (
                review.get('reviewText') or 
                review.get('text') or 
                review.get('review') or
                review.get('comments') or
                ''
            )
            
            date = (
                review.get('reviewSubmissionTime') or 
                review.get('submissionTime') or 
                review.get('date') or
                review.get('createdAt') or
                ''
            )
            
            verified = (
                review.get('isVerifiedPurchase', False) or 
                review.get('verifiedPurchase', False) or
                review.get('verified', False)
            )
            
            helpful = (
                review.get('positiveFeedbackCount', 0) or 
                review.get('upVotes', 0) or
                review.get('helpfulCount', 0) or
                0
            )
            
            # Only return if we have at least review text
            if review_text:
                return {
                    'reviewer_name': str(reviewer_name),
                    'rating': rating,
                    'title': str(title),
                    'review_text': str(review_text),
                    'date': str(date),
                    'verified_purchase': bool(verified),
                    'helpful_count': int(helpful) if helpful else 0
                }
            
            return None
            
        except Exception as e:
            print(f"   ⚠️  Error parsing review: {e}")
            return None
    
    def save_to_json(self, reviews: List[Dict], filename: str = None):
        """Save reviews to JSON file"""
        if filename is None:
            filename = f"walmart_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(reviews, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Reviews saved to {filename}")
    
    def save_to_csv(self, reviews: List[Dict], filename: str = None):
        """Save reviews to CSV file"""
        if filename is None:
            filename = f"walmart_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        if not reviews:
            print("No reviews to save")
            return
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['reviewer_name', 'rating', 'title', 'review_text', 'date', 'verified_purchase', 'helpful_count']
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            
            writer.writeheader()
            for review in reviews:
                writer.writerow(review)
        
        print(f"💾 Reviews saved to {filename}")


def main():
    """Example usage"""
    
    print("=" * 70)
    print(" " * 15 + "WALMART REVIEW SCRAPER")
    print(" " * 12 + "(No API Keys Required)")
    print("=" * 70)
    print()
    
    # Initialize scraper
    scraper = WalmartReviewScraper()
    
    # Get product URL from user
    product_url = input("🔗 Enter Walmart product URL: ").strip()
    
    if not product_url:
        print("❌ No URL provided. Exiting...")
        return
    
    try:
        # Scrape reviews
        print("\n🔍 Starting scraping process...\n")
        reviews = scraper.scrape_from_html(product_url, max_pages=50)
        
        if reviews:
            # Remove duplicates based on review text and reviewer
            unique_reviews = []
            seen = set()
            
            for review in reviews:
                key = (review.get('review_text', ''), review.get('reviewer_name', ''))
                if key not in seen:
                    seen.add(key)
                    unique_reviews.append(review)
            
            if len(reviews) != len(unique_reviews):
                print(f"🔄 Removed {len(reviews) - len(unique_reviews)} duplicate reviews")
                reviews = unique_reviews
            
            # Save to both JSON and CSV
            scraper.save_to_json(reviews)
            scraper.save_to_csv(reviews)
            
            # Print summary
            print("\n" + "=" * 70)
            print(" " * 25 + "📊 REVIEW SUMMARY")
            print("=" * 70)
            
            ratings = [r['rating'] for r in reviews if r.get('rating') and isinstance(r.get('rating'), (int, float))]
            
            if ratings:
                avg_rating = sum(ratings) / len(ratings)
                print(f"\n   Total Reviews: {len(reviews)}")
                print(f"   Average Rating: {avg_rating:.2f} ⭐ / 5")
                
                verified = sum(1 for r in reviews if r.get('verified_purchase'))
                print(f"   Verified Purchases: {verified}")
                
                # Rating distribution
                print(f"\n   Rating Distribution:")
                for star in range(5, 0, -1):
                    count = sum(1 for r in ratings if r == star)
                    bar = "█" * (count // max(1, len(ratings) // 20))
                    print(f"   {star} ⭐: {count:3d} {bar}")
            else:
                print(f"\n   Total Reviews: {len(reviews)}")
            
            # Show sample reviews
            if reviews:
                print(f"\n" + "=" * 70)
                print(" " * 23 + "📝 SAMPLE REVIEWS")
                print("=" * 70)
                
                for i, review in enumerate(reviews[:3], 1):
                    print(f"\n   Review #{i}:")
                    print(f"   Reviewer: {review.get('reviewer_name', 'Anonymous')}")
                    if review.get('rating'):
                        print(f"   Rating: {review.get('rating')} ⭐")
                    if review.get('title'):
                        print(f"   Title: {review.get('title')}")
                    text = review.get('review_text', 'N/A')
                    print(f"   Text: {text[:200]}..." if len(text) > 200 else f"   Text: {text}")
                    if review.get('date'):
                        print(f"   Date: {review.get('date')}")
                    print()
        else:
            print("\n" + "=" * 70)
            print("❌ No reviews found. Possible reasons:")
            print("   • Product has no reviews")
            print("   • Walmart changed their page structure")
            print("   • URL is invalid")
            print("   • Network issues")
            print("=" * 70)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()