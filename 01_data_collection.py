import pandas as pd
import datetime
import time
import argparse
import re
import langdetect
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

def parse_arguments():
    parser = argparse.ArgumentParser(description='Scrape Twitter for specific topics with replies')
    parser.add_argument('--topic', type=str, help='Topic to search for')
    parser.add_argument('--start_date', type=str, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, help='End date in YYYY-MM-DD format')
    parser.add_argument('--output', type=str, default='twitter_data.csv', help='Output CSV filename')
    parser.add_argument('--limit', type=int, default=200, help='Maximum number of top-level posts to scrape')
    parser.add_argument('--replies_limit', type=int, default=50, help='Maximum number of replies to collect per post')
    parser.add_argument('--headless', action='store_true', help='Run browser in headless mode')
    parser.add_argument('--english_only', action='store_true', help='Only collect English tweets')
    args = parser.parse_args()

    # Prompt the user for input if any required argument is missing
    if not args.topic:
        args.topic = input("Enter the topic to search for: ")
    if not args.start_date:
        args.start_date = input("Enter the start date (YYYY-MM-DD): ")
    if not args.end_date:
        args.end_date = input("Enter the end date (YYYY-MM-DD): ")

    return args

def setup_driver(headless=True):
    """Set up and return a Chrome webdriver with appropriate options"""
    options = Options()
    if headless:
        options.add_argument("--headless=new")  # Run in headless mode (no GUI)
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    # Set page load timeout
    driver.set_page_load_timeout(30)
    
    return driver

def is_english_text(text):
    """Check if text is in English using langdetect"""
    try:
        return langdetect.detect(text) == 'en'
    except:
        # If detection fails, assume it's English
        return True

def get_metrics(tweet_element):
    """Extract metrics (likes, retweets, replies) from a tweet element with better selectors"""
    metrics = {'reply': 0, 'retweet': 0, 'like': 0}
    
    try:
        # Try multiple selectors for metrics
        # First approach - direct group selectors
        group_elements = tweet_element.find_elements(By.CSS_SELECTOR, '[data-testid="reply"], [data-testid="retweet"], [data-testid="like"]')
        
        for element in group_elements:
            try:
                test_id = element.get_attribute('data-testid')
                if test_id in metrics:
                    # Look for the text content - often within a span
                    count_text = element.text.strip()
                    if count_text:
                        # Extract only the number from text like "42 replies"
                        count_match = re.search(r'(\d+(?:\.\d+)?[KkMm]?)', count_text)
                        if count_match:
                            count_text = count_match.group(1)
                            
                            # Convert K/M to numbers
                            if 'K' in count_text or 'k' in count_text:
                                metrics[test_id] = int(float(count_text.lower().replace('k', '')) * 1000)
                            elif 'M' in count_text or 'm' in count_text:
                                metrics[test_id] = int(float(count_text.lower().replace('m', '')) * 1000000)
                            else:
                                metrics[test_id] = int(count_text) if count_text.isdigit() else 0
            except Exception as e:
                print(f"Error parsing metric {test_id}: {e}")
                
        # Second approach - look for spans with counts inside the elements
        if all(v == 0 for v in metrics.values()):
            for metric in metrics.keys():
                try:
                    elements = tweet_element.find_elements(By.XPATH, f'.//div[@data-testid="{metric}"]//span[contains(@class, "css-")]')
                    for el in elements:
                        count_text = el.text.strip()
                        if count_text and (count_text.isdigit() or 'K' in count_text or 'M' in count_text):
                            if 'K' in count_text or 'k' in count_text:
                                metrics[metric] = int(float(count_text.lower().replace('k', '')) * 1000)
                            elif 'M' in count_text or 'm' in count_text:
                                metrics[metric] = int(float(count_text.lower().replace('m', '')) * 1000000)
                            else:
                                metrics[metric] = int(count_text) if count_text.isdigit() else 0
                            break
                except:
                    pass
    except Exception as e:
        print(f"Error getting metrics: {e}")
        
    return metrics

def scrape_twitter(topic, start_date, end_date, limit, replies_limit, headless=True, english_only=True):
    print(f"Scraping Twitter for topic: {topic}")
    print(f"Settings: Limit={limit}, Replies limit={replies_limit}, English only={english_only}")

    # Format dates for Twitter's search syntax
    start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    formatted_start = start_date_obj.strftime('%Y-%m-%d')
    formatted_end = end_date_obj.strftime('%Y-%m-%d')

    # Initialize webdriver
    driver = setup_driver(headless)

    # Construct search URL (Twitter's advanced search with language filter if needed)
    lang_filter = "%20lang%3Aen" if english_only else ""
    search_url = f"https://twitter.com/search?q={topic}{lang_filter}%20until%3A{formatted_end}%20since%3A{formatted_start}&src=typed_query&f=top"

    try:
        driver.get(search_url)
        print(f"Accessing URL: {search_url}")
        time.sleep(5)  # Wait for page to load
        
        # Handle any cookie/login popups
        try:
            dismiss_buttons = driver.find_elements(By.XPATH, "//span[contains(text(), 'Dismiss')]")
            for button in dismiss_buttons:
                button.click()
                time.sleep(1)
        except:
            pass
            
        try:
            # For the "See what's happening" popup
            not_now_buttons = driver.find_elements(By.XPATH, "//span[contains(text(), 'Not now')]")
            for button in not_now_buttons:
                button.click()
                time.sleep(1)
        except:
            pass
    except Exception as e:
        print(f"Error accessing Twitter: {e}")
        driver.quit()
        return []

    tweets_list = []
    processed_tweets = set()
    scroll_attempts = 0
    max_scroll_attempts = 80  # Increased to ensure we can get enough tweets

    # Scroll and collect tweets
    previous_height = driver.execute_script("return document.body.scrollHeight")

    with tqdm(total=limit, desc="Collecting tweets") as pbar:
        while len(processed_tweets) < limit and scroll_attempts < max_scroll_attempts:
            try:
                # Find tweet elements
                tweet_elements = driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="tweet"]')
                
                if len(tweet_elements) > 0:
                    print(f"Found {len(tweet_elements)} tweet elements on page")

                for tweet in tweet_elements:
                    if len(processed_tweets) >= limit:
                        break
                        
                    try:
                        # Extract tweet ID from the article element
                        tweet_links = tweet.find_elements(By.CSS_SELECTOR, 'a[href*="/status/"]')
                        if not tweet_links:
                            continue

                        tweet_url = tweet_links[0].get_attribute('href')
                        if not re.search(r'/status/(\d+)', tweet_url):
                            continue
                            
                        tweet_id = re.search(r'/status/(\d+)', tweet_url).group(1)

                        # Skip if we've already processed this tweet
                        if tweet_id in processed_tweets:
                            continue
                        
                        # Check if this is a top-level tweet (not a reply)
                        is_reply = False
                        try:
                            reply_element = tweet.find_element(By.CSS_SELECTOR, 'div[data-testid="socialContext"]')
                            if "Replying to" in reply_element.text:
                                is_reply = True
                                continue  # Skip replies in main search, we'll get them when we visit tweets
                        except:
                            pass
                            
                        # Get username
                        try:
                            username_element = tweet.find_element(By.CSS_SELECTOR, 'div[data-testid="User-Name"] a')
                            username = username_element.get_attribute('href').split('/')[-1]
                        except:
                            try:
                                username_element = tweet.find_element(By.CSS_SELECTOR, 'a[data-testid="User-Name"]')
                                username = username_element.get_attribute('href').split('/')[-1]
                            except:
                                username = "unknown"

                        # Get tweet text
                        try:
                            text_element = tweet.find_element(By.CSS_SELECTOR, 'div[data-testid="tweetText"]')
                            text = text_element.text
                        except:
                            text = ""
                            
                        # Skip if not English and option selected
                        if english_only and text and not is_english_text(text):
                            print(f"Skipping non-English tweet: {text[:50]}...")
                            continue

                        # Get timestamp
                        try:
                            time_element = tweet.find_element(By.CSS_SELECTOR, 'time')
                            timestamp = time_element.get_attribute('datetime')
                        except:
                            timestamp = datetime.datetime.now().isoformat()
                            
                        # Get metrics using the improved function
                        metrics = get_metrics(tweet)
                            
                        # Store the tweet with the specific field names requested
                        tweet_data = {
                            'id': tweet_id,
                            'speaker_nm': username,
                            'conversation_id': tweet_id,
                            'reply_to_id': None,  # Top-level tweets don't have a reply_to
                            'reply_to_nm': None,  # Top-level tweets don't have a reply_to_nm
                            'timestamp': timestamp,
                            'text': text,
                            'likes': metrics.get('like', 0),
                            'shares': metrics.get('retweet', 0),
                            'num_comments': metrics.get('reply', 0)
                        }
                        
                        tweets_list.append(tweet_data)
                        processed_tweets.add(tweet_id)
                        
                        print(f"Processed tweet {len(processed_tweets)}: @{username} - Likes: {metrics.get('like', 0)}, "
                              f"Shares: {metrics.get('retweet', 0)}, Comments: {metrics.get('reply', 0)}")
                        
                        # Now visit the tweet to get replies if there are any
                        if metrics.get('reply', 0) > 0:
                            replies = get_twitter_replies(driver, tweet_url, tweet_id, username, replies_limit, english_only)
                            tweets_list.extend(replies)  # Ensure replies are added to the main list
                            
                            # Add a small pause after processing a tweet with many replies
                            if metrics.get('reply', 0) > 20:
                                print(f"Taking a short break after processing many replies...")
                                time.sleep(2)
                            
                        pbar.update(1)

                    except Exception as e:
                        print(f"Error processing tweet: {e}")
                        continue

                # Scroll down
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(3)

                # Check if page has loaded new content
                current_height = driver.execute_script("return document.body.scrollHeight")
                if current_height == previous_height:
                    scroll_attempts += 1
                    print(f"No new content loaded, scroll attempt {scroll_attempts}/{max_scroll_attempts}")
                    
                    # Try another scroll with a different approach if we're getting stuck
                    if scroll_attempts % 3 == 0:
                        print("Trying alternative scroll method...")
                        driver.execute_script("window.scrollBy(0, -100);")
                        time.sleep(1)
                        driver.execute_script("window.scrollBy(0, 200);")
                        time.sleep(2)
                else:
                    scroll_attempts = 0
                    print("New content loaded, continuing to scroll")

                previous_height = current_height

            except Exception as e:
                print(f"Error during Twitter scrolling: {e}")
                scroll_attempts += 1

    driver.quit()
    print(f"Scraped {len(tweets_list)} tweets in total (including replies)")
    return tweets_list

def get_twitter_replies(driver, tweet_url, parent_id, parent_username, replies_limit, english_only=True):
    """Visit a tweet page and extract replies"""
    replies = []
    try:
        print(f"Getting replies for tweet: {tweet_url}")
        
        # Open tweet in a new tab
        driver.execute_script(f"window.open('{tweet_url}', '_blank');")
        driver.switch_to.window(driver.window_handles[-1])
        
        # Wait for the page to load
        time.sleep(5)
        
        # Scroll a bit to load replies
        scroll_attempts = 0
        max_scroll_attempts = 20  # Increased to get more replies
        processed_reply_ids = set()  # Track processed replies
        
        with tqdm(total=replies_limit, desc="Collecting replies") as pbar:
            while scroll_attempts < max_scroll_attempts and len(replies) < replies_limit:
                # Find reply elements
                reply_elements = driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="tweet"]')
                
                if len(reply_elements) > 0:
                    print(f"Found {len(reply_elements)} reply elements")
                
                for i, reply in enumerate(reply_elements):
                    # Skip the first one as it's usually the parent tweet
                    if i == 0:
                        continue
                        
                    if len(replies) >= replies_limit:
                        break
                        
                    try:
                        # Extract reply data
                        reply_links = reply.find_elements(By.CSS_SELECTOR, 'a[href*="/status/"]')
                        if not reply_links:
                            continue
                            
                        reply_url = reply_links[0].get_attribute('href')
                        reply_id = re.search(r'/status/(\d+)', reply_url).group(1)
                        
                        # Skip if we've already processed this reply
                        if reply_id in processed_reply_ids:
                            continue
                            
                        processed_reply_ids.add(reply_id)
                        
                        # Get username
                        try:
                            username_element = reply.find_element(By.CSS_SELECTOR, 'div[data-testid="User-Name"] a')
                            username = username_element.get_attribute('href').split('/')[-1]
                        except:
                            try:
                                username_element = reply.find_element(By.CSS_SELECTOR, 'a[data-testid="User-Name"]')
                                username = username_element.get_attribute('href').split('/')[-1]
                            except:
                                username = "unknown"
                        
                        # Get reply text
                        try:
                            text_element = reply.find_element(By.CSS_SELECTOR, 'div[data-testid="tweetText"]')
                            text = text_element.text
                        except:
                            text = ""
                            
                        # Skip if not English and option selected
                        if english_only and text and not is_english_text(text):
                            print(f"Skipping non-English reply: {text[:50]}...")
                            continue
                        
                        # Get timestamp
                        try:
                            time_element = reply.find_element(By.CSS_SELECTOR, 'time')
                            timestamp = time_element.get_attribute('datetime')
                        except:
                            timestamp = datetime.datetime.now().isoformat()
                        
                        # Get metrics using improved function
                        metrics = get_metrics(reply)
                        
                        # Store the reply with the specific field names requested
                        reply_data = {
                            'id': reply_id,
                            'speaker_nm': username,
                            'conversation_id': parent_id,
                            'reply_to_id': parent_id,
                            'reply_to_nm': parent_username,
                            'timestamp': timestamp,
                            'text': text,
                            'likes': metrics.get('like', 0),
                            'shares': metrics.get('retweet', 0),
                            'num_comments': metrics.get('reply', 0)
                        }
                        
                        replies.append(reply_data)
                        print(f"Processed reply: @{username} - Likes: {metrics.get('like', 0)}, "
                              f"Shares: {metrics.get('retweet', 0)}, Comments: {metrics.get('reply', 0)}")
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        print(f"Error processing reply: {e}")
                        continue
                        
                # Scroll down to load more replies
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
                # Check if page has loaded new content
                current_height = driver.execute_script("return document.body.scrollHeight")
                if current_height == driver.execute_script("return document.body.scrollHeight"):
                    scroll_attempts += 1
                    print(f"No new replies loaded, scroll attempt {scroll_attempts}/{max_scroll_attempts}")
                    
                    # Try a more aggressive scroll if we're not getting new content
                    if scroll_attempts % 3 == 0:
                        print("Trying alternative scroll technique for replies...")
                        # Scroll up a bit then back down to trigger new content loading
                        driver.execute_script("window.scrollBy(0, -300);")
                        time.sleep(1)
                        driver.execute_script("window.scrollBy(0, 500);")
                        time.sleep(2)
                else:
                    scroll_attempts = 0
                    print("New replies loaded, continuing to scroll")
                
            # Close the tab and switch back
            driver.close()
            driver.switch_to.window(driver.window_handles[0])
            
    except Exception as e:
        print(f"Error getting Twitter replies: {e}")
        # Make sure we're back to the main window
        if len(driver.window_handles) > 1:
            driver.close()
            driver.switch_to.window(driver.window_handles[0])
            
    return replies

def save_to_csv(data, filename):
    print(f"Saving data to {filename}")
    df = pd.DataFrame(data)

    # Ensure all required fields are present
    for field in ['id', 'speaker_nm', 'conversation_id', 'reply_to_id', 'reply_to_nm', 
                 'timestamp', 'text', 'likes', 'shares', 'num_comments']:
        if field not in df.columns:
            df[field] = ''
    
    # Keep only the requested fields in the specified order
    df = df[['id', 'speaker_nm', 'conversation_id', 'reply_to_id', 'reply_to_nm', 
             'timestamp', 'text', 'likes', 'shares', 'num_comments']]

    # Fill NaN values with appropriate placeholders
    df = df.fillna({
        'reply_to_id': 'None',
        'reply_to_nm': 'None',
        'shares': 0,
        'num_comments': 0,
    })

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} records to {filename}")
    
    # Print some statistics
    print("\nData Summary:")
    print(f"Total tweets: {len(df)}")
    print(f"Unique conversations: {df['conversation_id'].nunique()}")
    print(f"Top-level tweets: {len(df[df['reply_to_id'] == 'None'])}")
    print(f"Replies: {len(df[df['reply_to_id'] != 'None'])}")
    print(f"Average likes: {df['likes'].mean():.2f}")
    print(f"Average shares: {df['shares'].mean():.2f}")
    print(f"Average comments: {df['num_comments'].mean():.2f}")

def main():
    args = parse_arguments()

    all_data = []

    # Update default limit if user didn't specify
    if args.limit == 50 or args.limit == 120:  # Check for both old defaults
        args.limit = 200
        print(f"Updated tweet limit to {args.limit}")

    # Scrape Twitter
    try:
        twitter_data = scrape_twitter(args.topic, args.start_date, args.end_date, 
                                      args.limit, args.replies_limit, args.headless, 
                                      english_only=True)  # Set English only to True
        all_data.extend(twitter_data)
        print(f"Scraped {len(twitter_data)} tweets and replies")
    except Exception as e:
        print(f"Error scraping Twitter: {e}")

    # Save all data to CSV
    save_to_csv(all_data, args.output)

if __name__ == "__main__":
    main()
