"""
scrape_reddit.py
----------------
Scrapes posts from Reddit subreddits into data/raw/reddit_scrape.jsonl

Usage:
    python scripts/scrape_reddit.py

Setup:
    1. Go to https://www.reddit.com/prefs/apps
    2. Click 'Create App' -> choose 'script'
    3. Fill in name: CognitiveMotive, redirect uri: http://localhost
    4. Copy the client_id (under the app name) and client_secret
    5. Add them to your .env file
"""

import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
import praw
from tqdm import tqdm

load_dotenv()

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = RAW_DIR / "reddit_scrape.jsonl"

# Subreddits and how many posts to pull from each
SUBREDDITS = {
    "AmItheAsshole":      500,   # "AITA for doing X" — rich behaviour descriptions
    "relationship_advice": 500,  # "my partner did X" — third-person behaviour analysis
    "BreakUps":           300,   # leaving relationships
    "antiwork":           300,   # quitting jobs, work resentment
    "confessions":        300,   # honest first-person behaviour accounts
    "decidingtobebetter": 200,   # positive behaviour change
    "offmychest":         200,   # raw emotional behaviour explanations
}

def setup_reddit():
    client_id     = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent    = os.getenv("REDDIT_USER_AGENT", "CognitiveMotive/1.0")

    if not client_id or not client_secret:
        raise ValueError(
            "Missing Reddit credentials in .env\n"
            "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET"
        )

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )


def scrape_subreddit(reddit, subreddit_name: str, limit: int) -> list[dict]:
    records = []
    subreddit = reddit.subreddit(subreddit_name)

    for post in tqdm(subreddit.hot(limit=limit), total=limit, desc=subreddit_name):
        if post.is_self and len(post.selftext) > 100:
            # Get top 3 comments as the "explanation" side of the pair
            post.comments.replace_more(limit=0)
            top_comments = [
                c.body for c in post.comments.list()[:3]
                if len(c.body) > 50 and not c.body.startswith("[")
            ]

            if not top_comments:
                continue

            records.append({
                "source":    subreddit_name,
                "post_id":   post.id,
                "title":     post.title,
                "body":      post.selftext[:2000],  # cap at 2000 chars
                "comments":  top_comments,
                "score":     post.score,
                "url":       f"https://reddit.com{post.permalink}",
            })

        time.sleep(0.5)  # be polite to Reddit's API

    return records


def main():
    print("Connecting to Reddit API...")
    try:
        reddit = setup_reddit()
        reddit.user.me()  # test auth
        print("Connected.\n")
    except Exception as e:
        print(f"Auth failed: {e}")
        return

    all_records = []

    for subreddit_name, limit in SUBREDDITS.items():
        try:
            records = scrape_subreddit(reddit, subreddit_name, limit)
            all_records.extend(records)
            print(f"  -> {len(records)} posts from r/{subreddit_name}\n")
        except Exception as e:
            print(f"[error] r/{subreddit_name}: {e}")

    # Save as jsonl (one record per line — easy to stream during training)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(all_records)} posts to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
