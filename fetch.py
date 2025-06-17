# fetch.py

import requests
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any

def fetch_all_news_by_date(
    date: str,
    base_url: str = "https://news-ingest.ekhon.tv/api/articles",
    limit_per_page: int = 20,
    output_file: Optional[str] = None,
    delay_between_requests: float = 0.5
) -> Dict[str, Any]:
    """
    Fetch all news articles for a specific date via the API.
    """
    # 1. Validate the date string
    try:
        datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Date must be 'YYYY-MM-DD'")

    all_articles = []
    page = 1

    while True:
        params = {
            'limit': limit_per_page,
            'page': page,
            'startDate': f"'{date}'",
            'endDate': f"'{date}'"
        }
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        docs = data.get('docs', [])
        all_articles.extend(docs)

        # Stop if no more pages
        if not data.get('hasNextPage', False) or not docs:
            break

        page += 1
        time.sleep(delay_between_requests)

    result = {
        'date': date,
        'total_articles': len(all_articles),
        'fetch_timestamp': datetime.now().isoformat(),
        'articles': all_articles
    }

    # Optionally write to disk
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result
