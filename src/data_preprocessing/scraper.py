import asyncio
import aiohttp
from bs4 import BeautifulSoup
import os
import re
from urllib.parse import urljoin, urlparse

# Configuration
BASE_URL = 'https://www.khoury.northeastern.edu/'
MAX_DEPTH = 3             # maximum recursion depth (base URL is depth 0)
CONCURRENT_REQUESTS = 10  # maximum number of concurrent requests

# Create folder for scraped pages
if not os.path.exists('scraped_pages'):
    os.makedirs('scraped_pages')

def safe_filename(url):
    """
    Generates a safe filename based on the URL path.
    """
    parsed = urlparse(url)
    # Use 'index' if the path is empty
    path = parsed.path.strip('/') or 'index'
    # Replace any characters that aren't letters, numbers, underscores, or dashes
    filename = re.sub(r'[^A-Za-z0-9_\-]', '_', path)
    return os.path.join('scraped_pages', f"{filename}.html")

# Global visited set and its lock to avoid re-scraping URLs
visited = set()
visited_lock = asyncio.Lock()

async def fetch(session, url, semaphore):
    """
    Asynchronously fetches the content of the URL using a semaphore to limit concurrency.
    """
    try:
        async with semaphore:
            async with session.get(url) as response:
                if response.status != 200:
                    print(f"Failed to retrieve {url} (status code: {response.status})")
                    return None
                text = await response.text()
                return text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

async def async_scrape(url, depth=0, session=None, semaphore=None):
    """
    Recursively scrape pages asynchronously up to MAX_DEPTH.
    Uses a global visited set to ensure each URL is scraped only once.
    """
    if depth > MAX_DEPTH:
        return

    # Check and update the global visited set
    async with visited_lock:
        if url in visited:
            return
        visited.add(url)

    print(f"Scraping (depth {depth}): {url}")

    html = await fetch(session, url, semaphore)
    if html is None:
        return

    # Save the page content to a file
    file_path = safe_filename(url)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(html)

    # Parse the page and schedule scraping of internal links
    soup = BeautifulSoup(html, 'html.parser')
    tasks = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        next_url = urljoin(url, href)
        # Only follow links within the same domain
        if urlparse(next_url).netloc == urlparse(BASE_URL).netloc:
            next_url = next_url.split('#')[0]  # Remove URL fragments
            tasks.append(async_scrape(next_url, depth + 1, session, semaphore))
    if tasks:
        await asyncio.gather(*tasks)

async def main():
    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession() as session:
        await async_scrape(BASE_URL, depth=0, session=session, semaphore=semaphore)
    print("Scraping complete.")

if __name__ == '__main__':
    asyncio.run(main())
