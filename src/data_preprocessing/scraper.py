import asyncio
import aiohttp
from bs4 import BeautifulSoup
import os
import json
import re
from urllib.parse import urljoin, urlparse
from src.data_preprocessing.logger import logging
from src.data_preprocessing.exception import CustomException
import sys
import os
from dotenv import load_dotenv
load_dotenv()
# Configuration
BASE_URL = os.getenv('BASE_URL')
DATA_FOLDER = os.getenv('DATA_FOLDER')
URL_LIST_FILE = os.path.join(DATA_FOLDER, "urls_to_scrape.json")
MAX_DEPTH = os.getenv('MAX_DEPTH')
CONCURRENT_REQUESTS = os.getenv('CONCURRENT_REQUESTS')

# Ensure data folder exists
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

def safe_filename(url):
    try:
        parsed = urlparse(url)
        path = parsed.path.strip('/') or 'index'
        filename = re.sub(r'[^A-Za-z0-9_\-]', '_', path) + ".json"
        return os.path.join(DATA_FOLDER, filename)
    except Exception as e:
        raise CustomException(e,sys)

async def fetch(session, url):
    try:
        async with session.get(url) as response:
            if response.status != 200:
                return None
            return await response.text()
    except Exception as e:
        return CustomException(e,sys)

async def collect_urls():
    """Scrapes BASE_URL for internal links and stores them."""
    try:
        async with aiohttp.ClientSession() as session:
            html = await fetch(session, BASE_URL)
            if not html:
                return
            soup = BeautifulSoup(html, 'html.parser')
            urls = set()
            for link in soup.find_all('a', href=True):
                next_url = urljoin(BASE_URL, link['href']).split('#')[0]
                if urlparse(next_url).netloc == urlparse(BASE_URL).netloc:
                    urls.add(next_url)
            with open(URL_LIST_FILE, 'w') as f:
                json.dump(list(urls), f)
    except Exception as e:
        raise CustomException(e,sys)

def collect_urls_task():
    """Wrapper for Airflow task."""
    try:
        asyncio.run(collect_urls())
    except Exception as e:
        raise CustomException(e,sys)

async def scrape_url(url, session):
    """Scrapes a single URL and saves the content."""
    try:
        html = await fetch(session, url)
        if not html:
            return
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        page_data = {"url": url, "text": text}
        with open(safe_filename(url), 'w', encoding='utf-8') as f:
            json.dump(page_data, f, indent=4)
    except Exception as e:
        raise CustomException(e,sys)

async def scrape_all_urls():
    """Reads URLs from file and scrapes them asynchronously."""
    try:
        if not os.path.exists(URL_LIST_FILE):
            return
        with open(URL_LIST_FILE) as f:
            urls = json.load(f)
        async with aiohttp.ClientSession() as session:
            await asyncio.gather(*(scrape_url(url, session) for url in urls))
    except Exception as e:
        raise CustomException(e,sys)
    
def scrape_all_urls_task():
    """Wrapper for Airflow task."""
    try:
        asyncio.run(scrape_all_urls())
    except Exception as e:
        raise CustomException(e,sys)

if __name__=="__main__":
    scrape_all_urls_task()