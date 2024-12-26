#!/usr/bin/env python3
# scraper.py

import argparse
import asyncio
import aiohttp
import os
import re
import json
import csv
import sys
import logging
import yaml
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from typing import List, Dict, Any, Optional, Set
import mimetypes
import hashlib
from pathlib import Path
import aiofiles
import tldextract
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.logging import RichHandler
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from InquirerPy import inquirer
from rich.panel import Panel
from collections import defaultdict
import pickle
import pyppeteer
from aiolimiter import AsyncLimiter

# Configure Rich Console
console = Console()

# Add welcome panel
welcome_message = """
[bold cyan]Welcome to JMScraper![/bold cyan]

A powerful and ethical web scraping tool.

[yellow]Usage:[/yellow]
    - Run with --help for available options
    - Interactive menu will guide you through the process

[blue]GitHub:[/blue] http://github.com/jmitander/JMScraper
[green]Version:[/green] 2.0.0
"""

console.print(
    Panel(
        welcome_message,
        title="[bold red]JMScraper[/bold red]",
        subtitle="[italic]By JM[/italic]",
        border_style="bold blue",
        padding=(1, 2),
        expand=False
    )
)

# Configure Logging with RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("WebScraper")

# Constants
DEFAULT_USER_AGENTS = [
    # Ensure each line is a distinct user-agent
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
]

DEFAULT_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Pragma': 'no-cache',
    'Cache-Control': 'no-cache',
}

# Helper Functions
def extract_emails(text: str) -> List[str]:
    email_regex = re.compile(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    )
    return email_regex.findall(text)

def extract_meta_fallback(text: str, name: str) -> str:
    meta_regex = re.compile(
        rf'<meta\s+name=["\']{name}["\']\s+content=["\']([^"\']+)["\']',
        re.IGNORECASE
    )
    match = meta_regex.search(text)
    return match.group(1).strip() if match else 'N/A'


# Scraper Class with Enhanced Features
class WebScraper:
    def __init__(
        self,
        config: Dict[str, Any],
        resume_file: Optional[str] = None
    ):
        self.urls = config.get('urls', [])
        self.output_file = config.get('output_file', 'results.json')
        self.output_format = config.get('output_format', 'json')
        self.delay = config.get('delay', 1.0)
        self.proxy = config.get('proxy', None)
        self.mode = config.get('mode', ['metadata'])
        self.concurrent_requests = config.get('concurrency', 5)
        self.alternative_method = config.get('alternative_method', 'beautifulsoup')
        self.max_retries = config.get('max_retries', 3)
        self.recursive = config.get('recursive', False)
        self.max_depth = config.get('max_depth', 1)
        self.subdomains = config.get('subdomains', False)
        self.user_agents = config.get('user_agents', DEFAULT_USER_AGENTS)
        self.ssl_verify = config.get('ssl_verify', True)
        self.handle_js = config.get('handle_js', False)
        self.session_cookies = config.get('cookies', {})
        self.config_file = config.get('config_file', None)

        self.results: List[Dict[str, Any]] = []
        self.visited_urls: Set[str] = set()
        self.domain_whitelist: Set[str] = set()
        self._setup_domain_whitelist()
        self.rate_limit = config.get('rate_limit', self.delay)
        self.dynamic_rate_limit = True
        self.resume_file = resume_file
        self.resume_data = {}
        self._load_resume()

        self.base_output_dir = Path(self.output_file).parent
        self.media_dir = self.base_output_dir / "media"
        self.images_dir = self.media_dir / "images"
        self.videos_dir = self.media_dir / "videos"
        self.documents_dir = self.media_dir / "documents"
        self._create_directories()

        # Initialize Rate Limiter (e.g., 10 requests per second)
        self.rate_limiter = AsyncLimiter(max_rate=10, time_period=1)

        # Initialize Browser for JS Rendering (if handle_js=True)
        self.browser = None
        if self.handle_js:
            asyncio.get_event_loop().run_until_complete(self.launch_browser())

    def _create_directories(self):
        """Create necessary directories for media storage"""
        for directory in [self.media_dir, self.images_dir, self.videos_dir, self.documents_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def _setup_domain_whitelist(self):
        """Setup allowed domains based on input URLs using tldextract"""
        for url in self.urls:
            try:
                parsed = urlparse(url if url.startswith(('http://', 'https://')) else f'https://{url}')
                ext = tldextract.extract(parsed.netloc)
                domain = f"{ext.domain}.{ext.suffix}"
                self.domain_whitelist.add(domain)
            except Exception as e:
                logger.warning("Failed to extract domain from %s: %s", url, e)

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL should be scraped based on settings and normalization"""
        try:
            url, _ = urldefrag(url)  # Remove fragment
            parsed = urlparse(url)
            ext = tldextract.extract(parsed.netloc)
            domain = f"{ext.domain}.{ext.suffix}"

            if self.subdomains:
                valid = any(domain.endswith(d) for d in self.domain_whitelist)
            else:
                valid = domain in self.domain_whitelist

            return valid and bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False

    def _get_headers(self):
        headers = DEFAULT_HEADERS.copy()
        # Rotate user agents based on a hash
        index = int(hashlib.md5(''.join(headers.values()).encode()).hexdigest()[0], 16) % len(self.user_agents)
        headers['User-Agent'] = self.user_agents[index]
        return headers

    def _get_safe_filename(self, url: str, content_type: str = None) -> str:
        """Generate safe filename from URL and content type"""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        ext = mimetypes.guess_extension(content_type) if content_type else os.path.splitext(url)[1]
        return f"{url_hash}{ext if ext else '.bin'}"

    async def _download_media(
        self, session: ClientSession, url: str, content_type: str = None
    ) -> Optional[Dict[str, str]]:
        """Download media file and return metadata"""
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    return None

                content_type = response.headers.get('content-type', content_type)
                if not content_type:
                    return None

                if 'image' in content_type:
                    save_dir = self.images_dir
                    media_type = 'image'
                elif 'video' in content_type:
                    save_dir = self.videos_dir
                    media_type = 'video'
                elif 'application' in content_type or 'text' in content_type:
                    save_dir = self.documents_dir
                    media_type = 'document'
                else:
                    return None

                filename = self._get_safe_filename(url, content_type)
                filepath = save_dir / filename

                async with aiofiles.open(filepath, 'wb') as f:
                    await f.write(await response.read())

                return {
                    'url': url,
                    'local_path': str(filepath),
                    'media_type': media_type,
                    'content_type': content_type,
                    'size': os.path.getsize(filepath)
                }

        except Exception as e:
            logger.warning("Failed to download %s: %s", url, e)
            return None

    async def extract_media(
        self, session: ClientSession, soup: BeautifulSoup, base_url: str
    ) -> Dict[str, List[Dict[str, str]]]:
        """Extract and download all media from the page"""
        media_data = {
            'images': [],
            'videos': [],
            'documents': []
        }

        # Extract and download images
        images = [urljoin(base_url, img['src']) for img in soup.find_all('img', src=True)]
        for url in images:
            if self._is_valid_url(url):
                result = await self._download_media(session, url)
                if result:
                    media_data['images'].append(result)

        # Extract and download videos
        videos = [
            urljoin(base_url, vid['src'])
            for vid in soup.find_all(['video', 'source'], src=True)
        ]
        for url in videos:
            if self._is_valid_url(url):
                result = await self._download_media(session, url)
                if result:
                    media_data['videos'].append(result)

        # Extract and download documents (pdf, doc, etc.)
        documents = [
            urljoin(base_url, a['href'])
            for a in soup.find_all('a', href=True)
            if any(ext in a['href'].lower() for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx'])
        ]
        for url in documents:
            if self._is_valid_url(url):
                result = await self._download_media(session, url)
                if result:
                    media_data['documents'].append(result)

        return media_data

    async def fetch(self, session: ClientSession, url: str) -> Dict[str, Any]:
        """
        Fetch a single URL and return a dictionary with the scraped data.
        This function always returns a dict, never None.
        """
        # Initialize return data
        data = {
            'url': url,
            'error': None
        }

        # Ensure we have a proper scheme
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'

        logger.info("Fetching: %s", url)

        for attempt in range(self.max_retries):
            try:
                session.headers.update(self._get_headers())
                ssl = True if self.ssl_verify else False

                # Handle JavaScript-rendered content
                if self.handle_js:
                    page = await self.browser.newPage()
                    await page.setUserAgent(session.headers['User-Agent'])
                    await page.goto(url, {'waitUntil': 'networkidle2'})
                    text = await page.content()
                    await page.close()
                else:
                    async with session.get(
                        url, timeout=ClientTimeout(total=30),
                        allow_redirects=True, ssl=ssl
                    ) as response:
                        response.raise_for_status()
                        text = await response.text()

                soup = BeautifulSoup(text, 'html.parser')

                # Extract data depending on mode
                if 'all' in self.mode:
                    all_data = self.extract_all(soup, text, url)
                    data.update(all_data)
                else:
                    if 'metadata' in self.mode:
                        metadata = self.extract_metadata(soup, text)
                        data.update(metadata)
                    if 'links' in self.mode:
                        links = self.extract_links(soup, url)
                        data.update(links)
                    if 'images' in self.mode:
                        images = self.extract_images(soup, url)
                        data.update(images)

                # Extract emails and favicon
                data['emails'] = extract_emails(text)
                data['favicon'] = await self.extract_favicon(session, soup, url)

                # Download media if required
                if 'all' in self.mode or 'images' in self.mode:
                    media_data = await self.extract_media(session, soup, url)
                    data.update(media_data)

                # Dynamic Rate Limiting based on response headers
                if (
                    not self.handle_js and hasattr(response, 'headers')
                    and 'Retry-After' in response.headers
                ):
                    self.rate_limit = float(response.headers['Retry-After'])
                    logger.info("Rate limited. Adjusting delay to %s seconds.", self.rate_limit)
                else:
                    self.rate_limit = self.delay

                data['error'] = None
                break  # Success, exit the retry loop

            except aiohttp.ClientResponseError as e:
                if e.status == 429 and attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt * self.rate_limit
                    logger.warning("Rate limit hit for %s. Waiting %s seconds before retrying...", url, wait_time)
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    data['error'] = f"HTTP error: {e.status} {e.message}"
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                data['error'] = f"Connection error: {str(e)}"
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt * self.rate_limit
                    logger.warning("Attempt %s failed for %s. Retrying in %s seconds...", attempt + 1, url, wait_time)
                    await asyncio.sleep(wait_time)
                    continue
            except Exception as e:
                data['error'] = f"Unexpected error: {str(e)}"
                logger.error("Unexpected error for %s: %s", url, e)
                break

        # Always return a dictionary, never None
        # `data` includes an 'error' field if something went wrong
        return data

    def extract_all(self, soup: BeautifulSoup, text: str, base_url: str) -> Dict[str, Any]:
        data = {}
        data.update(self.extract_metadata(soup, text))
        data.update(self.extract_links(soup, base_url))
        data.update(self.extract_images(soup, base_url))
        return data

    def extract_metadata(self, soup: BeautifulSoup, text: str) -> Dict[str, Any]:
        metadata = {}
        try:
            metadata = {
                'title': soup.title.string.strip() if soup.title else 'N/A',
                'description': self.get_meta_content(soup, 'description'),
                'keywords': self.get_meta_content(soup, 'keywords'),
                'h1_tags': [h1.get_text(strip=True) for h1 in soup.find_all('h1')]
            }
            logger.debug("Extracted metadata: %s", metadata)
        except Exception as e:
            logger.warning("Metadata extraction failed, attempting fallback: %s", e)
            metadata = {
                'title': self.extract_title_fallback(text),
                'description': extract_meta_fallback(text, 'description'),
                'keywords': extract_meta_fallback(text, 'keywords'),
                'h1_tags': self.extract_h1_fallback(text)
            }
            logger.debug("Extracted metadata with fallback: %s", metadata)
        return metadata

    def extract_title_fallback(self, text: str) -> str:
        title_regex = re.compile(r'<title>(.*?)</title>', re.IGNORECASE | re.DOTALL)
        match = title_regex.search(text)
        return match.group(1).strip() if match else 'N/A'

    def extract_h1_fallback(self, text: str) -> List[str]:
        h1_regex = re.compile(r'<h1[^>]*>(.*?)</h1>', re.IGNORECASE | re.DOTALL)
        return [match.strip() for match in h1_regex.findall(text)] if h1_regex.search(text) else []

    def get_meta_content(self, soup: BeautifulSoup, name: str) -> str:
        meta = soup.find('meta', attrs={'name': name})
        return meta['content'].strip() if meta and 'content' in meta.attrs else 'N/A'

    def extract_links(self, soup: BeautifulSoup, base_url: str) -> Dict[str, Any]:
        try:
            if self.alternative_method == 'beautifulsoup':
                links = [urljoin(base_url, a['href']) for a in soup.find_all('a', href=True)]
            else:  # fallback to regex
                links = self.extract_links_regex(soup.prettify(), base_url)

            normalized_links = list({
                urldefrag(link)[0]
                for link in links
                if self._is_valid_url(link)
            })

            logger.debug("Extracted %s links", len(normalized_links))
            return {'links': normalized_links}

        except Exception as e:
            logger.warning("Links extraction failed, attempting fallback: %s", e)
            links = self.extract_links_fallback(soup.prettify(), base_url)
            return {'links': links}

    def extract_links_regex(self, text: str, base_url: str) -> List[str]:
        link_regex = re.compile(r'href=["\'](.*?)["\']', re.IGNORECASE)
        raw_links = link_regex.findall(text)
        normalized_links = [urljoin(base_url, link) for link in raw_links]
        return normalized_links

    def extract_links_fallback(self, text: str, base_url: str) -> List[str]:
        link_regex = re.compile(r'href=["\'](.*?)["\']', re.IGNORECASE)
        raw_links = link_regex.findall(text)
        normalized_links = []
        for link in raw_links:
            full_link = urljoin(base_url, link)
            if self._is_valid_url(full_link):
                normalized_links.append(full_link)
        return normalized_links

    def extract_images(self, soup: BeautifulSoup, base_url: str) -> Dict[str, Any]:
        try:
            if self.alternative_method == 'beautifulsoup':
                images = [urljoin(base_url, img['src']) for img in soup.find_all('img', src=True)]
            else:  # fallback to regex
                images = self.extract_images_regex(soup.prettify(), base_url)

            normalized_images = list({
                urldefrag(img)[0]
                for img in images
                if self._is_valid_url(img)
            })
            logger.debug("Extracted %s images", len(normalized_images))
            return {'images': normalized_images}

        except Exception as e:
            logger.warning("Images extraction failed, attempting fallback: %s", e)
            images = self.extract_images_fallback(soup.prettify(), base_url)
            return {'images': images}

    def extract_images_regex(self, text: str, base_url: str) -> List[str]:
        img_regex = re.compile(r'src=["\'](.*?)["\']', re.IGNORECASE)
        raw_images = img_regex.findall(text)
        normalized_images = [urljoin(base_url, img) for img in raw_images]
        return normalized_images

    def extract_images_fallback(self, text: str, base_url: str) -> List[str]:
        img_regex = re.compile(r'src=["\'](.*?)["\']', re.IGNORECASE)
        raw_images = img_regex.findall(text)
        normalized_images = []
        for img in raw_images:
            full_img = urljoin(base_url, img)
            if self._is_valid_url(full_img):
                normalized_images.append(full_img)
        return normalized_images

    async def extract_favicon(self, session: ClientSession, soup: BeautifulSoup, base_url: str) -> str:
        """
        Attempt to find a <link rel="icon" ...>, otherwise default to /favicon.ico.
        Returns 'N/A' if not found.
        """
        try:
            favicon = soup.find('link', rel=lambda x: x and 'icon' in x.lower())
            favicon_url = urljoin(base_url, favicon['href']) if favicon and 'href' in favicon.attrs else 'N/A'
            if favicon_url != 'N/A':
                async with session.head(favicon_url, ssl=self.ssl_verify) as resp:
                    if resp.status != 200:
                        favicon_url = await self.extract_favicon_fallback(base_url, session)
            logger.debug("Extracted favicon: %s", favicon_url)
            return favicon_url
        except Exception as e:
            logger.warning("Favicon extraction failed, attempting fallback: %s", e)
            return await self.extract_favicon_fallback(base_url, session)

    async def extract_favicon_fallback(self, base_url: str, session: ClientSession) -> str:
        favicon_url = urljoin(base_url, '/favicon.ico')
        try:
            async with session.head(favicon_url, ssl=self.ssl_verify) as resp:
                if resp.status == 200:
                    return favicon_url
        except Exception:
            pass
        return 'N/A'

    async def scrape(self, progress: Progress):
        """
        Scrapes all self.urls. If recursive=True, does BFS up to self.max_depth.
        """
        connector = TCPConnector(
            limit_per_host=self.concurrent_requests,
            force_close=True,
            enable_cleanup_closed=True,
            ssl=self.ssl_verify
        )
        timeout = ClientTimeout(total=30)
        cookie_jar = aiohttp.CookieJar()

        if self.session_cookies:
            for name, value in self.session_cookies.items():
                cookie_jar.update_cookies({name: value})

        session_kwargs = {
            'connector': connector,
            'timeout': timeout,
            'trust_env': True,
            'headers': self._get_headers(),
            'cookie_jar': cookie_jar
        }
        if self.proxy:
            session_kwargs['proxy'] = self.proxy

        async with ClientSession(**session_kwargs) as session:
            if self.recursive:
                urls_to_scrape = set([
                    url if url.startswith(('http://', 'https://')) else f'https://{url}'
                    for url in self.urls
                ])
                current_depth = 0
                new_urls = set()

                while urls_to_scrape and current_depth < self.max_depth:
                    logger.info("Scraping in progress")
                    task_id = progress.add_task(
                        f"Depth {current_depth + 1}",
                        total=len(urls_to_scrape)
                    )

                    tasks = []
                    for url in urls_to_scrape:
                        if url not in self.visited_urls:
                            tasks.append(self.bound_fetch(session, url, progress, task_id))
                            self.visited_urls.add(url)

                    # gather results
                    results = await asyncio.gather(*tasks, return_exceptions=False)

                    # Skip None results to avoid 'NoneType' iteration
                    for result in results:
                        if not result:
                            logger.error("A scraping task returned None, skipping.")
                            continue
                        if 'links' in result:
                            for link in result['links']:
                                if self._is_valid_url(link) and link not in self.visited_urls:
                                    new_urls.add(link)

                    urls_to_scrape = new_urls
                    new_urls = set()
                    current_depth += 1

            else:
                task_id = progress.add_task("Scraping...", total=len(self.urls))
                tasks = [self.bound_fetch(session, url, progress, task_id) for url in self.urls]
                results = await asyncio.gather(*tasks, return_exceptions=False)

                # Skip None results if any
                for result in results:
                    if not result:
                        logger.error("A scraping task returned None, skipping.")
                        continue

    async def bound_fetch(self, session: ClientSession, url: str, progress: Progress, task_id: int):
        """
        Rate-limited wrapper around fetch(...). Saves the result to self.results.
        """
        async with self.rate_limiter:
            result = await self.fetch(session, url)
            self.results.append(result)
            progress.advance(task_id, 1)
            self._save_resume(url)

    def _save_resume(self, url: str):
        """Save the current state to a resume file (if specified)."""
        if self.resume_file:
            with open(self.resume_file, 'wb') as f:
                pickle.dump(self.results, f)

    def _load_resume(self):
        """Load previously saved state from self.resume_file (if exists)."""
        if self.resume_file and os.path.exists(self.resume_file):
            with open(self.resume_file, 'rb') as f:
                self.resume_data = pickle.load(f)
                self.results.extend(self.resume_data)
                self.visited_urls.update([entry['url'] for entry in self.resume_data])

    async def launch_browser(self):
        """Launch a single browser instance for JS rendering."""
        if not self.browser:
            self.browser = await pyppeteer.launch(headless=True, args=['--no-sandbox'])

    async def close_browser(self):
        """Close the browser instance if handle_js=True."""
        if self.browser:
            await self.browser.close()
            self.browser = None

    def save_results(self):
        console.print("\nSaving Results")
        try:
            if self.output_format.lower() == 'json':
                console.print(f"Writing data to {self.output_file}...")
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f, indent=4, ensure_ascii=False)
                total_size = os.path.getsize(self.output_file)
                console.print(f"Saved {len(self.results)} items ({total_size/1024:.1f} KB) to {self.output_file}")

            elif self.output_format.lower() == 'csv':
                if not self.results:
                    logger.warning("No data to write to CSV.")
                    return
                console.print(f"Writing data to {self.output_file}...")
                keys = set()
                for entry in self.results:
                    keys.update(entry.keys())
                keys = sorted(keys)
                with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(self.results)
                total_size = os.path.getsize(self.output_file)
                console.print(f"Saved {len(self.results)} rows ({total_size/1024:.1f} KB) to {self.output_file}")
            else:
                logger.error("Unsupported format: %s", self.output_format)
                sys.exit(1)
        except Exception as e:
            logger.error("Failed to save data: %s", e)

        # Media Summary
        console.print("\nMedia Download Summary:")
        media_summary = defaultdict(int)
        total_media_size = 0
        for result in self.results:
            for media_type in ['images', 'videos', 'documents']:
                media_items = result.get(media_type, [])
                if media_items:
                    media_summary[media_type] += len(media_items)
                    total_media_size += sum(item['size'] for item in media_items)

        for media_type, count in media_summary.items():
            console.print(f"Downloaded {count} {media_type} files")

        console.print(f"Total media size: {total_media_size/1024/1024:.2f} MB")
        console.print(f"Media files saved in: {self.media_dir}")

        if self.handle_js:
            asyncio.run(self.close_browser())

    async def handle_interrupt(self, sig, loop):
        """Handle graceful shutdown on interrupt signals."""
        logger.warning("Received exit signal %s...", sig.name)
        await self.save_on_interrupt()
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for t in tasks:
            t.cancel()
        logger.info("Cancelling outstanding tasks")
        await asyncio.gather(*tasks, return_exceptions=True)
        loop.stop()

    async def save_on_interrupt(self):
        """Save current progress on interruption (e.g. Ctrl+C)."""
        if self.resume_file:
            self.save_results()


# Interactive Menu using InquirerPy
def interactive_menu() -> Dict[str, Any]:
    keyboard_help = """
[bold yellow]Keyboard Navigation:[/bold yellow]
↑/↓ - Move up/down
SPACE - Select option
ENTER - Confirm selection
ESC - Go back/Cancel
    """
    console.print(Panel(keyboard_help, title="[bold cyan]Keyboard Controls[/bold cyan]"))

    # Step 1: Enter URLs
    console.print("\n[bold cyan]Step 1:[/bold cyan] Enter URLs (one or multiple)")
    console.print("[dim]Tip: For multiple URLs, separate them with commas[/dim]")
    urls_input = Prompt.ask("URLs to scrape")
    urls = [url.strip() for url in urls_input.split(',') if url.strip()]
    if not urls:
        console.print("[bold red]No valid URLs provided. Exiting.[/bold red]")
        sys.exit(1)

    # Step 2: Scraping Methods
    console.print("\n[bold cyan]Step 2:[/bold cyan] Select Scraping Methods")
    console.print("[dim]Navigate with ↑/↓, select with SPACE, confirm with ENTER[/dim]")
    scraping_methods = inquirer.checkbox(
        message="Select scraping methods:",
        choices=[
            {"name": "metadata - Basic page information", "value": "metadata", "enabled": True},
            {"name": "links - All page links", "value": "links"},
            {"name": "images - All page images", "value": "images"},
            {"name": "all - Everything available", "value": "all"},
        ],
        default=["metadata"],
        instruction="(SPACE to select/unselect, ENTER to confirm)",
        transformer=lambda result: f"Selected: {', '.join(result) if result else 'metadata (default)'}"
    ).execute()

    if not scraping_methods:
        scraping_methods = ["metadata"]
        console.print("[yellow]Using default: metadata[/yellow]")

    # Step 3: Depth
    console.print("\n[bold cyan]Step 3:[/bold cyan] Configure Scraping Depth")
    recursive = Confirm.ask("Would you like to scrape linked pages recursively?", default=False)
    if recursive:
        max_depth = int(Prompt.ask("Maximum depth to scrape", default="2", show_default=True))
        subdomains = Confirm.ask("Include subdomains in recursive scraping?", default=False)
    else:
        max_depth = 1
        subdomains = False

    # Step 4: Advanced Settings
    console.print("\n[bold cyan]Step 4:[/bold cyan] Configure Advanced Settings")
    console.print("[dim]Press ENTER to use defaults, or 'y' to customize[/dim]")
    edit_advanced = Confirm.ask("Edit advanced settings?", default=False)

    if edit_advanced:
        console.print("\n[bold cyan]Advanced Configuration:[/bold cyan]")
        proxy = Prompt.ask("Proxy server (e.g., http://proxy:port) (ENTER to skip)", default="", show_default=False)
        concurrency = Prompt.ask("Number of concurrent requests", default="5", show_default=True)
        delay = Prompt.ask("Delay between requests (seconds)", default="1.0", show_default=True)
        output_file = Prompt.ask("Output file path", default="results.json", show_default=True)
        output_format = inquirer.select(
            message="Output format:",
            choices=[
                {"name": "JSON - Standard format", "value": "json"},
                {"name": "CSV - Spreadsheet compatible", "value": "csv"}
            ],
            default="json"
        ).execute()

        extraction_method = inquirer.select(
            message="Extraction method:",
            choices=[
                {"name": "BeautifulSoup - Recommended", "value": "beautifulsoup"},
                {"name": "Regex - Fallback method", "value": "regex"}
            ],
            default="beautifulsoup"
        ).execute()

        user_agent_choice = inquirer.select(
            message="User-Agent customization:",
            choices=[
                {"name": "Use default user-agents", "value": "default"},
                {"name": "Specify custom user-agents", "value": "custom"}
            ],
            default="default"
        ).execute()

        if user_agent_choice == "custom":
            custom_user_agents = Prompt.ask(
                "Enter user-agent strings separated by semicolons (;)", default="", show_default=False
            )
            user_agents = [ua.strip() for ua in custom_user_agents.split(';') if ua.strip()]
            if not user_agents:
                user_agents = DEFAULT_USER_AGENTS
                console.print("[yellow]No valid user-agents provided. Using default.[/yellow]")
        else:
            user_agents = DEFAULT_USER_AGENTS

        handle_js = Confirm.ask("Handle JavaScript-rendered content using headless browsers?", default=False)
        ssl_verify = not Confirm.ask("Disable SSL verification?", default=False)

        try:
            concurrency = int(concurrency)
            delay = float(delay)
        except ValueError:
            console.print("[bold red]Invalid input for delay or concurrency. Using defaults.[/bold red]")
            concurrency = 5
            delay = 1.0

        config = {
            'user_agents': user_agents,
            'proxy': proxy if proxy else None,
            'concurrency': concurrency,
            'delay': delay,
            'output_file': output_file,
            'output_format': output_format,
            'alternative_method': extraction_method,
            'handle_js': handle_js,
            'ssl_verify': ssl_verify
        }
    else:
        # Use default settings
        proxy = None
        concurrency = 5
        delay = 1.0
        output_file = "results.json"
        output_format = "json"
        extraction_method = "beautifulsoup"
        user_agents = DEFAULT_USER_AGENTS
        handle_js = False
        ssl_verify = True

        config = {
            'user_agents': user_agents,
            'proxy': proxy,
            'concurrency': concurrency,
            'delay': delay,
            'output_file': output_file,
            'output_format': output_format,
            'alternative_method': extraction_method,
            'handle_js': handle_js,
            'ssl_verify': ssl_verify
        }

    return {
        'urls': urls,
        'mode': [method.lower() for method in scraping_methods],
        'recursive': recursive,
        'max_depth': max_depth,
        'subdomains': subdomains,
        **config
    }


def load_config_file(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            logger.info("Loaded configuration from %s", file_path)
            return config
    except Exception as e:
        logger.error("Failed to load configuration file: %s", e)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Advanced and Professional Web Scraper",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input', '-i', help="Input file containing URLs (one per line)")
    parser.add_argument('--output', '-o', help="Output file to save the scraped data")
    parser.add_argument('--format', '-f', choices=['json', 'csv'], default='json', help="Output format (default: json)")
    parser.add_argument('--delay', '-d', type=float, default=1.0, help="Delay between requests in seconds (default: 1.0)")
    parser.add_argument('--proxy', '-p', help="Proxy server to use (e.g., http://proxy:port)")
    parser.add_argument('--mode', '-m', choices=['metadata', 'links', 'images', 'all'], help="Scraping mode")
    parser.add_argument('--concurrency', '-n', type=int, default=5, help="Number of concurrent requests (default: 5)")
    parser.add_argument('--alternative', '-a', choices=['beautifulsoup', 'regex'], help="Extraction method")
    parser.add_argument('--recursive', '-r', action='store_true', help="Recursively scrape linked pages")
    parser.add_argument('--depth', type=int, default=1, help="Maximum depth for recursive scraping")
    parser.add_argument('--subdomains', '-s', action='store_true', help="Include subdomains in recursive scraping")
    parser.add_argument('--config', '-C', help="Path to YAML configuration file")
    parser.add_argument('--resume', '-R', help="Path to resume file to recover from interruptions")
    parser.add_argument('--cli', action='store_true', help="Force CLI mode instead of interactive")

    args = parser.parse_args()

    # Check if config file is provided
    if args.config:
        config = load_config_file(args.config)
    # Check if CLI mode is forced and all required arguments are provided
    elif args.cli and args.input and args.output:
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip()]
                if not urls:
                    logger.error("Input file is empty or contains no valid URLs.")
                    sys.exit(1)
        except Exception as e:
            logger.error("Failed to read input file: %s", e)
            sys.exit(1)

        config = {
            'urls': urls,
            'mode': [args.mode.lower()] if args.mode else ['metadata'],
            'recursive': args.recursive,
            'max_depth': args.depth,
            'subdomains': args.subdomains,
            'output_file': args.output,
            'output_format': args.format.lower(),
            'concurrency': args.concurrency,
            'delay': args.delay,
            'proxy': args.proxy,
            'alternative_method': args.alternative.lower() if args.alternative else 'beautifulsoup',
            'user_agents': DEFAULT_USER_AGENTS,
            'handle_js': False,
            'ssl_verify': True
        }
    else:
        # Default to interactive mode
        console.print("[green]Starting in interactive mode...[/green]")
        config = interactive_menu()

    # Initialize Scraper
    resume_file = args.resume if args.resume else None
    scraper = WebScraper(config=config, resume_file=resume_file)

    # Display Summary Table
    summary_table = Table(title="Scraper Configuration")
    summary_table.add_column("Parameter", style="cyan", no_wrap=True)
    summary_table.add_column("Value", style="magenta")
    for key, value in config.items():
        # Format user_agents for a cleaner display
        if key == 'user_agents' and isinstance(value, list):
            value = '\n'.join(value)
        summary_table.add_row(key.replace('_', ' ').title(), str(value))
    console.print(summary_table)

    # Confirm to proceed
    if not Confirm.ask("Proceed with the above configuration?"):
        console.print("[bold yellow]Operation cancelled by the user.[/bold yellow]")
        sys.exit(0)

    try:
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True
        ) as progress:
            asyncio.run(scraper.scrape(progress))
    except KeyboardInterrupt:
        logger.warning("Scraping interrupted by user. Saving progress...")
        asyncio.run(scraper.save_on_interrupt())
        console.print("[bold yellow]Progress saved. Exiting gracefully.[/bold yellow]")
        sys.exit(0)
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
        sys.exit(1)

    # Save Results
    scraper.save_results()

    # Display Results Summary
    success_count = len([r for r in scraper.results if not r.get('error')])
    error_count = len(scraper.results) - success_count
    summary = Table(title="Scraping Summary")
    summary.add_column("Total URLs", justify="right")
    summary.add_column("Successful", justify="right", style="green")
    summary.add_column("Failed", justify="right", style="red")
    summary.add_row(str(len(scraper.urls)), str(success_count), str(error_count))
    console.print(summary)

    # Optionally show error details
    if error_count > 0:
        if Confirm.ask("Would you like to view detailed error logs?"):
            error_table = Table(title="Error Details")
            error_table.add_column("URL", style="red")
            error_table.add_column("Error", style="yellow")
            for result in scraper.results:
                if result.get('error'):
                    error_table.add_row(result['url'], result['error'])
            console.print(error_table)


if __name__ == '__main__':
    main()
