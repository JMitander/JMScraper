#!/usr/bin/env python3
import argparse
import asyncio
import aiohttp
import os
import re
import json
import csv
import sys
import logging
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from aiohttp import ClientSession, ClientTimeout
from typing import List, Dict, Any, Optional, Set
import mimetypes
import hashlib
from pathlib import Path
import aiofiles
import tld

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.logging import RichHandler
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from InquirerPy import inquirer
from rich.panel import Panel

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
[green]Version:[/green] 1.0.0
"""

console.print(Panel(
    welcome_message,
    title="[bold red]JMScraper[/bold red]",
    subtitle="[italic]By JM[/italic]",
    border_style="bold blue",
    padding=(1, 2),
    expand=False
))

# Configure Logging with RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("WebScraper")

# Constants
USER_AGENTS = [
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

# Fallback extraction using regex if BeautifulSoup fails
def extract_meta_fallback(text: str, name: str) -> str:
    meta_regex = re.compile(
        rf'<meta\s+name=["\']{name}["\']\s+content=["\']([^"\']+)["\']', re.IGNORECASE
    )
    match = meta_regex.search(text)
    return match.group(1).strip() if match else 'N/A'

# Scraper Class with Fallback Methods
class WebScraper:
    def __init__(
        self,
        urls: List[str],
        output_file: str,
        output_format: str = 'json',
        delay: float = 1.0,
        proxy: Optional[str] = None,
        mode: str = 'metadata',
        concurrent_requests: int = 5,
        alternative_method: str = 'beautifulsoup',
        max_retries: int = 3,
        recursive: bool = False,  # New parameter
        max_depth: int = 1,      # New parameter
        subdomains: bool = False  # New parameter
    ):
        self.urls = urls
        self.output_file = output_file
        self.output_format = output_format
        self.delay = delay
        self.proxy = proxy
        self.mode = mode
        self.concurrent_requests = concurrent_requests
        self.alternative_method = alternative_method
        self.max_retries = max_retries
        self.results: List[Dict[str, Any]] = []
        self._session_counter = 0
        self.base_output_dir = Path(output_file).parent
        self.media_dir = self.base_output_dir / "media"
        self.images_dir = self.media_dir / "images"
        self.videos_dir = self.media_dir / "videos"
        self.documents_dir = self.media_dir / "documents"
        self._create_directories()
        self.recursive = recursive
        self.max_depth = max_depth
        self.subdomains = subdomains
        self.visited_urls: Set[str] = set()
        self.domain_whitelist: Set[str] = set()
        self._setup_domain_whitelist()

    def _create_directories(self):
        """Create necessary directories for media storage"""
        for directory in [self.media_dir, self.images_dir, self.videos_dir, self.documents_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def _setup_domain_whitelist(self):
        """Setup allowed domains based on input URLs"""
        for url in self.urls:
            try:
                domain = tld.get_fld(url, fix_protocol=True)
                self.domain_whitelist.add(domain)
            except Exception:
                continue

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL should be scraped based on settings"""
        try:
            parsed = urlparse(url)
            domain = tld.get_fld(url, fix_protocol=True)
            
            # Check if domain is allowed
            if not self.subdomains and domain not in self.domain_whitelist:
                return False
                
            # If subdomains allowed, check if parent domain matches
            if self.subdomains:
                if not any(domain.endswith(d) for d in self.domain_whitelist):
                    return False

            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False

    def _get_headers(self):
        # Rotate user agents and add random properties
        self._session_counter = (self._session_counter + 1) % len(USER_AGENTS)
        headers = DEFAULT_HEADERS.copy()
        headers['User-Agent'] = USER_AGENTS[self._session_counter]
        return headers

    def _get_safe_filename(self, url: str, content_type: str = None) -> str:
        """Generate safe filename from URL and content type"""
        # Create hash of URL for unique filename
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        
        # Get extension from content-type or URL
        ext = mimetypes.guess_extension(content_type) if content_type else os.path.splitext(url)[1]
        if not ext:
            ext = '.bin'  # Fallback extension
        
        # Create safe filename
        return f"{url_hash}{ext}"

    async def _download_media(self, session: ClientSession, url: str, content_type: str = None) -> Optional[Dict[str, str]]:
        """Download media file and return metadata"""
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    return None

                content_type = response.headers.get('content-type', content_type)
                if not content_type:
                    return None

                # Determine media type and directory
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

                # Download file
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
            logger.warning(f"Failed to download {url}: {e}")
            return None

    async def extract_media(self, session: ClientSession, soup: BeautifulSoup, base_url: str) -> Dict[str, List[Dict[str, str]]]:
        """Extract and download all media from the page"""
        media_data = {
            'images': [],
            'videos': [],
            'documents': []
        }

        # Extract and download images
        images = [urljoin(base_url, img['src']) for img in soup.find_all('img', src=True)]
        for url in images:
            if result := await self._download_media(session, url):
                media_data['images'].append(result)

        # Extract and download videos
        videos = [
            urljoin(base_url, vid['src']) 
            for vid in soup.find_all(['video', 'source'], src=True)
        ]
        for url in videos:
            if result := await self._download_media(session, url):
                media_data['videos'].append(result)

        # Extract and download linked documents (pdf, doc, etc.)
        documents = [
            urljoin(base_url, a['href'])
            for a in soup.find_all('a', href=True)
            if any(ext in a['href'].lower() for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx'])
        ]
        for url in documents:
            if result := await self._download_media(session, url):
                media_data['documents'].append(result)

        return media_data

    async def fetch(self, session: ClientSession, url: str) -> Dict[str, Any]:
        logger.info(f"[bold cyan]Fetching:[/bold cyan] {url}")
        data = {'url': url}
        
        for attempt in range(self.max_retries):
            try:
                session.headers.update(self._get_headers())
                
                # Add status message
                console.print(f"[dim]Connecting to {url}...[/dim]")
                
                async with session.get(
                    url,
                    timeout=ClientTimeout(total=15),
                    allow_redirects=True,
                    ssl=False
                ) as response:
                    response.raise_for_status()
                    console.print(f"[green]✓ Connected to {url} (Status: {response.status})[/green]")
                    
                    console.print("[dim]Reading page content...[/dim]")
                    text = await response.text()
                    soup = BeautifulSoup(text, 'html.parser')

                    # Show what we're extracting
                    console.print(f"[dim]Extracting {self.mode} data...[/dim]")
                    
                    if self.mode == 'metadata':
                        metadata = self.extract_metadata(soup, text)
                        data.update(metadata)
                    elif self.mode == 'links':
                        links = self.extract_links(soup, url)
                        data.update(links)
                    elif self.mode == 'images':
                        images = self.extract_images(soup, url)
                        data.update(images)
                    elif self.mode == 'all':
                        all_data = self.extract_all(soup, text, url)
                        data.update(all_data)
                    else:
                        logger.warning(f"Unknown mode: {self.mode}")

                    data.update({
                        'emails': extract_emails(text),
                        'favicon': self.extract_favicon(soup, url)
                    })
                    
                    # After extracting regular data, download media if needed
                    if self.mode in ['all', 'images']:
                        media_data = await self.extract_media(session, soup, url)
                        data.update(media_data)
                    
                    console.print(f"[green]✓ Successfully extracted data from {url}[/green]")
                    console.print(f"[green]✓ Downloaded {len(data.get('images', []))} images, "
                                 f"{len(data.get('videos', []))} videos, "
                                 f"{len(data.get('documents', []))} documents[/green]")
                    break  # Success, exit retry loop
                    
            except aiohttp.ClientResponseError as e:
                if e.status == 429:  # Too Many Requests
                    wait_time = 2 ** attempt * self.delay
                    logger.warning(f"Rate limit hit, waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                    continue
                data['error'] = f"HTTP {e.status}: {str(e)}"
                break
            except Exception as e:
                data['error'] = str(e)
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt * self.delay
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                break

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
            logger.debug(f"Extracted metadata: {metadata}")
        except Exception as e:
            logger.warning(f"Metadata extraction failed, attempting fallback: {e}")
            metadata = {
                'title': self.extract_title_fallback(text),
                'description': extract_meta_fallback(text, 'description'),
                'keywords': extract_meta_fallback(text, 'keywords'),
                'h1_tags': self.extract_h1_fallback(text)
            }
            logger.debug(f"Extracted metadata with fallback: {metadata}")
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
            links = [urljoin(base_url, a['href']) for a in soup.find_all('a', href=True)]
            logger.debug(f"Extracted {len(links)} links")
            return {'links': links}
        except Exception as e:
            logger.warning(f"Links extraction failed, attempting fallback: {e}")
            links = self.extract_links_fallback(soup.prettify(), base_url)
            return {'links': links}

    def extract_links_fallback(self, text: str, base_url: str) -> List[str]:
        link_regex = re.compile(r'href=["\'](.*?)["\']', re.IGNORECASE)
        raw_links = link_regex.findall(text)
        return [urljoin(base_url, link) for link in raw_links]

    def extract_images(self, soup: BeautifulSoup, base_url: str) -> Dict[str, Any]:
        try:
            images = [urljoin(base_url, img['src']) for img in soup.find_all('img', src=True)]
            logger.debug(f"Extracted {len(images)} images")
            return {'images': images}
        except Exception as e:
            logger.warning(f"Images extraction failed, attempting fallback: {e}")
            images = self.extract_images_fallback(soup.prettify(), base_url)
            return {'images': images}

    def extract_images_fallback(self, text: str, base_url: str) -> List[str]:
        img_regex = re.compile(r'src=["\'](.*?)["\']', re.IGNORECASE)
        raw_images = img_regex.findall(text)
        return [urljoin(base_url, img) for img in raw_images]

    def extract_favicon(self, soup: BeautifulSoup, base_url: str) -> str:
        try:
            favicon = soup.find('link', rel=lambda x: x and 'icon' in x.lower())
            favicon_url = urljoin(base_url, favicon['href']) if favicon and 'href' in favicon.attrs else 'N/A'
            logger.debug(f"Extracted favicon: {favicon_url}")
            return favicon_url
        except Exception as e:
            logger.warning(f"Favicon extraction failed, attempting fallback: {e}")
            return self.extract_favicon_fallback(base_url)

    def extract_favicon_fallback(self, base_url: str) -> str:
        # Common favicon locations
        potential_favicons = ['/favicon.ico']
        for path in potential_favicons:
            favicon_url = urljoin(base_url, path)
            # Here you could implement a check to see if the favicon exists
            return favicon_url
        return 'N/A'

    def extract_all(self, soup: BeautifulSoup, text: str, base_url: str) -> Dict[str, Any]:
        data = {}
        data.update(self.extract_metadata(soup, text))
        data.update(self.extract_links(soup, base_url))
        data.update(self.extract_images(soup, base_url))
        return data

    async def scrape(self, progress: Progress):
        connector = aiohttp.TCPConnector(
            limit_per_host=self.concurrent_requests,
            force_close=True,
            enable_cleanup_closed=True,
            ssl=False
        )
        
        timeout = ClientTimeout(total=20)
        
        # Configure session with proxy if provided
        session_kwargs = {
            'connector': connector,
            'timeout': timeout,
            'trust_env': True  # Allow environment HTTP/HTTPS proxy settings
        }
        if self.proxy:
            session_kwargs['proxy'] = self.proxy

        async with aiohttp.ClientSession(**session_kwargs) as session:
            if self.recursive:
                urls_to_scrape = set(self.urls)
                current_depth = 0
                
                while urls_to_scrape and current_depth < self.max_depth:
                    console.print(f"\n[cyan]Scraping depth {current_depth + 1}/{self.max_depth}[/cyan]")
                    new_urls = set()
                    
                    # Create progress bar for current depth
                    task_id = progress.add_task(
                        f"[cyan]Depth {current_depth + 1}[/cyan]",
                        total=len(urls_to_scrape)
                    )
                    
                    # Scrape current level
                    tasks = []
                    for url in urls_to_scrape:
                        if url not in self.visited_urls:
                            tasks.append(self.bound_fetch(session, url, progress, task_id))
                            self.visited_urls.add(url)
                    
                    results = await asyncio.gather(*tasks)
                    
                    # Collect new URLs from results
                    for result in results:
                        if 'links' in result:
                            new_urls.update([
                                link for link in result['links']
                                if self._is_valid_url(link) and link not in self.visited_urls
                            ])
                    
                    urls_to_scrape = new_urls
                    current_depth += 1
            else:
                # Original non-recursive scraping
                task_id = progress.add_task("[cyan]Scraping...", total=len(self.urls))
                tasks = [self.bound_fetch(session, url, progress, task_id) for url in self.urls]
                await asyncio.gather(*tasks)

    async def bound_fetch(self, session: ClientSession, url: str, progress: Progress, task_id: int):
        async with asyncio.Semaphore(self.concurrent_requests):
            result = await self.fetch(session, url)
            self.results.append(result)
            progress.advance(task_id, 1)
            await asyncio.sleep(self.delay)

    def save_results(self):
        console.print("\n[bold cyan]Saving Results[/bold cyan]")
        try:
            if self.output_format.lower() == 'json':
                console.print(f"[dim]Writing data to {self.output_file}...[/dim]")
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f, indent=4, ensure_ascii=False)
                
                # Show a mini summary of what was saved
                total_size = os.path.getsize(self.output_file)
                console.print(f"[green]✓ Saved {len(self.results)} items ({total_size/1024:.1f} KB) to {self.output_file}[/green]")
                
            elif self.output_format.lower() == 'csv':
                if not self.results:
                    logger.warning("No data to write to CSV.")
                    return
                    
                console.print(f"[dim]Writing data to {self.output_file}...[/dim]")
                keys = set()
                for entry in self.results:
                    keys.update(entry.keys())
                keys = sorted(keys)
                
                with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(self.results)
                
                total_size = os.path.getsize(self.output_file)
                console.print(f"[green]✓ Saved {len(self.results)} rows ({total_size/1024:.1f} KB) to {self.output_file}[/green]")
            
            else:
                logger.error(f"Unsupported format: {self.output_format}")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Failed to save data: {e}")

        # Add media summary
        console.print("\n[bold cyan]Media Download Summary:[/bold cyan]")
        total_media = 0
        total_size = 0

        for result in self.results:
            for media_type in ['images', 'videos', 'documents']:
                if media_items := result.get(media_type, []):
                    total_media += len(media_items)
                    total_size += sum(item['size'] for item in media_items)

        console.print(f"[green]✓ Downloaded {total_media} media files "
                     f"({total_size/1024/1024:.1f} MB)[/green]")
        console.print(f"[dim]Media files saved in: {self.media_dir}[/dim]")

# Interactive Menu using InquirerPy
def interactive_menu() -> tuple[str, str, str, str, str, int, float, str, bool, int, bool]:
    # Display keyboard navigation help
    keyboard_help = """
[bold yellow]Keyboard Navigation:[/bold yellow]
↑/↓ - Move up/down
SPACE - Select option
ENTER - Confirm selection
ESC - Go back/Cancel
    """
    console.print(Panel(keyboard_help, title="[bold cyan]Keyboard Controls[/bold cyan]"))

    # Step 1: Enter URLs with clear instructions
    console.print("\n[bold cyan]Step 1:[/bold cyan] Enter URLs (one or multiple)")
    console.print("[dim]Tip: For multiple URLs, separate them with commas[/dim]")
    urls_input = Prompt.ask(
        "URLs to scrape"
    )
    urls = [url.strip() for url in urls_input.split(',') if url.strip()]
    if not urls:
        console.print("[bold red]No valid URLs provided. Exiting.[/bold red]")
        sys.exit(1)

    # Step 2: Choose Scraping Methods with better instructions
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

    # Step 3: Configure Scraping Depth
    console.print("\n[bold cyan]Step 3:[/bold cyan] Configure Scraping Depth")
    recursive = Confirm.ask(
        "Would you like to scrape linked pages recursively?",
        default=False
    )

    if recursive:
        max_depth = int(Prompt.ask(
            "Maximum depth to scrape",
            default="2",
            show_default=True
        ))
        subdomains = Confirm.ask(
            "Include subdomains in recursive scraping?",
            default=False
        )
    else:
        max_depth = 1
        subdomains = False

    # Step 4: Advanced Settings with better visibility
    console.print("\n[bold cyan]Step 4:[/bold cyan] Configure Advanced Settings")
    console.print("[dim]Press ENTER to use defaults, or 'y' to customize[/dim]")
    edit_advanced = Confirm.ask("Edit advanced settings?", default=False)

    if edit_advanced:
        # Advanced Settings with clear instructions
        console.print("\n[bold cyan]Advanced Configuration:[/bold cyan]")
        proxy = Prompt.ask(
            "Proxy server (ENTER to skip)",
            default="",
            show_default=False
        )
        
        concurrency = Prompt.ask(
            "Number of concurrent requests",
            default="5",
            show_default=True
        )
        
        delay = Prompt.ask(
            "Delay between requests (seconds)",
            default="1.0",
            show_default=True
        )
        
        output_file = Prompt.ask(
            "Output file path",
            default="results.json",
            show_default=True
        )
        
        console.print("[dim]Select output format (↑/↓ to navigate, ENTER to select)[/dim]")
        output_format = inquirer.select(
            message="Output format:",
            choices=[
                {"name": "JSON - Standard format", "value": "json"},
                {"name": "CSV - Spreadsheet compatible", "value": "csv"}
            ],
            default="json"
        ).execute()

        console.print("[dim]Select extraction method (↑/↓ to navigate, ENTER to select)[/dim]")
        alternative_method = inquirer.select(
            message="Extraction method:",
            choices=[
                {"name": "BeautifulSoup - Recommended", "value": "beautifulsoup"},
                {"name": "Regex - Fallback method", "value": "regex"}
            ],
            default="beautifulsoup"
        ).execute()

        try:
            concurrency = int(concurrency)
            delay = float(delay)
        except ValueError:
            console.print("[bold red]Invalid input for delay or concurrency. Using defaults.[/bold red]")
            concurrency = 5
            delay = 1.0
    else:
        # Use default settings
        proxy = ""
        concurrency = 5
        delay = 1.0
        output_file = "results.json"
        output_format = "json"
        alternative_method = "beautifulsoup"

    return (
        ','.join(scraping_methods).lower(),
        alternative_method.lower(),
        ','.join(urls),
        output_file,
        output_format.lower(),
        concurrency,
        delay,
        proxy,
        recursive,           # New return value
        max_depth,          # New return value
        subdomains         # New return value
    )

# Main Function with CLI and Interactive Menu
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
    parser.add_argument('--mode', '-m', choices=['metadata', 'links', 'images', 'all'], help="Scraping mode: metadata, links, images, or all")
    parser.add_argument('--concurrency', '-c', type=int, default=5, help="Number of concurrent requests (default: 5)")
    parser.add_argument('--alternative', '-a', choices=['beautifulsoup', 'regex'], help="Extraction method: beautifulsoup or regex")
    parser.add_argument('--recursive', '-r', action='store_true', help="Recursively scrape linked pages")
    parser.add_argument('--depth', type=int, default=1, help="Maximum depth for recursive scraping")
    parser.add_argument('--subdomains', '-s', action='store_true', help="Include subdomains in recursive scraping")
    args = parser.parse_args()

    # If no required arguments are provided, launch interactive menu
    if not all([args.input, args.output]):
        console.print("[bold green]Launching Interactive Menu...[/bold green]")
        mode, method, urls_str, output_file, output_format, concurrency, delay, proxy, recursive, max_depth, subdomains = interactive_menu()
        urls = [url.strip() for url in urls_str.split(',') if url.strip()]
    else:
        # Use CLI arguments
        mode = args.mode.lower() if args.mode else 'metadata'
        method = args.alternative.lower() if args.alternative else 'beautifulsoup'
        input_file = args.input
        output_file = args.output
        output_format = args.format.lower()
        concurrency = args.concurrency
        delay = args.delay
        proxy = args.proxy if args.proxy else ''

        # Read URLs from input file
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip()]
                if not urls:
                    logger.error("Input file is empty or contains no valid URLs.")
                    sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to read input file: {e}")
            sys.exit(1)

    if not args.input or not args.output:
        # URLs, mode, etc., already obtained from interactive_menu
        pass
    else:
        # URLs already obtained from CLI
        pass

    # Initialize Scraper
    scraper = WebScraper(
        urls=urls,
        output_file=output_file,
        output_format=output_format,
        delay=delay,
        proxy=proxy,
        mode=mode,
        concurrent_requests=concurrency,
        alternative_method=method,
        recursive=args.recursive if args.input else recursive,
        max_depth=args.depth if args.input else max_depth,
        subdomains=args.subdomains if args.input else subdomains
    )

    # Display Summary Table
    summary_table = Table(title="Scraper Configuration", style="bold blue")
    summary_table.add_column("Parameter", style="cyan", no_wrap=True)
    summary_table.add_column("Value", style="magenta")
    summary_table.add_row("Input URLs", ', '.join(scraper.urls))
    summary_table.add_row("Output File", scraper.output_file)
    summary_table.add_row("Output Format", scraper.output_format)
    summary_table.add_row("Scraping Mode", scraper.mode)
    summary_table.add_row("Extraction Method", scraper.alternative_method)
    summary_table.add_row("Delay (s)", str(scraper.delay))
    summary_table.add_row("Concurrency", str(scraper.concurrent_requests))
    summary_table.add_row("Proxy", scraper.proxy if scraper.proxy else "None")
    console.print(summary_table)

    # Confirm to proceed
    if not Confirm.ask("Proceed with the above configuration?"):
        console.print("[bold yellow]Operation cancelled by the user.[/bold yellow]")
        sys.exit(0)

    # Run Scraper with Progress Bar
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True
    ) as progress:
        try:
            asyncio.run(scraper.scrape(progress))
        except KeyboardInterrupt:
            logger.warning("Scraping interrupted by user.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            sys.exit(1)

    # Save Results
    scraper.save_results()

    # Display Results Summary
    success_count = len([r for r in scraper.results if 'error' not in r])
    error_count = len(scraper.results) - success_count
    summary = Table(title="Scraping Summary", style="bold green")

    summary.add_column("Total URLs", style="cyan")
    summary.add_column("Successful", style="green")
    summary.add_column("Failed", style="red")
    summary.add_row(str(len(scraper.urls)), str(success_count), str(error_count))
    console.print(summary)

    # Optionally, display detailed errors
    if error_count > 0:
        if Confirm.ask("Would you like to view detailed error logs?"):
            error_table = Table(title="Error Details", style="bold red")
            error_table.add_column("URL", style="cyan")
            error_table.add_column("Error", style="red")
            for result in scraper.results:
                if 'error' in result:
                    error_table.add_row(result['url'], result['error'])
            console.print(error_table)

if __name__ == '__main__':
    main()

