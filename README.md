# JMScraper

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)
![Version](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Interactive Mode (Default)](#interactive-mode-default)
  - [Advanced Mode with Command-Line Arguments](#advanced-mode-with-command-line-arguments)
- [Configuration](#configuration)
- [Logging](#logging)
- [Media Downloading](#media-downloading)
- [Respectful Scraping](#respectful-scraping)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview
**JMScraper** is a powerful and ethical web scraping tool designed to efficiently extract data from multiple web pages. Whether you're a beginner needing basic metadata or an advanced user requiring comprehensive data extraction with media downloads, JMScraper caters to all your scraping needs with ease and reliability.

## Features

- **Interactive Sequential Prompts**: Guided prompts to configure scraping settings one at a time for a seamless user experience.
- **Optional Command-Line Arguments**: Advanced users can bypass interactive prompts by providing command-line arguments for more control.
- **Asynchronous Requests**: Utilizes `aiohttp` and `asyncio` to handle multiple requests concurrently, enhancing performance.
- **Real-Time Progress Bar**: Dynamic loading bar displays scraping progress in real-time without cluttering the terminal.
- **Robust Fallback Methods**: Ensures data retrieval even if the primary extraction method fails by implementing fallback strategies using regex.
- **Comprehensive Logging**: Detailed logs are maintained both in the terminal and in a log file (`scraper.log`) using the `Rich` library.
- **Flexible Output Formats**: Supports both JSON and CSV formats for saving scraped data.
- **Media Downloading**: Automatically downloads and organizes media files (images, videos, documents) into structured directories.
- **Respectful Scraping Practices**: Fetches and includes `robots.txt` content for each domain to adhere to website scraping policies.
- **User-Friendly Interface**: Enhanced terminal outputs using `Rich` and interactive prompts with `InquirerPy` for an intuitive experience.

## Prerequisites

- **Python 3.7+**: Ensure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/jmitander/JMScraper.git
   cd JMScraper
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *If you don't have a `requirements.txt`, you can install the necessary packages directly:*

   ```bash
   pip install aiohttp beautifulsoup4 rich InquirerPy aiofiles
   ```

## Usage

### Interactive Mode (Default)

When you run the scraper without any command-line arguments, it launches an interactive terminal menu that guides you through the scraping process step-by-step.

```bash
python scraper.py
```

**Interactive Steps:**

1. **Enter URL(s)**: Provide one or multiple URLs separated by commas.

   ```
   ? Enter the URL(s) you want to scrape (separated by commas): https://www.example.com, https://www.python.org
   ```

2. **Choose Scraping Method(s)**: Select from Metadata, Links, Images, or All Data.

   ```
   ? Choose scraping method(s): [Metadata, Links, Images, All Data]
   [✔] Metadata
   [✔] Links
   ```

3. **Configure Advanced Settings** (Optional):

   ```
   ? Do you want to edit advanced settings? Yes
   ```

   - **Proxy Server**: Enter a proxy URL or leave blank.
   - **Concurrency**: Define the number of concurrent requests.
   - **Delay**: Set the delay between requests in seconds.
   - **Output File**: Specify the path for the output file.
   - **Output Format**: Choose between JSON or CSV.
   - **Extraction Method**: Select between BeautifulSoup and Regex.

4. **Confirmation**: Review the configuration summary and confirm to proceed.

   ```
   Scraper Configuration
   =====================
   
   ╭─────────────────────┬────────────────────────────╮
   │ Parameter           │ Value                      │
   ╞═════════════════════╪════════════════════════════╡
   │ Input URLs          │ https://www.example.com,   │
   │                     │ https://www.python.org      │
   ├─────────────────────┼────────────────────────────┤
   │ Output File         │ results.json               │
   ├─────────────────────┼────────────────────────────┤
   │ Output Format       │ json                       │
   ├─────────────────────┼────────────────────────────┤
   │ Scraping Mode       │ metadata, links            │
   ├─────────────────────┼────────────────────────────┤
   │ Extraction Method   │ beautifulsoup              │
   ├─────────────────────┼────────────────────────────┤
   │ Delay (s)           │ 1.0                        │
   ├─────────────────────┼────────────────────────────┤
   │ Concurrency         │ 5                          │
   ├─────────────────────┼────────────────────────────┤
   │ Proxy               │ None                       │
   ╰─────────────────────┴────────────────────────────╯
   
   ? Proceed with the above configuration? Yes
   ```

### Advanced Mode with Command-Line Arguments

Advanced users can bypass the interactive menu by providing command-line arguments to customize scraping parameters directly.

**Example Command:**

```bash
python scraper.py --input urls.txt --output results.csv --format csv --mode all --concurrency 10 --delay 0.5 --proxy http://proxy:port
```

**Arguments:**

- `--input`, `-i`: Path to the input file containing URLs (one per line).
- `--output`, `-o`: Path for the output file.
- `--format`, `-f`: Output format (`json` or `csv`). Default is `json`.
- `--delay`, `-d`: Delay between requests in seconds. Default is `1.0`.
- `--proxy`, `-p`: Proxy server to use (e.g., `http://proxy:port`).
- `--mode`, `-m`: Scraping mode (`metadata`, `links`, `images`, or `all`).
- `--concurrency`, `-c`: Number of concurrent requests. Default is `5`.
- `--alternative`, `-a`: Extraction method (`beautifulsoup` or `regex`).

**Example Command with Partial Arguments:**

```bash
python scraper.py -i urls.txt -o results.csv -f csv -m links
```

## Configuration

### Interactive Prompts

The interactive mode guides users through a series of prompts to configure:

1. **URL(s)**: Enter one or multiple URLs separated by commas.
2. **Scraping Method(s)**: Select one or more methods (Metadata, Links, Images, All Data).
3. **Advanced Settings** (Optional):
   - **Proxy Server**: Enter a proxy URL or leave blank.
   - **Concurrency**: Number of concurrent requests.
   - **Delay**: Delay between requests in seconds.
   - **Output File**: Path for the output file.
   - **Output Format**: Choose between JSON or CSV.
   - **Extraction Method**: Select between BeautifulSoup and Regex.

### Command-Line Arguments

Advanced users can specify configurations directly using command-line arguments for more control and automation.

```bash
python scraper.py --input urls.txt --output results.csv --format csv --mode all --concurrency 10 --delay 0.5 --proxy http://proxy:port
```

## Logging

The scraper maintains detailed logs both in the terminal and in a log file named `scraper.log` using the `Rich` library. This aids in monitoring the scraping process and debugging if necessary.

## Media Downloading

JMScraper automatically downloads and organizes media files (images, videos, documents) into structured directories:

- **Images**: Saved in `media/images/`
- **Videos**: Saved in `media/videos/`
- **Documents**: Saved in `media/documents/`

Each media file is saved with a unique, safe filename generated using a hash of the URL and appropriate file extensions based on content type.

## Respectful Scraping

JMScraper adheres to ethical scraping practices by:

- **Fetching `robots.txt`**: Retrieves and includes the contents of `robots.txt` for each domain to respect website scraping policies.
- **User-Agent Rotation**: Rotates through a list of common User-Agent strings to mimic different browsers and reduce the risk of being blocked.
- **Rate Limiting**: Implements delay and concurrency controls to avoid overwhelming target servers.

**Important**: Always ensure you have permission to scrape the websites you target and comply with their `robots.txt` and terms of service.

## Contributing

Contributions are welcome! If you'd like to contribute to JMScraper, please follow these steps:

1. **Fork the Repository**

2. **Create a New Branch**

   ```bash
   git checkout -b feature/YourFeatureName
   ```

3. **Make Your Changes**

4. **Commit Your Changes**

   ```bash
   git commit -m "Add your message here"
   ```

5. **Push to the Branch**

   ```bash
   git push origin feature/YourFeatureName
   ```

6. **Open a Pull Request**

Please ensure your contributions adhere to the existing code style and include appropriate documentation and tests where necessary.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions or support, please open an issue in the [GitHub repository](https://github.com/jmitander/JMScraper/issues).

---

**Happy Scraping!**
