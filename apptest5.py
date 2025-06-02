import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin

def fetch_page_content(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch {url}: Status code {response.status_code}")
            return None
        print(f"Successfully fetched {url}")
        return response.text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_content(soup, tag, base_url, visited_urls, content_list):
    elements = soup.find_all(tag)
    print(f"Found {len(elements)} {tag} elements at {base_url}")
    for element in elements:
        text = element.get_text(strip=True)
        if text:
            content_list.append({"tag": tag, "content": text})
        else:
            print(f"No text in {tag} at {base_url}")
        if tag in ['p', 'div', 'td']:
            links = element.find_all('a', href=True)
            for link in links:
                href = link['href']
                absolute_url = urljoin(base_url, href)
                if absolute_url not in visited_urls and 'departments.nitj.ac.in' in absolute_url:
                    visited_urls.add(absolute_url)
                    print(f"Visiting {absolute_url}")
                    recursive_extract(absolute_url, visited_urls, content_list)

def recursive_extract(url, visited_urls, content_list):
    html_content = fetch_page_content(url)
    if not html_content:
        return
    
    soup = BeautifulSoup(html_content, 'html.parser')
    tags = ['div', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'table', 'td', 'tr']
    
    for tag in tags:
        extract_content(soup, tag, url, visited_urls, content_list)

def save_to_json(content_list, filename="nitj_content.json"):
    try:
        print(f"Content to save: {content_list}")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(content_list, f, ensure_ascii=False, indent=4)
        print(f"Content saved to {filename}")
    except Exception as e:
        print(f"Error saving to JSON: {e}")

def main():
    start_url = "https://departments.nitj.ac.in/dept/cse/Faculty/6430446c38bff038a780870c"
    visited_urls = set([start_url])
    content_list = []
    
    print(f"Starting extraction from {start_url}")
    recursive_extract(start_url, visited_urls, content_list)
    
    save_to_json(content_list)
    print(f"Extracted content from {len(visited_urls)} unique URLs")
    print(f"Total items extracted: {len(content_list)}")

if __name__ == "__main__":
    main()