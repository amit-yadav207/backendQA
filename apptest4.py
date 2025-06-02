from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time

def extract_content_with_selenium(url):
    options = Options()
    options.headless = True  # Run in headless mode (no browser window)
    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
        time.sleep(2)  # wait for JavaScript to render content

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Optional: target specific div or section
        content_div = soup.find('div', class_='faculty-profile-content')  # update class as per actual DOM
        if content_div:
            return content_div.get_text(separator='\n', strip=True)
        else:
            return soup.get_text(separator='\n', strip=True)

    finally:
        driver.quit()

# Example usage
url = "https://departments.nitj.ac.in/dept/cse/Faculty/6430446e38bff038a7808a14"
content = extract_content_with_selenium(url)
print(content[:1000])  # Print first 1000 characters
