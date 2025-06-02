import requests
from bs4 import BeautifulSoup

def extract_faculty_profile(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Try to find the faculty profile section
    profile_section = soup.find("div", class_="faculty-profile")

    if profile_section:
        # Extract clean text from profile section
        text = profile_section.get_text(separator=" ", strip=True)
    else:
        # Fallback to full text
        text = soup.get_text(separator=" ", strip=True)

    return text

# Example usage
url = "https://departments.nitj.ac.in/dept/cse/Faculty/6430446e38bff038a7808a14"
text = extract_faculty_profile(url)
print(f"Extracted Text (first 500 chars):\n{text[:500]}")
