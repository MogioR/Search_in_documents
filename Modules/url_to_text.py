import re
import requests
from bs4 import BeautifulSoup


def url_to_text(url: str) -> str:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) '
                             'Chrome/41.0.2228.0 Safari/537.3'}
    response = requests.get(url, headers=headers, timeout=40, verify=False)

    # Del unicode trash
    response_text = response.text.encode('utf-8')

    # Del script and style tags
    soup = BeautifulSoup(response_text)
    for script in soup(["script", "style"]):
        script.extract()

    # Del unicode trash again
    clean_html = str(soup.body.get_text())
    clean_html = re.sub("(<!--.*?-->)", "", clean_html, flags=re.DOTALL)
    # clean_html = clean_html.replace('\u20bd', ' ')

    # Del html tags
    re_clean = re.compile('<.*?>')
    clean_text = re.sub(re_clean, ' ', clean_html)

    # Del unicode trash
    clean_text = ''.join([char for char in clean_text if ord(char) < 2048])

    # Normalise text
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in clean_text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    clean_text = ' '.join(chunk for chunk in chunks if chunk)

    return clean_text

