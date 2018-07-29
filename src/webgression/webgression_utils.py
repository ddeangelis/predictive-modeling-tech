'''
    File name: webgression_utils.py
    Author: Tyche Analytics Co.
'''
from bs4 import BeautifulSoup

def text_from_html(html):
    """extract visible text from raw html string"""
    soup = BeautifulSoup(html)
    [s.extract() for s in
     soup(['style', 'script', '[document]', 'head', 'title'])]
    visible_text = soup.getText()
    return visible_text
