import requests
from bs4 import BeautifulSoup
import os
import zipfile

BASE_URL = 'http://garner.ucsd.edu/pub/measuresESESES_products/Timeseries/CurrentUntarred/Raw_M_TrendNeuTimeSeries_comb_20240922/'
OUTPUT_DIR = 'Data/'

def get_links(url, extensions=('.Z', '.zip')):
    """
    Scrapes a webpage and retrieves all file links matching the specified extensions.
    """
    print(f"Scraping {url} for files with extensions {extensions}")
    resp = requests.get(url)
    resp.raise_for_status()
    
    sp = BeautifulSoup(resp.content, 'html.parser')
    
    links = []
    for link in sp.find_all('a', href=True):
        href = link['href']
        if href.endswith(extensions):
            links.append(href)
    
    files = [url + href for href in links]
    print(f"Found {len(files)} files.")
    
    return files

def download_file(url, file):
    """
    Downloads a file from a URL and saves it to a local path.
    """
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
    return file

def extract_zip_file(file, output):
    """
    Extracts a .zip file and saves the contents to the specified folder.
    """
    with zipfile.ZipFile(file, 'r') as zip:
        zip.extractall(output)

def process_files(base_url, output):
    """
    Extract all .zip and .Z files from a website and save them in the output folder.
    """
    file_urls = get_links(base_url)

    if not os.path.exists(output):
        os.makedirs(output)

    for url in file_urls:
        file = url.split('/')[-1] 

        download_file(url, file)
        
        extract_zip_file(file, output)

        os.remove(file)

def main():
    process_files(BASE_URL, OUTPUT_DIR)

if __name__ == '__main__':
    main()