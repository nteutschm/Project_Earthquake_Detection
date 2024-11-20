import requests
from bs4 import BeautifulSoup
import os
import zipfile

# If our data is no longer available, the BASE_URL should be updated to point to the latest version.
BASE_URL = 'http://garner.ucsd.edu/pub/measuresESESES_products/Timeseries/CurrentUntarred/Raw_M_TrendNeuTimeSeries_comb_20240922/'
OUTPUT_DIR = 'Data/'

def get_links(url, extensions=('.Z', '.zip')):
    """
    Scrapes a webpage and retrieves all file links matching the specified extensions.

    Parameters:
    url (str): The URL to scrape for files.
    extensions (tuple): A tuple of file extensions to filter the links by (default is ('.Z', '.zip')).

    Returns:
    list: A list of complete file URLs that match the specified extensions.
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
    
    Parameters:
    url (str): The URL of the file to download.
    file (str): The local file path where the downloaded content will be saved.

    Returns:
    str: The path of the saved file.
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
    
    Parameters:
    file (str): The path to the ZIP file to be extracted.
    output (str): The directory where the extracted files will be stored.

    Returns:
    None
    """
    
    with zipfile.ZipFile(file, 'r') as zip:
        zip.extractall(output)

def process_files(base_url, output):
    """
    Downloads and extracts files from a specified URL. The function first retrieves the list of files 
    by scraping the base URL, then downloads and extracts each file to the specified output directory. 
    
    Parameters:
    base_url (str): The base URL where the files are located and need to be scraped.
    output (str): The directory where the files will be extracted.

    Returns:
    None
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