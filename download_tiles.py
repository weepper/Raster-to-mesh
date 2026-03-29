import os
import re
import argparse
import requests
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def extract_links_from_file(filepath):
    """Reads a file and finds all standard HTTP/HTTPS links."""
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    url_pattern = re.compile(r'https?://[^\s<>"]+')
    return url_pattern.findall(text)

def download_file(url, download_folder):
    """Downloads a single file and returns a status message."""
    os.makedirs(download_folder, exist_ok=True)

    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)

    if 'FILENAME' in query_params:
        filename = query_params['FILENAME'][0]
    else:
        filename = os.path.basename(parsed_url.path)
        if not filename:
            filename = "downloaded_file.tif"

    filepath = os.path.join(download_folder, filename)

    if os.path.exists(filepath):
        return f"Skipped: {filename} (Already exists)"

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        with open(filepath, 'wb') as file:
            for data in response.iter_content(chunk_size=65536):
                file.write(data)

        return f"Success: {filename}"

    except requests.exceptions.RequestException as e:
        return f"Failed: {filename} - Error: {e}"

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Parallel downloader for IGN GeoTIFF files from a text file."
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        default="dalles.txt",
        help="Path to the text file containing the URLs (default: dalles.txt)"
    )

    parser.add_argument(
        "-d", "--dir",
        type=str,
        default="ign_dalles_downloads",
        help="Directory to save the downloaded .tif files (default: ign_dalles_downloads)"
    )

    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=10,
        help="Number of simultaneous parallel downloads (default: 10)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Could not find '{args.input}'. Make sure the file exists.")
    else:
        links = extract_links_from_file(args.input)
        unique_links = list(set(links))

        print(f"Found {len(unique_links)} unique link(s) in '{args.input}'.")
        print(f"Starting parallel download ({args.workers} at a time) to '{args.dir}'...\n")

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_url = {executor.submit(download_file, link, args.dir): link for link in unique_links}

            with tqdm(total=len(unique_links), desc="Total Progress", unit="file") as pbar:
                for future in as_completed(future_to_url):
                    result_message = future.result()
                    tqdm.write(result_message)
                    pbar.update(1)

        print("\nAll downloads finished!")
