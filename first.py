import os
import csv
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_and_save_html(index, url, label, output_folder='Alloutput'):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Save raw HTML
        os.makedirs(output_folder, exist_ok=True)
        filename = f"{index}_{label}.txt"
        filepath = os.path.join(output_folder, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"URL: {url}\n\n")
            f.write("=== RAW HTML CONTENT ===\n\n")
            f.write(response.text)

        print(f"✅ Saved: {filename}")
    except Exception as e:
        print(f"❌ Failed: {url} -> {e}")

def main(csv_file, max_threads=500):
    tasks = []

    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if len(row) >= 2:
                url, label = row[0], row[1]
                tasks.append((idx, url, label))

    print(f"Processing {len(tasks)} URLs with {max_threads} threads...\n")

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(fetch_and_save_html, idx, url, label) for idx, url, label in tasks]
        for _ in as_completed(futures):
            pass  # Wait for all threads to finish


main('TrainingSet.csv', max_threads=500)