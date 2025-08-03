import os
import asyncio
from playwright.async_api import async_playwright
import pandas as pd
from PIL import Image
from io import BytesIO

# Function to download the images that meet the size condition
async def download_images_from_url(url, folder_path):
    # Launch Playwright and navigate to the URL
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            # Increase timeout to 60 seconds
            await page.goto(url, timeout=5000)  # Increase timeout to 60 seconds
            
            # Wait for the body to load
            await page.wait_for_selector('body', timeout=5000)
            
            # Get all image elements on the page
            images = await page.query_selector_all('img')

            # Iterate over images and take screenshots if they meet the size condition
            for index, image in enumerate(images):
                bbox = await image.bounding_box()
                if bbox and bbox['width'] >= 200 and bbox['height'] >= 200:
                    # Screenshot of the image
                    img_bytes = await image.screenshot()

                    # Convert to Pillow image for size check
                    img = Image.open(BytesIO(img_bytes))
                    if img.width >= 200 and img.height >= 200:
                        # Save image to the folder
                        image_filename = f"{folder_path}/image_{index}.png"
                        img.save(image_filename)
                        print(f"Saved image: {image_filename}")

        except Exception as e:
            # If there is an error (like timeout), log it without printing it out
            print()

        # Close the browser after processing
        await browser.close()

# Function to process URLs from the Excel file and take screenshots
def process_urls_from_xlsx(file_path, save_folder):
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Iterate over each URL in the Excel file
    for idx, row in df.iterrows():
        url = row['URL']  # Assuming the column with URLs is named 'URL'
        
        # Create a folder for each URL with a format: image_index
        folder_name = f"{save_folder}/images_{idx}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Download images from the URL and save them in the folder
        asyncio.run(download_images_from_url(url, folder_name))

# Usage
if _name_ == "_main_":
    images_folder = "images"  # Folder where images will be stored
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    
    xlsx_file = "TrainingSet.xlsx"  # Excel file containing the URLs
    process_urls_from_xlsx(xlsx_file, images_folder)
