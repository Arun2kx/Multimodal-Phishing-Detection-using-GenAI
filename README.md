# Multimodal Phishing Detection using Gen-AI

This project identifies phishing websites using **multimodal data** — both **HTML text** and **image text (OCR)**.  
Modern phishing websites use advanced techniques like deceptive UI, screenshots, and hidden text inside images.  
So this project combines multiple signals to detect phishing pages more accurately.

---

## 🚀 Project Overview

Nowadays, many online scams happen through fake websites that look almost real.  
They steal passwords, OTPs, and bank information.

To solve this, I built a **multimodal phishing detection system** using:

- **HTML content**  
- **Images extracted from the webpage**
- **OCR text from images (EasyOCR)**
- **Deep text embeddings using RoBERTa**
- **A classifier trained on combined features**

This helps detect even **very realistic phishing pages**.

---

## 🧠 Key Features

### ✅ 1. HTML Downloader (Part 1)
- Automatically downloads raw HTML from each URL.
- Saves content into text files.
- Used for text-based feature extraction.

**File:** `part1_html_downloader.py`

---

### ✅ 2. Image Scraper (Part 2)
- Uses Playwright to open webpages.
- Extracts all images larger than **200×200px**.
- OCR reads the text inside images (important for phishing).

**File:** `part2_image_scraper.py`

---

### ✅ 3. OCR + Embedding + Training (Part 3)
- Reads HTML + OCR text.
- Converts text into embeddings using RoBERTa.
- Trains a classifier to distinguish **phishing vs legitimate** URLs.
- Saves embeddings + model.

**File:** `part3_feature_extractor_and_trainer.py`

---

## 📂 Folder Structure (After Running Scripts)

```
multimodal-phishing-detection/
│
├── html_downloader.py
├── image_scraper.py
├── feature_extractor_and_trainer.py
│
├── Alloutputs/         # HTML files saved here
├── images/             # Extracted images saved here
├── features_output.xlsx
├── phishing_clf.joblib # Saved model (after training)
│
├── README.md
└── requirements.txt
```

---

## ⚙️ Tech Stack Used

- **Python**
- **Playwright**
- **EasyOCR**
- **Pandas**
- **PyTorch**
- **RoBERTa**
- **Scikit-learn**
- **Multithreading & Asyncio**

---

## 📊 Dataset

I collected **34,000+ URLs** from:

- Mendeley Phishing Dataset  
- Various trusted legitimate sites  

Each URL contains:

- HTML content  
- Images (converted to text using OCR)  
- Labels: *phishing* / *legitimate*

---

## 🔧 How to Run the Project (Pipeline)

### ▶️ **Step 1: Download HTML**
```
python html_downloader.py
```

### ▶️ **Step 2: Extract Images**
```
python image_scraper.py
```

### ▶️ **Step 3: Train Model**
```
python feature_extractor_and_trainer.py
```

---

## 🎯 Project Goal

To build a **robust and scalable** phishing detection system that can analyze:

- Text content  
- Visual content  
- OCR text  
- Deep contextual meaning  

Useful for:

- **Banking**
- **E-commerce**
- **Cyber security tools**

---

## 🧑‍💻 Author

**Arun Banoth**  
B.Tech CSE, NIT Patna  

---

