# 🏍️ Automated Detection of Non-Helmeted Motorcyclists & License Plate Recognition

### 🚦 YOLOv8 + PaddleOCR Based Traffic Violation Detection System

![Python](https://img.shields.io/badge/PYTHON-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOV8-ULTRALYTICS-00FFFF?style=for-the-badge)
![PaddleOCR](https://img.shields.io/badge/PADDLEOCR-OCR-FF6F00?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/STREAMLIT-WEB%20APP-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

Deep learning-based traffic monitoring system using YOLOv8, YOLOv3, CNN, and OCR (PaddleOCR / TrOCR) to detect non-helmeted motorcyclists and automatically recognize vehicle license plate numbers from traffic video feeds.

---

## 📌 Features

- **Helmet Violation Detection:** Detects motorcyclists riding without a helmet using real-time object detection models.
- **License Plate Extraction:** Automatically crops and extracts license plates from violating vehicles.
- **Optical Character Recognition (OCR):** Extracts license plate text using PaddleOCR / TrOCR.
- **CSV Data Export:** Saves recorded violations with detected plate text to CSV file (`no_helmet_plates_with_text.csv`).
- **Interactive Web UI:** Built-in Streamlit app (`newapp.py`) for processing video uploads and displaying live detection results.

---

## 📁 Repository Structure

```
.
├── detect.py                     # YOLOv3 + CNN detection script
├── detect_lic.py                 # License plate detection & OCR script
├── newapp.py                     # Streamlit web application
├── helmet.ipynb                  # Training and analysis notebook
├── yolov3-custom.cfg             # YOLOv3 model configuration
├── no_helmet_plates_with_text.csv # Recorded violations CSV output
├── README.md                     # Project documentation
└── .gitignore                    # Git ignore file for heavy weight models
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/gowthamkumar42/helmet-violation-detection-yolov8-paddleocr.git
cd helmet-violation-detection-yolov8-paddleocr
```

### 2. Install dependencies
```bash
pip install opencv-python numpy tensorflow streamlit paddleocr ultralytics imutils
```

### 3. Download Model Weights
Place the following model weight files into the root project folder:
- `yolov3-custom_7000.weights`
- `helmet-nonhelmet_cnn.h5`

### 4. Run Web Application
```bash
streamlit run newapp.py
```

---

## 📄 License
This project is open-source and available for educational & research purposes.
