# 📈 AdsVision: Advanced Social Media Campaign Analytics

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B.svg)
![Plotly](https://img.shields.io/badge/Plotly-Interactive--Charts-3F4F75.svg)
![Data Analysis](https://img.shields.io/badge/Focus-Data%20Analysis-green.svg)

**AdsVision** is a professional-grade marketing intelligence dashboard built to transform fragmented advertising data into actionable strategic insights. It goes beyond simple charts by providing a full performance audit, financial tracking, and automated reporting.

---

## 🚀 Key Features

### 💎 Premium User Interface
- **Custom-Themed UI:** A modern dark-mode interface built with custom CSS gradients and glassmorphism effects.
- **Responsive Layout:** Optimized for high-resolution screens and wide layouts.

### 📊 Deep Analytics Engine
- **Multi-Level KPI Tracking:** Instant metrics for Revenue, ROI (Return on Investment), CPA (Cost Per Acquisition), and Budget Utilization.
- **Cross-Platform Benchmarking:** Compare performance across different social media platforms using dual-axis interactive charts.
- **Demographic Drill-down:** Heatmaps and distribution charts for Location, Age, Gender, and User Interests.

### 📋 Smart Reporting System
- **Automated Audit:** One-click generation of comprehensive performance reports.
- **Multi-Format Export:** Download your insights as **Structured Text**, **Interactive HTML**, or **Filtered CSV** for further use.

---

## 📂 Project Architecture

```text
├── app.py                 # Core application logic & UI
├── Cleaned_Data.csv       # Default sample dataset
├── requirements.txt       # Environment dependencies
└── README.md              # Project documentation
```


---

## 🛠️ Technical Implementation

### Data Processing Pipeline

The app utilizes a robust processing function that:

1. Validates required marketing columns.
2. Performs time-series feature engineering (Month, Quarter, Duration).
3. Handles missing values and calculates financial health metrics (Profit & ROI).

### Visualization Stack

* **Plotly Graph Objects:** Used for complex dual-axis charts (Revenue vs. ROI).
* **Plotly Express:** Used for Treemaps (Language distribution) and Sunburst charts (Geographic performance).

---

## ⚙️ Installation & Usage

1. **Clone the repository:**
```bash
git clone [https://github.com/your-username/adsvision-dashboard.git](https://github.com/your-username/adsvision-dashboard.git)
cd adsvision-dashboard

```


2. **Install Dependencies:**
```bash
pip install -r requirements.txt

```


3. **Launch the Dashboard:**
```bash
streamlit run app.py

```



---

## 💡 Strategic Insights Covered

The dashboard's internal logic is designed to automatically identify:

* 🥇 **Top Performers:** Best platform/location combinations.
* 💸 **Waste Detection:** Platforms with high CPA but low conversion.
* 🚀 **Expansion Opportunities:** Under-budgeted segments with high ROI.

---

## ✨ Contributor

* **Mariam Mohamed Sayed** – *Data Analyst & Developer*

---
