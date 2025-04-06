# 🌍 Alpiq Datathon 2025 – Energy Demand Forecasting

## 📊 Project: Time Series Forecasting of Energy Demand Data

Welcome to the solution repository by team musslos for the **Alpiq Challenge at the Datathon 2025**. This project tackles the problem of accurately forecasting energy demand using real-world time series data provided by the challenge organizers.

Note that the main Italy data is missing

---

## 👥 Team

- Nessim  
- Vinícius  
- Carlo  
- Tristan  

---

## 📌 Challenge Overview

The goal was to **forecast hourly energy usage** for a set of anonymized customers using provided historical data. The dataset included customer profiles such as:

- 🏠 Private Households – showing periodic usage patterns with visible day/night cycles and lower activity on weekends/holidays.
- 🏭 Industrial Consumers – exhibiting more constant usage profiles.

---

## 🔍 Problem Analysis

We identified key characteristics in the data:

- **Strong daily and weekly seasonality**
- **Day/night consumption differences**
- **Distinct customer usage profiles**

We conducted **Fourier Analysis** to detect periodic signals and decomposed the data into trend, seasonal, and residual components.

---

## 🧠 Our Approach

We evaluated a variety of methods from classical to deep learning approaches, but ultimately selected a **feature-engineered Machine Learning model** using **LightGBM** for its performance and interpretability.

### Explored Techniques:

- **Classical Time Series Models**: ARIMA, SARIMA
- **Deep Learning Models**: LSTM, TCN, Transformer
- **Machine Learning Models**: LightGBM (selected)

---

## 🏗️ Solution Pipeline

1. **Data Preprocessing**  
   - Time series alignment  
   - Handling missing values  
   - Feature extraction (e.g., hour of day, day of week, holidays)

2. **Exploratory Analysis**  
   - Fourier transform to understand frequency components  
   - Correlation analysis for seasonal behavior  

3. **Modeling**  
   - LightGBM regression with time-aware features  
   - Boosted trees to capture non-linear patterns in consumption  

4. **Evaluation**  
   - Train-test split based on time  
   - Metrics: RMSE, MAE  

---

## 📁 Repository Structure

```
.
├── data/                    # Sample and processed datasets (if included)
├── notebooks/              # EDA and modeling notebooks
├── src/                    # Core scripts for preprocessing and modeling
├── presentation/           # Slides for the 5-minute pitch
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 📽️ Presentation

The project presentation (5 minutes) is available in the `presentation/` folder or view it [here](./presentation/).

---

## ✅ Submission Notes

- GitHub repo submitted via: [http://datathon.ai/submission](http://datathon.ai/submission)  
- All files were frozen before the challenge deadline (12:00), in accordance with the rules.  
- No post-deadline changes were made to ensure compliance.

---

## 🚀 Getting Started

To reproduce our results:

```bash
git clone https://github.com/vimohr/ETH_Datathon_2025.git
cd ETH_Datathon_2025
pip install -r requirements.txt
```

Run the pipeline via the notebooks or scripts in `src/`.

---

## 📬 Contact

Feel free to reach out for collaborations or questions!
