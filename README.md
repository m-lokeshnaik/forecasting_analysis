
---

## 📄 Dataset Description

The dataset contains weekly sales data with the following key columns:

| Column Name      | Description                                 |
|------------------|---------------------------------------------|
| `week_end_date`  | Date marking the end of the sales week      |
| `channel`        | Sales channel (e.g., Online, Offline)       |
| `brand`          | Product brand                               |
| `category`       | Product category                            |
| `sub_category`   | Product sub-category                        |
| `SerialNum`      | Unique identifier for each product          |
| `quantity`       | Weekly units sold (target variable)         |

---

## ⚙️ Project Workflow

### 1. Data Preparation
- Converted date column to datetime format.
- Sorted chronologically and extracted time-based features (week, month, year).

### 2. Exploratory Data Analysis (EDA)
- Performed trend analysis on sales quantity over time.
- Analyzed sales by product category and brand.

### 3. Feature Engineering
- Applied one-hot encoding to categorical features: `channel`, `brand`, `category`, `sub_category`.

### 4. Model Building
- Built **individual Random Forest Regressor models** per `SerialNum`.
- Trained on data before **June 2024**.
- Validated on **June to August 2024** using **Mean Absolute Error (MAE)**.

### 5. Forecasting
- Forecasted quantities for **September to November 2024**.
- Combined predictions saved to a single CSV file.

### 6. Model Saving
- Each trained model is saved in `.pkl` format inside the `models/` folder for reproducibility.

---

## 🧪 Model Details

- **Model Type**: `RandomForestRegressor`
- **Framework**: `scikit-learn`
- **Validation Metric**: `Mean Absolute Error (MAE)`
- **Separate models** were trained per product to account for product-level trends.

---

## 🚀 How to Run

1. **Install required libraries**:
```bash
pip install pandas scikit-learn joblib matplotlib seaborn
