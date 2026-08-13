# 🛒 Shelf Placement Movement Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://shelf-placement-app-ridayastha.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning web application built with **Streamlit** and **scikit-learn** that predicts product turnover class (**Fast ⚡**, **Slow 🐢**, or **Ignored 💤**) based on shelf placement, store demographics, product pricing, and customer traffic metrics.

🚀 **Live Web App:** [https://shelf-placement-app-ridayastha.streamlit.app/](https://shelf-placement-app-ridayastha.streamlit.app/)

---

## 📌 Project Overview

Optimizing retail shelf space is critical for maximizing sales velocity and reducing unsold inventory. This application leverages a trained **Logistic Regression** model and **StandardScaler** pipeline to forecast how quickly a product will move off the shelf based on spatial, financial, and promotional factors.

Key capabilities include:
- Interactive parameter sliders and input controls for real-time scenario analysis.
- Multi-class classification predictions (**Fast**, **Slow**, **Ignored**).
- Probability confidence breakdown for all turnover classes.
- Standardized feature transformation matching trained model parameters.

---

## 🛠️ Tech Stack

- **Frontend / UI:** [Streamlit](https://streamlit.io/)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning:** [scikit-learn](https://scikit-learn.org/) (LogisticRegression, StandardScaler)
- **Model Serialization:** [Joblib](https://joblib.readthedocs.io/)
- **Version Control & Deployment:** Git, GitHub, Streamlit Community Cloud

---

## 📊 Model Features & Input Parameters

The prediction pipeline evaluates **8 core features**:

| Feature Name | Description | Data Type | Range / Options |
| :--- | :--- | :--- | :--- |
| `zone_code` | Categorical store zone identifier | Integer | `0` to `10` |
| `height_code` | Vertical shelf placement height | Categorical | `0` (Low), `1` (Eye Level), `2` (High) |
| `item_category_code` | Category code for the product | Integer | `0` to `20` |
| `city_code` | Geographic city location identifier | Integer | `0` to `10` |
| `price` | Retail price of the item ($) | Numeric | `>= 0.00` |
| `discount_percent` | Promotional discount percentage | Continuous | `0.0%` to `100.0%` |
| `daily_customer_traffic` | Average daily foot traffic past shelf | Continuous | `>= 0.0` |
| `nearby_promotion` | Active promotional banner or display nearby | Binary | `0` (No), `1` (Yes) |

---

## 📁 Repository Structure

```text
shelf_placement_app/
├── app.py              # Streamlit web application entry point
├── model.pkl           # Pre-trained Logistic Regression model artifact
├── scaler.pkl          # Fitted StandardScaler preprocessing artifact
├── requirements.txt    # Python dependencies for deployment
├── .gitignore          # Git exclusion rules
└── README.md           # Project documentation
🚀 Quickstart: Local Setup & Running
Follow these steps to run the application locally on your machine:

1. Clone the Repository
Bash
git clone https://github.com/ridayastha/shelf_placement_app.git
cd shelf_placement_app
2. Set Up Virtual Environment (Optional)
Bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
3. Install Dependencies
Bash
pip install -r requirements.txt
4. Launch the Streamlit App
Bash
streamlit run app.py
📜 License
Distributed under the MIT License. See LICENSE for more information.


---

### Terminal commands to push it:

Run these in your command prompt after saving the file:

```cmd
git add README.md
git commit -m "Fix README formatting"
git push
