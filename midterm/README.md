# NY Housing Price Prediction

Since I just moved to NYC, I wanted to use a machine learning model to predict housing prices.

## Description of the Problem

Predicting housing prices in New York City is a complex task due to the city's diverse neighborhoods and varying property characteristics. This project aims to develop a machine learning model that accurately predicts housing prices based on various features such as location, property size, number of bedrooms and bathrooms, and more.

### Data

This dataset contains prices of New York houses, providing valuable insights into the real estate market in the region. It includes information such as:

- Broker titles
- House types
- Prices
- Number of bedrooms and bathrooms
- Property square footage
- Addresses
- State
- Administrative and local areas
- Street names
- Geographical coordinates

## Project Structure
```css
NY-Housing-Price-Prediction/
├── data/
│   └── ny_housing_data.csv
├── notebooks/
│   └── data_exploration.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── model_evaluation.py
├── models/
│   └── housing_price_model.pkl
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.7 or higher
- Jupyter Notebook
- Required Python libraries listed in `requirements.txt`

## Instructions on How to Run the Project

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/NY-Housing-Price-Prediction.git
   cd NY-Housing-Price-Prediction
   ```
2. **Set Up a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    jupyter notebook notebooks/data_exploration.ipynb
    python src/model_training.py
    python src/model_evaluation.py
    python src/predict.py --input data/new_listings.csv --output predictions.csv
    ```



## Results
The machine learning model achieved an R-squared score of X.XX, indicating a strong correlation between the predicted and actual housing prices. The model can assist potential buyers and sellers in making informed decisions based on predictive analytics.


## License
This project is licensed under the MIT License.
