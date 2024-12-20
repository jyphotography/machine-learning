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
midterm/
├── data/
│   └── NY-House-Dataset.csv
├── notebooks/
│   └── notebook.ipynb
├── src/
│   ├── training.py
│   ├── predict.py
│   └── model_evaluation.py
│   └── Dockerfile
├── requirements.txt
├── README.md
```

## Requirements

- Python 3.11 or higher
- Docker

## Instructions on How to Run the Project

1. **Clone the Repository**

   ```bash
   git clone https://github.com/jyphotography/machine-learning.git
   cd machine-learning/midterm/
   ```
2. **Set Up a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
3. **EDA & ModeL Training**
    ```bash
    jupyter notebook notebooks/notebook.ipynb
    ```
4. **Model Reproduce & Deployment**
    ```bash
    python src/training.py
    python src/predict.py
    python src/test.py # test model pipeline
    ```
5. **Containerization**
    ```bash
    # docker was installed and leave venv 
    cd src
    docker build -t house-prediction .
    docker run -it -p 9696:9696 house-prediction:latest
    python src/test.py # test model pipeline in main folder
    ```
    You should be able to use Postman to call this API with a JSON file to get your house prediction result.
    ```bash
    http://0.0.0.0:9696/predict
    {"BEDS": 4, "BATH": 2, "PROPERTYSQFT": 2184, "SUBLOCALITY": "Queens County"}
    ```
    ![Result](api_call_result.png)



## Results
The machine learning model achieved an RMSE 580k. The price range in Mahanttan is huge, and need to segment for actual use cases in the future.
The model can assist potential buyers and sellers in making informed decisions based on predictive analytics.


## License
This project is licensed under the MIT License.
