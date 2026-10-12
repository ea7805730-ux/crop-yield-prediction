# 🌾 Crop Yield Prediction

Machine learning project that predicts crop yield (tonnes/hectare) from rainfall,
temperature, humidity, soil pH, soil nutrients (N, P, K), soil type, crop type and
fertilizer use.

## Models compared
Linear Regression, Random Forest, Gradient Boosting — the best one (by R²) is saved.

## Setup
```bash
git clone https://github.com/<your-username>/crop-yield-prediction.git
cd crop-yield-prediction
pip install -r requirements.txt
```

## Run
```bash
python train.py      # trains, compares models, saves model + plots
python predict.py    # predicts yield for a sample field
```

To use your own data (e.g. from Kaggle), put it in `data/crop_yield.csv` with a
`Yield_t_ha` target column, or run `python train.py --data file.csv --target YourColumn`.
If no dataset is present, demo data is generated automatically.

## Project structure
```
├── data/            dataset (CSV)
├── models/          saved model
├── outputs/         plots
├── train.py         training + evaluation
├── predict.py       prediction on new input
└── requirements.txt
```
