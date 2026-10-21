# Predictor — NFL Analytics (Streamlit)

## Run locally
1. Install Python 3.10 or 3.11.
2. Put all files from the Colab export ZIP next to `app.py`.
3. Install dependencies: `pip install -r requirements.txt`
4. Start the app: `streamlit run app.py`

## Deploy
Push these files to a private or public GitHub repository, then create a Streamlit Community Cloud app using `app.py` as the main file.

## Included model package files
- `ridge_model.joblib`
- `feature_cols.json`
- `current_form.csv`
- `current_elo.csv`
- `week7_predictions.csv`
- `historical_data.csv`

## Limitations
This is a first connected prototype. It uses the trained Ridge pipeline for matchup margin predictions, current form and Elo snapshots from the exported package, and the saved weekly predictions CSV. It does not update live scores, odds, injury data, or current team features automatically. Win probabilities are rough logistic conversions from margin and must be calibrated before being treated as reliable.
