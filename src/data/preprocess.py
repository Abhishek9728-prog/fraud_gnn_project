import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
CSV_PATH = ROOT_DIR.parent / 'archive' / 'final_fraud_dataset.csv'

TARGET = 'isFraud'

NUMERIC_FEATURES = [
    'TransactionAmt',
    'TransactionDT',
    'C1', 'C2', 'C3', 'C4', 'C5',
    'D1', 'D2', 'D3', 'D4', 'D5',
    'id_01', 'id_02', 'id_03',
    'id_05', 'id_11'
]

CATEGORICAL_FEATURES = [
    'ProductCD',
    'card1',
    'card2',
    'addr1',
    'P_emaildomain',
    'R_emaildomain',
    'DeviceType',
    'DeviceInfo',
    'id_12'
]

SCALE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(nrows=None):

    print(f"Loading data from {CSV_PATH}...")

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Could not find data at {CSV_PATH}"
        )

    usecols = (
        [TARGET]
        + NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    df = pd.read_csv(
        CSV_PATH,
        usecols=usecols,
        nrows=nrows
    )

    print(f"Loaded {len(df)} rows.")

    return df


def preprocess(df):

    print("Preprocessing data...")

    for col in NUMERIC_FEATURES:

        df[col] = df[col].fillna(
            df[col].median()
        )

    for col in CATEGORICAL_FEATURES:

        df[col] = df[col].fillna('unknown')
        df[col] = df[col].astype(str)

    print("Encoding categorical features...")

    encoders = {}

    for col in CATEGORICAL_FEATURES:

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col])

        encoders[col] = le

    print("Scaling numerical features...")

    scaler = StandardScaler()

    df[SCALE_FEATURES] = scaler.fit_transform(
        df[SCALE_FEATURES]
    )

    df = df.sort_values(
        'TransactionDT'
    ).reset_index(drop=True)

    return df, encoders, scaler

def save_artifacts(encoders, scaler):
    print("Saving preprocessing artifacts...")
    encoder_path = PROCESSED_DIR / 'label_encoders.pkl'
    scaler_path = PROCESSED_DIR / 'scaler.pkl'

    with open(encoder_path, 'wb') as f:
        pickle.dump(encoders, f)

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Saved encoders to: {encoder_path}")
    print(f"Saved scaler to: {scaler_path}")


if __name__ == '__main__':

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )
    df_raw = load_data(nrows=100000)
    df_clean, encoders, scaler = preprocess(df_raw)
    out_path = (
        PROCESSED_DIR
        / 'cleaned_transactions.parquet'
    )
    print(f"Saving preprocessed data to {out_path}...")
    df_clean.to_parquet(
        out_path,
        index=False
    )

    save_artifacts(encoders, scaler)

    print("Done!")