import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from src.data_loaders.loaders import FEATURE_COLUMNS, TARGET_COLUMN


def prepare_data(df: pd.DataFrame, target_column: str):
    df = df.copy()

    le = LabelEncoder()
    df[target_column] = le.fit_transform(df[target_column])
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    return X_train, X_test, y_train, y_test, le


def prepare_simulation_data(df: pd.DataFrame):
    df = df.copy()
    SPLIT_STEP = int(df["time"].max() * 0.8)
    train_df = df[df["time"] < SPLIT_STEP]
    test_df = df[df["time"] >= SPLIT_STEP]

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test
