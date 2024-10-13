import pandas as pd


def parse_dt(df, cols=["timestamp"]):
    return df.assign(
        **{
            col: lambda df: pd.to_datetime(df[col].astype(int), unit="ms")
            for col in cols
        }
    )


def handle_dtypes(df):
    return df.assign(rating=lambda df: df["rating"].astype(float))
