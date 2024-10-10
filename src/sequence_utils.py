import numpy as np


def generate_item_sequences(
    df,
    user_col,
    item_col,
    timestamp_col,
    sequence_length,
    padding=True,
    padding_value=-1,
):
    """
    Generates a column 'item_sequence' containing lists of previous item indices for each user.

    Parameters:
    - df: DataFrame containing the data
    - user_col: The name of the user column
    - item_col: The name of the item column
    - timestamp_col: The name of the timestamp column
    - sequence_length: The maximum length of the item sequence to keep
    - padding: whether to pad the item sequence with `padding_value` if it's shorter than sequence_length
    - padding_value: value used for padding

    Returns:
    - DataFrame with an additional column 'item_sequence'
    """

    def get_item_sequence(sub_df):
        sequences = []
        for i in range(len(sub_df)):
            # Get item indices up to the current row (excluding the current row)
            sequence = sub_df.iloc[:i].tolist()[-sequence_length:]
            if padding:
                padding_needed = sequence_length - len(sequence)
                sequence = np.pad(
                    sequence,
                    (padding_needed, 0),  # Add padding at the beginning
                    "constant",
                    constant_values=padding_value,
                )
            sequences.append(sequence)
        return sequences

    df = df.sort_values(timestamp_col)
    df["item_sequence"] = df.groupby(user_col, group_keys=True)[item_col].transform(
        get_item_sequence
    )
    df["item_sequence"] = df["item_sequence"].fillna("").apply(list)

    return df
