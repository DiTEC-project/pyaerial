import pandas as pd
from mlxtend.preprocessing import TransactionEncoder


def one_hot_encode_dataframe(dataframe: pd.DataFrame):
    transactions = prepare_transactions(dataframe)

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_ary, columns=te.columns_)


def prepare_transactions(dataframe: pd.DataFrame):
    return [
        [f"{column}__{value}" for column, value in row.items()]
        for _, row in dataframe.iterrows()
    ]
