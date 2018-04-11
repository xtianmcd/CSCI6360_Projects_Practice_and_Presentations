import pandas as pd

white = pd.read_csv('winequality-white.csv', sep=';')
red = pd.read_csv('winequality-red.csv', sep=';')

whitec = white
redc = red

# print(white.head())
# print(red.head())

print(white.shape)
print(red.shape)


dfs = [white, red]
df_num = 0

for df in dfs:
    unique = 0

    for col in df:
        # print(col)
        unique = df[col].nunique()
        # print("{} has {} unique vals".format(col, unique))
        # print("{} max: {}, min: {}".format(col, df[col].max(), df[col].min()))
        df[col] = pd.cut(df[col], 10, labels=False, include_lowest=True)
        unique = df[col].nunique()
        # print("{} has {} unique vals".format(col, unique))
        # print("{} max: {}, min: {}".format(col, df[col].max(), df[col].min()))
    df.to_csv("winequality-{}-discretized.csv".format(df_num))
    df_num += 1
