import pandas as pd
df_frame = pd.read_csv("epilepsy.csv")
df_frame["output"] = df_frame.y == 1
df_frame["output"] = df_frame["output"].astype(int)
df_frame.pop('y')
#remove first column bc it has uninformative id number
df_frame.drop(df_frame.columns[0], axis=1, inplace=True)
