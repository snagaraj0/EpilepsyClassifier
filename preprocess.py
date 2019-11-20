import pandas as pd
import numpy as np
df_frame = pd.read_csv("epilepsy.csv")
df_frame["output"] = df_frame.y == 1
df_frame["output"] = df_frame["output"].astype(int)
df_frame.pop('y')
#remove first column bc it has uninformative id number
df_frame.drop(df_frame.columns[0], axis=1, inplace=True)

#create new dataframe with only the parameters and weights
inputs = df_frame.columns.tolist()
column_inputs = inputs[0:178]
df = df_frame[column_inputs + ["output"]]

#Balance the various parameter possibilities between the training and testing set
temp = len(df)
df = df.sample(temp)
df = df.reset_index(drop=True)
df_balance = df.sample(frac=0.3)
#test
df_test = df_balance.sample(frac=0.5)
#train set
df_train = df.drop(df_balance.index)

#for training we need to balance out yes epilepsy and no epilepsy results so our classifier doesn't get skewed either way.
rows = df_train.output == 1
#positive and negative results
df_yes = df_train.loc[rows]
df_no = df_train.loc[~rows]

n = np.min([len(df_yes), len(df_no)])

# put together yes no results
df_train_all = pd.concat([df_yes.sample(n=n, random_state=42), df_no.sample(n=n, random_state=42)], axis=0, ignore_index=True)
df_train_all = df_train_all.sample(n=len(df_train), random_state=42).reset_index(drop=True)

df_test.to_csv('test.csv', index=False)
df_train.to_csv('train.csv', index=False)
df_train_all.to_csv('train_all.csv', index=False)

import pickle
file = open('column_inputs.csv', 'wb')
pickle.dump(column_inputs, file)
            
X_train = df_train[column_inputs].values
X_train_all = df_train_all[column_inputs].values
X_test = df_test[column_inputs].values
            
y_train = df_train['output'].values


from sklearn.preprocessing import StandardScaler
scaler  = StandardScaler()
scaler.fit(X_train_all)

scalerfile = 'scaler.csv'
pickle.dump(scaler, open(scalerfile, 'wb'))
scaler = pickle.load(open(scalerfile, 'rb'))

# transform matrices
X_train_tf = scaler.transform(X_train)
X_test_tf = scaler.transform(X_test)
