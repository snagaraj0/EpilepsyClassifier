import pandas as pd
df_results = pd.DataFrame({'classifier':['KNN', 'Logistic', 'SGD', 'ExtraTrees'], 'auc':[knn_train_auc, knn_test_auc, lr_train_auc, lr_test_auc, sgd_train_auc, sgd_test_auc, etc_train_auc, etc_test_auc]})
import seaborn as sb
import matplotlib.pyplot as plt
sb.set(style="whitegrid")
plt.figure(figsize=(16, 8))
#wahoo seaborn plot
ax = sb.barplot(x = 'classifier', y = 'auc', hue = 'data_set', data = df_results)
ax.set_xlabel('Classifier', fontsize = 20)
ax.set_ylabel('AUC', fontsize = 20)
ax.tick_params(labelsize = 20)
plt.legend(bbox_to_anchor = (1.1, 1), loc = 2, borderaxespad = 0., fontsize = 20)
