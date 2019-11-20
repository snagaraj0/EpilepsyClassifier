import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors = 100)
knn.fit(X_train_tf, y_train)

y_train_preds = knn.predict_proba(X_train_tf)[:,1]
y_test_preds = knn.predict_proba(X_test_tf)[:,1]

print('KNN')
print('Training:')
knn_train_auc, knn_train_accuracy, knn_train_recall, knn_train_precision = print_report(y_train,y_train_preds, 0.5)
print('Testing:')
knn_test_auc, knn_test_accuracy, knn_test_recall, knn_test_precision = print_report(y_test,y_test_preds, 0.5)

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
def print_report(y_actual, y_pred, thresh):
    auc = roc_auc_score(y_actual, y_pred)
    accuracy = accuracy_score(y_actual, (y_pred > thresh))
    recall = recall_score(y_actual, (y_pred > thresh))
    precision = precision_score(y_actual, (y_pred > thresh))
    print('AUC:%.3f'% auc)
    print('accuracy:%.3f'%accuracy)
    print('recall:%.3f'%recall)
    print('precision:%.3f'%precision)
    return auc, accuracy, recall, precision
