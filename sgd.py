from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(random_state = 42)
sgd.fit(X_train_tf, y_train)

y_train_preds = lr.predict_proba(X_train_tf)[:,1]
y_test_preds = lr.predict_proba(X_test_tf)[:,1]

print('Stochastic Gradient Descent')
print('Training:')
lr_train_auc, lr_train_accuracy, lr_train_recall, lr_train_precision = print_report(y_train,y_train_preds, 0.5)
print('Testing:')
lr_test_auc, lr_test_accuracy, lr_test_recall, lr_test_precision = print_report(y_valid,y_valid_preds, 0.5)

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
def print_report(y_actual, y_pred, thresh):
    auc = roc_auc_score(y_actual, y_pred)
    accuracy = accuracy_score(y_actual, (y_pred > thresh))
    recall = recall_score(y_actual, (y_pred > thresh))
    precision = precision_score(y_actual, (y_pred > thresh))
    print('AUC:%.3f'%auc)
    print('accuracy:%.3f'%accuracy)
    print('recall:%.3f'%recall)
    print('precision:%.3f'%precision)
    return auc, accuracy, recall, precision

