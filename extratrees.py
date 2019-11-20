from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

etc = ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=1.0, min_samples_leaf=3, min_samples_split=20, n_estimators=100)
etc.fit(X_train_tf, y_train)

y_train_preds = etc.predict_proba(X_train_tf)[:, 1]
y_test_preds = etc.predict_proba(X_test_tf)[:, 1]

print('Extra Trees Classifier')
print('Training:')
etc_train_auc, etc_train_accuracy, etc_train_recall, etc_train_precision,etc_train_specificity = print_report(y_train, y_train_preds, thresh)
print('Testing:')
etc_valid_auc, etc_valid_accuracy, etc_valid_recall, etc_valid_precision, etc_valid_specificity = print_report(y_valid, y_valid_preds, thresh)
