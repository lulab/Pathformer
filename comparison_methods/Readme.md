# Comparison methods

We compared the classification performance of Pathformer with several existing multi-modal integration methods, including early integration methods based on base classifiers, i.e., nearest neighbor algorithm (KNN), support vector machine (SVM), logistic regression (LR), random forest (RF), and extreme gradient boosting (XGBoost); late integration methods based on KNN, SVM, LR, RF, and XGBoost; partial least squares-discriminant analysis (PLSDA) and sparse partial least squares-discriminant analysis (sPLSDA) of mixOmics; two deep learning-based integration methods, MOGONet and PathCNN. MOGONet is a multi-modal integration method based on graph convolutional neural network. PathCNN is a representative multi-modal integration method that combines pathway information. During comparison methods, the multi-modal data were preprocessed with the statistical indicators and features were prefiltered with ANOVA as input.


### 1. Multi-modal feature selection by ANOVA for comparison methods

```
bash data_feature_filter/log_data_feature_filter.sh
```

### 2. Early integration methods based on base classifiers

```
bash early_integration_method/log_early_integration_method.sh
```

### 3. Late integration methods based on base classifiers

```
bash late_integration_method/log_late_integration_method.sh
```

### 4. Supervised methods in mixOmics

```
bash mixOmics/log_PLSDA.sh
bash mixOmics/log_sPLSDA.sh
```
### 5. MOGONet

```
bash MOGONet/log_MOGONET.sh
```
### 6. PathCNN

```
bash PathCNN/main_log.sh
```
### 7. Result

Here we take three classification tasks of breast cancer as an example, including breast cancer subtype classification, breast cancer early- and late- stage classification, and breast cancer low- and high- survival risk classification.
