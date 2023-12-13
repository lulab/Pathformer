# Comparison methods

We conducted a meticulous benchmark of Pathformer and 18 other multi-omics data integration methods for various classification tasks in cancer diagnosis and prognosis. These methods can be categorized into 3 types. Type I includes early and late integration methods based on conventional classifiers, such as support vector machine (SVM), logistic regression (LR), random forest (RF), and extreme gradient boosting (XGBoost). Type II includes partial least squares-discriminant analysis (PLSDA) and sparse partial least squares-discriminant analysis (sPLSDA) of mixOmics. Type III consists of deep learning-based integration methods, i.e., eiNN, liNN, eiCNN, liCNN, MOGONet, MOGAT, P-NET and PathCNN. Among these, eiNN and eiCNN are early integration methods based on NN and CNN; liNN and liCNN are late integration methods based on fully connected neural network (NN) and convolutional neural network (CNN); MOGONet and MOGAT are multi-modal integration methods based on graph neural network; P-NET and PathCNN are representative multi-modal integration methods that combines pathway information. During comparison methods, the multi-modal data were preprocessed with the statistical indicators and features were prefiltered with ANOVA as input. More details see the article of Pathformer.

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
### 5. eiNN

```
bash eiNN/log_eiNN.sh
```

### 6. liNN

```
bash liNN/log_liNN.sh
```

### 7. eiCNN

```
bash eiCNN/log_eiCNN.sh
```

### 8. liCNN

```
bash liCNN/log_liCNN.sh
```

### 9. MOGONet

```
bash MOGONet/log_MOGONET.sh
```

### 10. MOGAT

```
bash MOGAT/log_MOGAT.sh
```

### 11. PathCNN

```
bash PathCNN/main_log.sh
```

### 12. P-NET

```
bash P_net/log_Pnet.sh
```

### 13. Result

Here we take two classification tasks of breast cancer as an example, including breast cancer early- and late- stage classification, and breast cancer low- and high- survival risk classification.
