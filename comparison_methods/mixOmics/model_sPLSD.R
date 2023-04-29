library(mixOmics)
library(stringr)
require(caret)
set.seed(10)
args<-commandArgs(TRUE)
matrix1 <-args[1]
matrix2 <-args[2]
matrix3 <-args[3]
label_path <-args[4]
dataset<-args[5]
save_path<-args[6]


datasets<-paste('dataset',dataset,sep='_')


message('read count matrix: ', matrix1)
mat1 <- read.table(matrix1, header = TRUE, row.names=1, check.names=FALSE, sep='\t')
message('read count matrix: ', matrix2)
mat2 <- read.table(matrix2, header = TRUE, row.names=1, check.names=FALSE, sep='\t')
message('read count matrix: ', matrix3)
mat3 <- read.table(matrix3, header = TRUE,row.names=1, check.names=FALSE, sep='\t')

message('read class information: ', label_path)
label_info <- read.table(label_path, check.names=FALSE, header = TRUE, sep='\t', as.is=TRUE)
row.names(label_info) <- label_info[,'sample_id']
sample_train<-label_info[label_info[datasets]!='validation','sample_id']
sample_test<-label_info[label_info[datasets]=='validation','sample_id']
y_train<-as.vector(label_info[sample_train,'y'])
y_test<-as.vector(label_info[sample_test,'y'])
ynum<-length(unique(y_train))

mat1_train <- as.matrix(mat1[,sample_train])
mat1_train<-t(mat1_train)
mat2_train <- as.matrix(mat2[,sample_train])
mat2_train<-t(mat2_train)
mat3_train <- as.matrix(mat3[,sample_train])
mat3_train<-t(mat3_train)
mat1_test <- as.matrix(mat1[,sample_test])
mat1_test<-t(mat1_test)
mat2_test <- as.matrix(mat2[,sample_test])
mat2_test<-t(mat2_test)
mat3_test <- as.matrix(mat3[,sample_test])
mat3_test<-t(mat3_test)

X_train <- list(RNA = mat1_train,methylation = mat2_train,CNV = mat3_train)
X_test <- list(RNA = mat1_test,methylation = mat2_test,CNV = mat3_test)

folds <- createFolds(y=sample_train,k=5)
for (n in c(2,5,10)){
    print(n)
    for (m in c(10,50,100,200,500)){
        if ((m*n>dim(mat1_train)[2])|(m*n>dim(mat2_train)[2])|(m*n>dim(mat3_train)[2])){
        print(m)
        next}
        keepX_train <- list(RNA = as.vector(matrix(m,n,1)), methylation = as.vector(matrix(m,n,1)), CNV = as.vector(matrix(m,n,1)))
        data <- array(0,dim=c(5,ynum+ynum+2))
        data_all<- array(0,dim=c(ynum+ynum+ynum+ynum+4,1))
        for(i in 1:5){
        mat1_test_cv<-mat1_train[folds[[i]],]
        mat1_train_cv<-mat1_train[-folds[[i]],]
        mat2_test_cv<-mat2_train[folds[[i]],]
        mat2_train_cv<-mat2_train[-folds[[i]],]
        mat3_test_cv<-mat3_train[folds[[i]],]
        mat3_train_cv<-mat3_train[-folds[[i]],]
        y_test_cv<-y_train[folds[[i]]]
        y_train_cv<-y_train[-folds[[i]]]
        X_train_cv <- list(RNA = mat1_train_cv,methylation = mat2_train_cv,CNV = mat3_train_cv)
        X_test_cv <- list(RNA = mat1_test_cv,methylation = mat2_test_cv,CNV = mat3_test_cv)
        clf_cv <- block.splsda(X_train_cv, y_train_cv,design='full',ncomp=n,keepX=keepX_train)
        predict_test_cv <- predict(clf_cv, newdata = X_test_cv,dist="max.dist")
        AveragedPredict_test_cv<-predict_test_cv$AveragedPredict
        WeightedPredict_test_cv<-predict_test_cv$WeightedPredict
        AveragedPredict_test_cv_mean = apply(AveragedPredict_test_cv, c(1,2), mean)
        WeightedPredict_test_cv_mean = apply(WeightedPredict_test_cv, c(1,2), mean)
        for(s in 1:ynum){data[i,s]<-str_c(as.vector(AveragedPredict_test_cv_mean[,s]),collapse=',')}
        for(s in 1:ynum){data[i,ynum+s]<-str_c(as.vector(WeightedPredict_test_cv_mean[,s]),collapse=',')}
        data[i,ynum+ynum+1]<-str_c(as.vector(sample_train[folds[[i]]]),collapse=',')
        data[i,ynum+ynum+2]<-str_c(as.vector(y_test_cv),collapse=',')
        }
        clf<- block.splsda(X_train, y_train,design='full',ncomp=n,keepX=keepX_train)
        Mypredict_train <- predict(clf, newdata = X_train, dist = "max.dist")
        Mypredict_test <- predict(clf, newdata = X_test, dist = "max.dist")
        AveragedPredict_train<-Mypredict_train$AveragedPredict
        AveragedPredict_test<-Mypredict_test$AveragedPredict
        WeightedPredict_train<-Mypredict_train$WeightedPredict
        WeightedPredict_test<-Mypredict_test$WeightedPredict
        AveragedPredict_train_mean = apply(AveragedPredict_train, c(1,2), mean)
        WeightedPredict_train_mean = apply(WeightedPredict_train, c(1,2), mean)
        AveragedPredict_test_mean = apply(AveragedPredict_test, c(1,2), mean)
        WeightedPredict_test_mean = apply(WeightedPredict_test, c(1,2), mean)
        for(s in 1:ynum){data_all[s,1]<-str_c(as.vector(AveragedPredict_train_mean[,s]),collapse=',')}
        for(s in 1:ynum){data_all[ynum+s,1]<-str_c(as.vector(AveragedPredict_test_mean[,s]),collapse=',')}
        for(s in 1:ynum){data_all[ynum+ynum+s,1]<-str_c(as.vector(WeightedPredict_train_mean[,s]),collapse=',')}
        for(s in 1:ynum){data_all[ynum+ynum+ynum+s,1]<-str_c(as.vector(WeightedPredict_test_mean[,s]),collapse=',')}
        data_all[ynum+ynum+ynum+ynum+1,1]<-str_c(as.vector(sample_train),collapse=',')
        data_all[ynum+ynum+ynum+ynum+2,1]<-str_c(as.vector(sample_test),collapse=',')
        data_all[ynum+ynum+ynum+ynum+3,1]<-str_c(as.vector(y_train),collapse=',')
        data_all[ynum+ynum+ynum+ynum+4,1]<-str_c(as.vector(y_test),collapse=',')
        write.table(data, file=paste(save_path,'result_cv_',n,'_',m,'_new.csv',sep=''),row.names=FALSE,col.names=FALSE,sep='\t')
        write.table(data_all, file=paste(save_path,'result_validation_',n,'_',m,'_new.csv',sep=''),row.names=FALSE,col.names=FALSE,sep='\t')
    }
}

#keepX_train <- list(RNA = c(50, 50), methylation = c(50,50), CNV = c(50, 50))








