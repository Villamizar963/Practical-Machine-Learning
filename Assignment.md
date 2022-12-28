---
title: "Machine Learning Final Report"
subtitle: "Prediction Assignment Writeup"
autor: Angel Villamizar
date: "12/28/2022"
output:
  html_document:
    keep_md: true
    theme: cerulean
    toc: yes
    toc_depth: 1
editor_options: 
  chunk_output_type: console
---


```r
suppressMessages(library(dplyr))
suppressMessages(library(tidyr))
suppressMessages(library(data.table))
suppressMessages(library(ggplot2))
suppressMessages(library(knitr))
suppressMessages(library(lattice))
suppressMessages(library(caret))
suppressMessages(library(rpart))
suppressMessages(library(rpart.plot))
suppressMessages(library(rattle))
suppressMessages(library(randomForest))
suppressMessages(library(corrplot))
suppressMessages(library(gbm))

directory=getwd()

workspace=getwd()
workspace=sprintf("%s/.RData",workspace)
```

# 1.- Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

# 2.- Data Processing

## 2.1.- Data Loading

The training data for this project are available here:

[TRAINING](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

The test data are available here:

[TEST](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)


```r
traincsv <- read.csv("pml-training.csv")
testcsv <- read.csv("pml-testing.csv")

dim(traincsv)
```

```
## [1] 19622   160
```


```r
dim(testcsv)
```

```
## [1]  20 160
```

## 2.2.- Clean data

We remove NA columns. The features containing NA are the variance. Since these values do not add to prediction, they can be removed. We will use this on the training set and assign the same column features to test data set. Remove the first seven features as they are not nummeric or are related to time series.


```r
traincsv <- traincsv[,colMeans(is.na(traincsv)) == 0]
traincsv <- traincsv[,-c(1:7)]
```

Check for near zero variance predictors and drop them if necessary.

```r
nvz <- nearZeroVar(traincsv)
traincsv <- traincsv[,-nvz]
dim(traincsv)
```

```
## [1] 19622    53
```

Select the same predictors in the test set.

```r
colNames <- setdiff(names(traincsv),"classe")

testcsv <- testcsv[,c(colNames,"problem_id")]
dim(testcsv)
```

```
## [1] 20 53
```


```r
dim(traincsv)
```

```
## [1] 19622    53
```

Partioning Data Set.

Split the data into 70% training data and 30% test data. The "classe" variable is in the train set.

```r
set.seed(1234)
inTrain <- createDataPartition(y=traincsv$classe, p=0.7, list=F)

train <- traincsv[inTrain,]
valid <- traincsv[-inTrain,]

dim(train)
```

```
## [1] 13737    53
```

# 3.- Model building

Cross validation is done for each model with K = 3:


```r
fitControl <- trainControl(method='cv', number = 3)
```

For this Project, three prediction methods are used.

## 3.1.- Random Forest


```r
mod_rf <- train(
  classe ~ ., 
  data=train,
  trControl=fitControl,
  method='rf',
  ntree=100
)

saveRDS(mod_rf, file='mod_rf.rds')
```

Prediction with Random Forest

```r
mod_rf <- readRDS(file='mod_rf.rds')

pred_rf <- predict(mod_rf, valid)

cmrf <- confusionMatrix(pred_rf, factor(valid$classe))
cmrf
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    7    0    0    0
##          B    1 1128   11    0    1
##          C    0    4 1013    8    0
##          D    0    0    2  955    0
##          E    0    0    0    1 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9941          
##                  95% CI : (0.9917, 0.9959)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9925          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9903   0.9873   0.9907   0.9991
## Specificity            0.9983   0.9973   0.9975   0.9996   0.9998
## Pos Pred Value         0.9958   0.9886   0.9883   0.9979   0.9991
## Neg Pred Value         0.9998   0.9977   0.9973   0.9982   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1917   0.1721   0.1623   0.1837
## Detection Prevalence   0.2855   0.1939   0.1742   0.1626   0.1839
## Balanced Accuracy      0.9989   0.9938   0.9924   0.9951   0.9994
```

Plot Random Forest

```r
plot(mod_rf)
```

![](Assignment_files/figure-html/unnamed-chunk-12-1.png)<!-- -->

## 3.2.- Decision Tree


```r
model_cart <- train(
  classe ~ ., 
  data=train,
  trControl=fitControl,
  method='rpart'
)

saveRDS(model_cart, file='model_cart.rds')
```

Prediction with Decision Tree

```r
pred_trees <- predict(model_cart, valid)
cmtrees <- confusionMatrix(pred_trees, factor(valid$classe))
cmtrees
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1519  473  484  451  156
##          B   28  401   45  167  148
##          C  123  265  497  346  289
##          D    0    0    0    0    0
##          E    4    0    0    0  489
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4938          
##                  95% CI : (0.4809, 0.5067)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.338           
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9074  0.35206  0.48441   0.0000  0.45194
## Specificity            0.6286  0.91825  0.78946   1.0000  0.99917
## Pos Pred Value         0.4927  0.50824  0.32697      NaN  0.99189
## Neg Pred Value         0.9447  0.85518  0.87881   0.8362  0.89002
## Prevalence             0.2845  0.19354  0.17434   0.1638  0.18386
## Detection Rate         0.2581  0.06814  0.08445   0.0000  0.08309
## Detection Prevalence   0.5239  0.13407  0.25828   0.0000  0.08377
## Balanced Accuracy      0.7680  0.63516  0.63693   0.5000  0.72555
```

Plot Decision Tree

```r
model_cart <- readRDS(file='model_cart.rds')

fancyRpartPlot(model_cart$finalModel)
```

![](Assignment_files/figure-html/unnamed-chunk-15-1.png)<!-- -->

## 3.3.- Generalized Boosted Model (GBM)


```r
model_gbm <- train(
  classe ~ ., 
  data=train,
  trControl=fitControl,
  method='gbm'
)
saveRDS(model_gbm, file='model_gbm.rds')
```

Prediction with Generalized Boosted Model (GBM)

```r
model_gbm <- readRDS(file='model_gbm.rds')

pred_gbm <- predict(model_gbm, valid)

cmgbm <- confusionMatrix(pred_gbm, factor(valid$classe))
cmgbm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1650   36    0    3    2
##          B   15 1078   36    6    9
##          C    5   24  980   22   10
##          D    2    1   10  926    9
##          E    2    0    0    7 1052
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9662          
##                  95% CI : (0.9612, 0.9707)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9572          
##                                           
##  Mcnemar's Test P-Value : 3.934e-06       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9857   0.9464   0.9552   0.9606   0.9723
## Specificity            0.9903   0.9861   0.9874   0.9955   0.9981
## Pos Pred Value         0.9758   0.9423   0.9414   0.9768   0.9915
## Neg Pred Value         0.9943   0.9871   0.9905   0.9923   0.9938
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2804   0.1832   0.1665   0.1573   0.1788
## Detection Prevalence   0.2873   0.1944   0.1769   0.1611   0.1803
## Balanced Accuracy      0.9880   0.9663   0.9713   0.9781   0.9852
```

Plot (GBM)

```r
plot(model_gbm)
```

![](Assignment_files/figure-html/unnamed-chunk-18-1.png)<!-- -->

# 4.- Results


```r
AccuracyResults <- data.frame(
  Model = c('Random Forest', 'Decision Tree', 'GBM'),
  Accuracy = rbind(cmrf$overall[1], cmtrees$overall[1], cmgbm$overall[1])
)
print(AccuracyResults)
```

```
##           Model  Accuracy
## 1 Random Forest 0.9940527
## 2 Decision Tree 0.4937978
## 3           GBM 0.9661852
```

Based on an assessment of these 3 model fits and out-of-sample results, it looks like both gradient boosting and random forests outperform the Decision Tree model, with random forests being slightly more accurate. 

As a last step in the project, the validation data sample (‘pml-testing.csv’) is used to predict a classe for each of the 20 observations based on the other information about these observations contained in the validation sample.


```r
ptest <- predict(mod_rf, testcsv)

ValidationPredictionResults <- data.frame(
  problem_id=testcsv$problem_id,
  predicted=ptest
)
print(ValidationPredictionResults)
```

```
##    problem_id predicted
## 1           1         B
## 2           2         A
## 3           3         B
## 4           4         A
## 5           5         A
## 6           6         E
## 7           7         D
## 8           8         B
## 9           9         A
## 10         10         A
## 11         11         B
## 12         12         C
## 13         13         B
## 14         14         A
## 15         15         E
## 16         16         E
## 17         17         A
## 18         18         B
## 19         19         B
## 20         20         B
```

Despite these remaining questions on missing data in the samples, the random forest model with cross-validation produces a surprisingly accurate model that is sufficient for predictive analytics.
