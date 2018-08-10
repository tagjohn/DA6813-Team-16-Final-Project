# DA6813-Team-16-Final-Project
```{r}
library(sqldf)
library(dplyr)
library(ggplot2)
library(e1071)
library(caret)
library(FNN)
library(Hmisc)
library(randomForest)
library(pROC)
require(MASS)
library (tree)
```


Build a model to predict if certain groupings of factors indicate that people of different races will be attracted

***********************
```{r}

df <- read.csv("/Users/hannina/Documents/Summer 18/Project/Speed Dating Data.csv", head = TRUE, na.strings = c("NA", ""), stringsAsFactors = F)
#str(df)
dim(df)


# Changing the Variables
# Gender variables changed to "F" and "M".
# Same race variables changed to "Yes" and "No".
# Matching variables changed to "Yes" and "No".
df[df$gender == 0,]$gender <- "F"
df[df$gender == 1,]$gender <- "M"
df[df$samerace == 0,]$samerace <- "No"
df[df$samerace == 1,]$samerace <- "Yes"      
df[df$match == 0,]$match <- "No"
df[df$match == 1,]$match <- "Yes" 


# filter not samerace
df_match = df[df$samerace == "No", ]
dim(df_match)

# remove wave 6-9
wave = c(1:5,10:21)
df_match_wave = df_match[df_match$wave %in% wave,]
dim(df_match_wave)

#remove variables with more than 20% NA

f <- function(x) {
    sum(is.na(x)) < length(x) * 0.2
}

df_final = df_match_wave[, vapply(df_match_wave, f, logical(1)), drop = F]

# Find the iid of the only missing id
iid <- df_final[is.na(df_final$id),]$iid
# Assign this iid' id to the missing iid..
df_final[is.na(df_final$id),]$id <- head(df_final[df_final$iid == iid,]$id, 1)


# remove the missing pid rows because in wave 5, there are 9F and 10M but each male was recorded meeting 10 times
df_final <- df_final[complete.cases(df_final$pid), ]


#seem like we can't impute missing age
# remove rows that have missing age and age_o

df_final <- df_final[complete.cases(df_final$age), ]
df_final <- df_final[complete.cases(df_final$age_o), ]



# field_cd, career_c, race, race_o: NA = "0" for "Other"
df_final$field_cd[is.na(df_final$field_cd)] = 0
df_final$career_c[is.na(df_final$career_c)] = 0
df_final$race[is.na(df_final$race)] = 0
df_final$race_o[is.na(df_final$race_o)] = 0


dim(df_final)

sapply(df_final, function(x) sum(is.na(x)))


```
```{r}

date_columns = c("iid", "field_cd", "career_c")
date_info = subset(df_final, select = date_columns)

names(date_info) = c("pid", "field_cd_o", "career_c_o")

date_info <- subset(date_info, !duplicated(date_info[,1])) 

df_final = merge(df_final, date_info, by = "pid")

dim(df_final)


# create new indicator variables for when two people have the same field, career, age (grouping age 18-20, 20-25, 25-30, and so on)

df_final$field_same <- df_final$field_cd == df_final$field_cd_o

df_final$career_same <- df_final$career_c == df_final$career_c_o


df_final$agegroup = findInterval(df_final$age, c(20,25,30,35,40,45,50))
df_final$agegroup_o = findInterval(df_final$age_o, c(20,25,30,35,40,45,50))
df_final$age_same <- df_final$agegroup == df_final$agegroup_o

dim(df_final)
#head(df_final)


sapply(df_final, function(x) sum(is.na(x)))

```

```{r}

#drop variables not important

drops = c("iid","id","idg","condtn","wave","round","position","order",
                        "partner","pid","samerace","age_o","dec_o","like_o","prob_o","met_o",
                        "age","from","zipcode","goal","career","career_c","career_c_o","field","field_cd","field_cd_o","dec","match_es","satis_2","length","numdat_2")

data = df_final[,!(names(df_final) %in% drops)]

dim(data)

sapply(data, function(x) sum(is.na(x)))

```

```{r}
# impute missing in numeric variables and replace with the mean, by gender
              
for (i in 1:length(names(data))){
  data[,names(data)[i]] <- with(data, ave(data[,names(data)[i]], gender,
                                    FUN = function(x) replace(x, is.na(x), round(mean(x, na.rm= TRUE)))))
  
}

sapply(data, function(x) sum(is.na(x)))
dim(data)

```

```{r}

# convert categorical variables to factor

data$match <- as.factor(data$match)
data$field_same <- as.factor(data$field_same)
data$career_same <- as.factor(data$career_same)
data$age_same <- as.factor(data$age_same)

dim(data)


```

```{r}

# Find number of men/women for each race
races <- data %>%
  group_by(gender, race) %>%
  summarise(
    my.n = n()
  )

# plot race
ggplot(races, aes(x = race, y = my.n, fill = factor(gender))) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_discrete(name = "Gender") +
  xlab("Race") + ylab("Count") + ggtitle("Race repartition") +
  scale_x_continuous(labels = c("Black", "European", "Latino", "Asian", "Native","Other"), breaks = 1:6)


# Find number of men/women for each age group
age_data <- data %>%
  group_by(gender, agegroup) %>%
  summarise(
    my.n = n()
  )

# plot race
ggplot(age_data, aes(x = agegroup, y = my.n, fill = factor(gender))) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_discrete(name = "Gender") +
  xlab("Age Group") + ylab("Count") + ggtitle("Age repartition") +
  scale_x_continuous(labels = c("<20","20", "25", "30", "35", "40","45","<50"), breaks = 0:7)


```

```{r}
# drop variables that are unused
drops2 = c("race","race_o","agegroup","agegroup_o")

data = data[,!(names(data) %in% drops2)]

dim(data)

sapply(data, function(x) sum(is.na(x)))
```

```{r}
# Isolate the men/female from the dataset
df_M <- data[data$gender == "M",]
df_M = df_M[,-1]
dim(df_M)

df_F <- data[data$gender == "F",]
df_F = df_F[,-1]
dim(df_F)
#head(df_F)

```

```{r}
# plot attributes

#### run histogram for classification variables to see if there is skewness and preresentation. 

```

```{r}

categorical = c("match","field_same","career_same","age_same")
numvar = names(df_F[,!(names(df_F) %in% categorical)])

#Determining Skewness

#seems like there are skewness in the data

se <- function(x) sd(x)/sqrt(length(x))
result = NULL
print("Skewness for Female Data")
for (i in numvar){
 # twose = 
  #skew = 
  if (abs(skewness(df_F[,i])) > 2*se(df_F[,i])){
    result = "Yes"
  } else {result = "No"}
  print(sprintf("%s | 2*se = %s | skewness = %s | %s",i, 2*se(df_F[,i]), skewness(df_F[,i]), result))

}


print("Skewness for Male Data")
for (i in numvar){
 # twose = 
  #skew = 
  if (abs(skewness(df_M[,i])) > 2*se(df_M[,i])){
    result = "Yes"
  } else {result = "No"}
  print(sprintf("%s | 2*se = %s | skewness = %s | %s",i, 2*se(df_M[,i]), skewness(df_M[,i]), result))

}

```

```{r}

# Correlations plot for F
correlations_F = cor(df_F[,!(names(df_F) %in% categorical)])
corrplot::corrplot(correlations_F, type = "full")

# Correlations plot for M
correlations_M = cor(df_M[,!(names(df_M) %in% categorical)])
corrplot::corrplot(correlations_M, type = "full")


```

```{r}
#FINAL FINAL DATA

final_F = df_F
final_M = df_M

#names(final_F)

dim(final_F)
  
```

# Class Imbalance
summary(final_F$match)
#  No  Yes 
#1694  308 
summary(train_set_F$match)
#  No  Yes 
#1356  247 
summary(test_set_F$match)
#No Yes 
#338  61

#names(train_set_F)

ggplot(train_set_F, aes(factor(match), attr)) + geom_boxplot()

ggplot(train_set_F, aes(factor(match), fun_o)) + geom_boxplot()

ggplot(train_set_F, aes(factor(match), imprace)) + geom_boxplot()

ggplot() + 
  geom_bar(data = train_set_F,
           aes(x = factor(match),fill = factor(field_same)),
           position = "fill")

ggplot() + 
  geom_bar(data = train_set_F,
           aes(x = factor(match),fill = factor(age_same)),
           position = "fill")


```{r}

#final_F = df_F
#final_M = df_M

#train_set_F = final_F[trainingRows,]
#test_set_F = final_F[-trainingRows,]

#Using 10-fold CV
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# a) linear algorithms
set.seed(1)
fit.lda <- train(match~., data=train_set_F, method="lda", metric=metric, trControl=control)

#Accuracy   Kappa    
#  0.8615091  0.3270967
#No Parameter


set.seed(1)
fit.qda <- train(match~., data=train_set_F, method="qda", metric=metric, trControl=control)
#Accuracy  Kappa    
#  0.769171  0.2474415

#no parameter

# b) nonlinear algorithms
# CART (Decision Tree)
set.seed(1)
fit.cart <- train(match~., data=train_set_F, method="rpart", metric=metric, trControl=control)

#cp          Accuracy   Kappa    
#  0.01619433  0.8577472  0.3124961

# kNN
set.seed(1)
fit.knn <- train(match~., data=train_set_F, method="knn", metric=metric, trControl=control)

#k  Accuracy   Kappa 
#9  0.8415516  0.01846986

# c) advanced algorithms
# SVM
set.seed(1)
fit.svm <- train(match~., data=train_set_F, method="svmRadial", metric=metric, trControl=control)

# C     Accuracy   Kappa
#1.00  0.8596340  0.180525959

#sigma = 0.007752414 and C = 1


# Random Forest
set.seed(1)
fit.rf <- train(match~., data=train_set_F, method="rf", metric=metric, trControl=control)

#mtry  Accuracy   Kappa 
#38    0.8752435  0.3683191

#Logistic Regression
set.seed(1)
fit.glm <- train(match~., data=train_set_F, method="glm", metric=metric, trControl=control)

#Accuracy   Kappa    
#  0.8621302  0.3674785

##Using 5-fold
control20 <- trainControl(method="cv", number=20)
metric20 <- "Accuracy"

set.seed(1)
fit.glm20 <- train(match~., data=train_set_F, method="glm", metric=metric20, trControl=control20)

predictions.glm20 <- predict(fit.glm20, test_set_F)
confusionMatrix(predictions.glm20, test_set_F$match, positive = "Yes")

```

resultsF <- resamples(list(lda=fit.lda, dqa=fit.qda, glm=fit.glm, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(resultsF)
dotplot(resultsF)

```{r}
# summarize Best Model
print(fit.rf)
```

```{r}
# estimate skill of RF on the validation dataset
#RF
predictions <- predict(fit.rf, test_set_F)
confusionMatrix(predictions, test_set_F$match)

#Accuracy is: 0.8797   

#LDA
# estimate skill of LDA on the validation dataset
predictions.lda <- predict(fit.lda, test_set_F)
confusionMatrix(predictions.lda, test_set_F$match, positive = "Yes")

#Accuracy : 0.8596 


#QDA
predictions.qda <- predict(fit.qda, test_set_F)
confusionMatrix(predictions.qda, test_set_F$match, positive = "Yes")

#Accuracy : 0.7744

#LOGISCTIC 
predictions.glm <- predict(fit.glm, test_set_F)
confusionMatrix(predictions.glm, test_set_F$match, positive = "Yes")
#Accuracy : 0.8546  

#SVM
predictions.svm <- predict(fit.svm, test_set_F)
confusionMatrix(predictions.svm, test_set_F$match, positive = "Yes")
#Accuracy : 0.8672 

#KNN
predictions.knn <- predict(fit.knn, test_set_F)
confusionMatrix(predictions.knn, test_set_F$match, positive = "Yes")
#Accuracy 0.8421

#CART (Decision Tree)
predictions.cart <- predict(fit.cart, test_set_F)
confusionMatrix(predictions.cart, test_set_F$match, positive = "Yes")
#Accuracy : 0.8471

#RANDOMFOREST FOR FEMALE
```

```{r} 
#final_F = df_F
#final_M = df_M

#train_set_F = final_F[trainingRows,]
#test_set_F = final_F[-trainingRows,]

#Using 10-fold CV
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# a) linear algorithms
set.seed(1)
fit.ldaM <- train(match~., data=train_set_M, method="lda", metric=metric, trControl=control)

#Accuracy   Kappa    
#  0.8627407  0.3368038

set.seed(1)
fit.qdaM <- train(match~., data=train_set_M, method="qda", metric=metric, trControl=control)
#GLM
set.seed(1)
fit.glmM <- train(match~., data=train_set_M, method="glm", metric=metric, trControl=control)

#Accuracy   Kappa    
#  0.8602601  0.3506886

# b) nonlinear algorithms
# CART (Decision Tree)
set.seed(1)
fit.cartM<- train(match~., data=train_set_M, method="rpart", metric=metric, trControl=control)

#cp          Accuracy   Kappa
#0.02777778  0.8496623  0.08428922

#higher Kappa --> 0.01562500  0.8465295  0.24588786

# kNN
set.seed(1)
fit.knnM <- train(match~., data=train_set_M, method="knn", metric=metric, trControl=control)

#k  Accuracy   Kappa
#7  0.8490450  0.11081890

# c) advanced algorithms
# SVM
set.seed(1)
fit.svmM <- train(match~., data=train_set_M, method="svmRadial", metric=metric, trControl=control)

# C     Accuracy   Kappa
#1.00  0.8664907  0.1847549

#sigma = 0.007765363 and C = 1

# Random Forest
set.seed(1)
fit.rfM <- train(match~., data=train_set_M, method="rf", metric=metric, trControl=control)

#mtry  Accuracy   Kappa  
#38    0.8715023  0.3176589

```

# summarize accuracy of models (MALE)
resultsM <- resamples(list(lda=fit.ldaM, dqa=fit.qdaM, glm=fit.glmM, cart=fit.cartM, knn=fit.knnM, svm=fit.svmM, rf=fit.rfM))
summary(resultsM)
dotplot(resultsM)

#It looks like RF is also the most accurate for Males data

```{r}
# summarize Best Model
print(fit.rfM)
print(fit.ldaM)
```

```{r}
#final_F = df_F
#final_M = df_M

#train_set_F = final_F[trainingRows,]
#test_set_F = final_F[-trainingRows,]

train_set_M = final_M[trainingRows,]
test_set_M = final_M[-trainingRows,]


# estimate skill of RandomForest on the validation dataset
predictionsM <- predict(fit.rfM, test_set_M)
confusionMatrix(predictionsM, test_set_M$match, positive = "Yes")

#Accuracy : 0.8446 

predictionsM <- predict(fit.ldaM, test_set_M)
confusionMatrix(predictionsM, test_set_M$match, positive = "Yes")

#Accuracy: 0.8446

#LDA and RandomForest comes close

predictionsM <- predict(fit.glmM, test_set_M)
confusionMatrix(predictionsM, test_set_M$match, positive = "Yes")

#Accuracy : 0.8421  

predictionsM <- predict(fit.svmM, test_set_M)
confusionMatrix(predictionsM, test_set_M$match, positive = "Yes")
#Accuracy : 0.8371

predictionsM <- predict(fit.knnM, test_set_M)
confusionMatrix(predictionsM, test_set_M$match, positive = "Yes")
#Accuracy : 0.802

predictionsM <- predict(fit.cartM, test_set_M)
confusionMatrix(predictionsM, test_set_M$match, positive = "Yes")
#Accuracy : 0.8296 
```

```{r}
library(randomForest)

predictions <- predict(fit.rf, test_set_F)
confusionMatrix(predictions, test_set_F$match)

data_balanced_over <- ovun.sample(match ~ ., data = train_set_F, method = "over",N = 2712)$data
table(data_balanced_over$match)

#Using upsampling
set.seed(1)
rfModel.up <- randomForest(match~.,data_balanced_over, ntree = 500, importance = TRUE)

rfTestPred.up <- predict(rfModel.up, test_set_F, type = "prob")
head(rfTestPred.up)

test_set_F$RFclass.up <- predict(rfModel.up, test_set_F)

confusionMatrix(data = test_set_F$RFclass.up,
                reference = test_set_F$match,
                positive = "Yes")

#----
#using 10-fold
set.seed(1)
fit.rf.up <- train(match~., data=data_balanced_over, method="rf", metric=metric, trControl=control)
set.seed(1)
fit.rf.rose <- train(match~., data=data.rose, method="rf", metric=metric, trControl=control)

PredRFModel.up <- predict(fit.rf.up, test_set_F)
confusionMatrix(PredRFModel.up, test_set_F$match, positive = "Yes")

#Reference
#Prediction  No Yes
#       No  326  40
#       Yes  12  21
                                          
#               Accuracy : 0.8697  

#GLM

set.seed(1)
fit.glm.up <- train(match~., data=data_balanced_over, method="glm", metric=metric, trControl=control)
set.seed(1)

PredGLMModel.up <- predict(fit.glm.up, test_set_F)
confusionMatrix(PredGLMModel.up, test_set_F$match, positive = "Yes")

#Reference
#Prediction  No Yes
#       No  263  18
#       Yes  75  43
                                          
#               Accuracy : 0.7669 

#LDA
set.seed(1)
fit.lda.up <- train(match~., data=data_balanced_over, method="lda", metric=metric, trControl=control)
set.seed(1)

PredldaModel.up <- predict(fit.lda.up, test_set_F)
confusionMatrix(PredldaModel.up, test_set_F$match, positive = "Yes")

#Reference
#Prediction  No Yes
#       No  251  16
#       Yes  87  45
                                         
#               Accuracy : 0.7419

#Decision Tree
set.seed(1)
fit.cart.up <- train(match~., data=data_balanced_over, method="rpart", metric=metric, trControl=control)
set.seed(1)

PredcartModel.up <- predict(fit.cart.up, test_set_F)
confusionMatrix(PredcartModel.up, test_set_F$match, positive = "Yes")

#Reference
#Prediction  No Yes
#       No  292  24
#       Yes  46  37
                                          
#               Accuracy : 0.8246


#SVM
set.seed(1)
fit.svm.up <- train(match~., data=data_balanced_over, method="svmRadial", metric=metric, trControl=control)
set.seed(1)

PredsvmModel.up <- predict(fit.svm.up, test_set_F)
confusionMatrix(PredsvmModel.up, test_set_F$match, positive = "Yes")

#Reference
#Prediction  No Yes
#       No  288  20
#       Yes  50  41
                                          
#               Accuracy : 0.8246


set.seed(1)
fit.svmL.up <- train(match~., data=data_balanced_over, method="svmLinear", metric=metric, trControl=control)
set.seed(1)

PredsvmLModel.up <- predict(fit.svmL.up, test_set_F)
confusionMatrix(PredsvmLModel.up, test_set_F$match, positive = "Yes")

#Reference
#Prediction  No Yes
#       No  260  14
#       Yes  78  47
                                          
#               Accuracy : 0.7694 


set.seed(1)
fit.svmP.up <- train(match~., data=data_balanced_over, method="svmPoly", metric=metric, trControl=control)
set.seed(1)

PredsvmPModel.up <- predict(fit.svmP.up, test_set_F)
confusionMatrix(PredsvmPModel.up, test_set_F$match, positive = "Yes")

#Reference
#Prediction  No Yes
#       No  318  31
#       Yes  20  30
                                          
#               Accuracy : 0.8722

#KNN

set.seed(1)
fit.knn.up <- train(match~., data=data_balanced_over, method="knn", metric=metric, trControl=control)
set.seed(1)

PredknnModel.up <- predict(fit.knn.up, test_set_F)
confusionMatrix(PredknnModel.up, test_set_F$match, positive = "Yes")

#Reference
#Prediction  No Yes
#       No  190  16
#       Yes 148  45
                                          
#               Accuracy : 0.589 

#Upsampling is the choice (using Random Forest with CV)

#PredRFModel.rose <- predict(fit.rf.rose, test_set_F)
#confusionMatrix(PredRFModel.rose, test_set_F$match, positive = "Yes")

#result.roc <- roc(test_set_F$match, PredRFModel.up$match[,2]) # Draw ROC curve.
#plot(result.roc, print.thres="best", print.thres.best.method="closest.topleft")

#-------
#varImp(rfModel.up)
#varImpPlot(rfModel.up)
#varImpPlot(rfModel.up, n.var = 10)

varImp(fit.rf.up)
plot(fit.rf.up)

varimpF.up <- varImp(fit.rf.up, scale = FALSE)
plot(varimpF.up, 10)

varimpM.up <- varImp(fit.rf.upM, scale = FALSE)
plot(varimpM.up, 10)

```

```{r}
#Oversampling

data_balanced_overM <- ovun.sample(match ~ ., data = train_set_M, method = "over",N = 2726)$data
table(data_balanced_overM$match)

set.seed(1)
rfModel.upM <- randomForest(match~.,data_balanced_overM, ntree = 500, importance = TRUE)

rfTestPred.upM <- predict(rfModel.upM, test_set_M, type = "prob")
head(rfTestPred.upM)

test_set_M$RFclass.upM <- predict(rfModel.upM, test_set_M)

confusionMatrix(data = test_set_M$RFclass.upM,
                reference = test_set_M$match,
                positive = "Yes")


#Using 10-fold CV
set.seed(1)
fit.rf.upM <- train(match~., data=data_balanced_overM, method="rf", metric=metric, trControl=control)
set.seed(1)

PredRFModel.upM <- predict(fit.rf.upM, test_set_M)
confusionMatrix(PredRFModel.upM, test_set_M$match, positive = "Yes")


#LDA
set.seed(1)
fit.lda.upM <- train(match~., data=data_balanced_overM, method="lda", metric=metric, trControl=control)
set.seed(1)

PredLDAModel.upM <- predict(fit.lda.upM, test_set_M)
confusionMatrix(PredLDAModel.upM, test_set_M$match, positive = "Yes")


#GLM
set.seed(1)
fit.glm.upM <- train(match~., data=data_balanced_overM, method="glm", metric=metric, trControl=control)
set.seed(1)

PredGLMModel.upM <- predict(fit.glm.upM, test_set_M)
confusionMatrix(PredGLMModel.upM, test_set_M$match, positive = "Yes")



#Decision Tree
set.seed(1)
fit.cart.upM <- train(match~., data=data_balanced_overM, method="rpart", metric=metric, trControl=control)
set.seed(1)

PredCARTModel.upM <- predict(fit.cart.upM, test_set_M)
confusionMatrix(PredCARTModel.upM, test_set_M$match, positive = "Yes")



set.seed(1)
fit.SVM.upM <- train(match~., data=data_balanced_overM, method="svmRadial", metric=metric, trControl=control)
set.seed(1)

PredSVMModel.upM <- predict(fit.SVM.upM, test_set_M)
confusionMatrix(PredSVMModel.upM, test_set_M$match, positive = "Yes")



set.seed(1)
fit.SVML.upM <- train(match~., data=data_balanced_overM, method="svmLinear", metric=metric, trControl=control)
set.seed(1)

PredSVMLModel.upM <- predict(fit.SVML.upM, test_set_M)
confusionMatrix(PredSVMLModel.upM, test_set_M$match, positive = "Yes")



set.seed(1)
fit.SVMP.upM <- train(match~., data=data_balanced_overM, method="svmPoly", metric=metric, trControl=control)
set.seed(1)

PredSVMPModel.upM <- predict(fit.SVMP.upM, test_set_M)
confusionMatrix(PredSVMPModel.upM, test_set_M$match, positive = "Yes")



#KNN
set.seed(1)
fit.knn.upM <- train(match~., data=data_balanced_overM, method="knn", metric=metric, trControl=control)
set.seed(1)

PredknnPModel.upM <- predict(fit.knn.upM, test_set_M)
confusionMatrix(PredknnPModel.upM, test_set_M$match, positive = "Yes")


varimpF.up <- varImp(fit.rf.upM, scale = FALSE)
plot(varimpF.up, 10)

varimpM <- varImp(fit.rfM, scale = FALSE)
plot(varimpM, 10)
```
