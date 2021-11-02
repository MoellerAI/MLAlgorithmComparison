#####################################
######## Logistic Regression ########
#####################################

wine <- read.table(file.choose(),header=TRUE)
wine_test <- read.table(file.choose(),header=TRUE)

model <- glm(factor(Class) ~., family='binomial', wine)
summary(model)
logLik(model)

model_reduced <- glm(factor(Class) ~ Fixed_acidity + Volatile_acidity + 
                       Residual_sugar + Chlorides + Density + 
                       pH + Sulphates, family='binomial', wine)

summary(model_reduced)
logLik(model_reduced)

####### reduced model ####### 
# testing data
probabilitiesLR <- predict(model_reduced, wine_test[,c(1,2,4,5,8,9,10)], type='response')
predictionLR <- ifelse(probabilitiesLR > 0.5, 'good', 'bad')
classificationtable <- table(predictionLR,wine_test[,12])
acctesttree <- sum(diag(classificationtable))/sum(classificationtable)
acctesttree 

# training data
probabilitiesLR <- predict(model_reduced, wine[,c(1,2,4,5,8,9,10)], type='response')
predictionLR <- ifelse(probabilitiesLR > 0.5, 'good', 'bad')
classificationtable <- table(predictionLR,wine[,12])
acctesttree <- sum(diag(classificationtable))/sum(classificationtable)
acctesttree 

#######  non-reduced model ####### 
# testing data
probabilitiesLR <- predict(model, wine_test[,-12], type='response')
predictionLR <- ifelse(probabilitiesLR > 0.5, 'good', 'bad')
classificationtable <- table(predictionLR,wine_test[,12])
acctesttree <- sum(diag(classificationtable))/sum(classificationtable)
acctesttree 

# training data
probabilitiesLR <- predict(model, wine[,-12], type='response')
predictionLR <- ifelse(probabilitiesLR > 0.5, 'good', 'bad')
classificationtable <- table(predictionLR,wine[,12])
acctesttree <- sum(diag(classificationtable))/sum(classificationtable)
acctesttree 

# 

# MDS
wholesale <- read.table(file.choose(),header=TRUE)
mtcars <- wholesale

minimum <- apply(mtcars,2,min) 
maximum <- apply(mtcars,2,max)
mtcarsNORM <- scale(mtcars,center=minimum,scale=(maximum-minimum)) 
mymtcarsNORM <- as.data.frame(mtcarsNORM) 
summary(mymtcarsNORM)
with(mymtcarsNORM, pairs(mymtcarsNORM))
myMDS <- cmdscale(dist(mymtcarsNORM), 2, eig=TRUE)
x <- myMDS$points[,1] 
y <- myMDS$points[,2]
plot(x,   y,   xlab="Representative's   Coordinate   1",   ylab="Representative's   Coordinate   2", main="MDS")
text(x, y, labels=row.names(mymtcarsNORM), cex = 0.7)

myMDSManhattan <- cmdscale(dist(mymtcarsNORM,method= "manhattan"), 2, eig=TRUE) 
x <- myMDSManhattan$points[,1]
y <- myMDSManhattan$points[,2]
plot(x,   y,   xlab="Repres  ntative's   Coordinate   1",   ylab="Representative's   Coordinate   2",main="MDSManhattan")

