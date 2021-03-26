# End of Term Assignment by A00288617
# Analysis of the "sobar-72.csv" using Decision Trees, Knn and Kmeans machine learning algorithms.
###############################################################################################################
# Decision Trees
###############################################################################################################

# install.packages("C50")
# install.packages("gmodels")
library(C50)
library(gmodels)

# Read data from the data set

cervix_cancer = read.csv("data/sobar-72.csv")
# Set the class as a factor
cervix_cancer$ca_cervix <- factor(cervix_cancer$ca_cervix, levels = c(1, 0),
                                  labels = c("Positive", "Negative"))
summary(cervix_cancer)

str(cervix_cancer)

# looking at the characteristics of the subject # check the histogram, summary and table
table(cervix_cancer$behavior_sexualRisk)
table(cervix_cancer$behavior_eating)
table(cervix_cancer$behavior_personalHygine)
table(cervix_cancer$intention_aggregation)
table(cervix_cancer$intention_commitment)
table(cervix_cancer$attitude_consistency)
table(cervix_cancer$attitude_spontaneity)
table(cervix_cancer$norm_significantPerson)
table(cervix_cancer$norm_fulfillment)
table(cervix_cancer$perception_vulnerability)
table(cervix_cancer$perception_severity)
table(cervix_cancer$motivation_strength)
table(cervix_cancer$motivation_willingness)
table(cervix_cancer$socialSupport_emotionality)
table(cervix_cancer$socialSupport_appreciation)
table(cervix_cancer$socialSupport_instrumental)
table(cervix_cancer$empowerment_knowledge)
table(cervix_cancer$empowerment_abilities)
table(cervix_cancer$empowerment_desires)

# look at the class variable
table(cervix_cancer$ca_cervix)

##############################################################################################################
# Training and Test Split Creating a random variable for the test and train data

set.seed(1)
cervix_cancer_rand <- cervix_cancer[order(runif(72)), ]

cervix_cancer_train <- cervix_cancer_rand[1:50, ]
cervix_cancer_test  <- cervix_cancer_rand[51:72, ]

# check the proportion of class variable
prop.table(table(cervix_cancer$ca_cervix))
prop.table(table(cervix_cancer$ca_cervix))

###############################################################################################################
## Building the model

library(C50)
model <- C5.0(ca_cervix ~ ., data = cervix_cancer_train)

model
model <- C5.0(ca_cervix ~ ., data = cervix_cancer_train, trials = 100)

# Tying to boost the model
# 
# model <- C5.0(ca_cervix ~ ., data = cervix_cancer_train, trials = 10)
# summary(model)
# model <- C5.0(ca_cervix ~ ., data = cervix_cancer_train, trials = 100)
# summary(model)
# There has been no boost to the model
# Plotting the model
plot(model)

# Detailed information about the model tree
summary(model)

###############################################################################################################
# Evaluating the model
predictions <- predict(model, cervix_cancer_test)

CrossTable(predictions, cervix_cancer_test$ca_cervix,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('predicted default', 'actual default'))

cm = table(predictions)
sum(diag(cm))/ sum(cm) # 1


##################################################################################################
# KNN
##################################################################################################

library(gmodels)
library(class)

# Using the same data set as previously used in Decision Trees
cervix_cancer

# Table of cervix cancer
table(cervix_cancer$ca_cervix)
class(cervix_cancer$ca_cervix)

# Having the Table of Proportions of the factor checked
prop.table(table(cervix_cancer$ca_cervix))

# Summarize the numeric feautures
summary(cervix_cancer[c("behavior_sexualRisk", "behavior_eating", 
                        "behavior_personalHygine", "intention_aggregation",
                        "intention_commitment", "attitude_consistency",
                        "attitude_spontaneity", "norm_significantPerson",
                        "norm_fulfillment", "perception_vulnerability",
                        "perception_severity", "motivation_strength",
                        "motivation_willingness", "socialSupport_emotionality",
                        "socialSupport_appreciation", "socialSupport_instrumental",
                        "empowerment_knowledge", "empowerment_abilities",
                        "empowerment_desires")])

# creation of the normalization function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Normalizing the data
cervix_cancer_n = as.data.frame(lapply(cervix_cancer[1:19], normalize))

# To check if the data has been normalized
summary(cervix_cancer_n$attitude_consistency)
hist(cervix_cancer_n$attitude_consistency)
# Data has been normalized

# Create the training and test data
cervix_cancer_knn_train = cervix_cancer_n[1:50,]
cervix_cancer_knn_test = cervix_cancer_n[51:72,]

# Create labels for the training and testing data
cacervix_train_label = cervix_cancer[1:50,20]
cacervix_test_label = cervix_cancer[51:72,20]

NROW(cacervix_train_label)

##############################################################################################
# Predictions
# k = 7
knn1 <- knn(train = cervix_cancer_knn_train, test = cervix_cancer_knn_test, 
            cl = cacervix_train_label, k=7)

CrossTable(knn1, cacervix_test_label, 
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE)
cm = table(knn1, cacervix_test_label)
sum(diag(cm))/sum(cm) # 0.7272727

# Trying to Improve the model's performance through scaling
cervix_cancer_z <- as.data.frame(scale(cervix_cancer_n[-1]))

# To check if the data has been normalized
summary(cervix_cancer_z$attitude_consistency)

# ReCreate the training and test data
cervix_cancer_knn_train2 = cervix_cancer_z[1:50,]
cervix_cancer_knn_test2 = cervix_cancer_z[51:72,]

knn2 <- knn(train = cervix_cancer_knn_train2, test = cervix_cancer_knn_test2, 
            cl = cacervix_train_label, k=7)

CrossTable(knn2, cacervix_test_label, 
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE)
cm = table(knn2, cacervix_test_label)
sum(diag(cm))/sum(cm) # 0.7272727
# Model was not improved after scaling


# Testing with different values of K 

# k = 1
knn3 <- knn(train = cervix_cancer_knn_train, test = cervix_cancer_knn_test, 
            cl = cacervix_train_label, k=1)

CrossTable(knn3, cacervix_test_label, 
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE)
cm = table(knn3, cacervix_test_label)
sum(diag(cm))/sum(cm) # Accuracy = 0.9090909

# k = 2
knn4 <- knn(train = cervix_cancer_knn_train, test = cervix_cancer_knn_test, 
            cl = cacervix_train_label, k=2)

CrossTable(knn4, cacervix_test_label, 
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE)
cm = table(knn3, cacervix_test_label)
sum(diag(cm))/sum(cm) # Accuracy = 0.8636364

# k = 3
knn5 <- knn(train = cervix_cancer_knn_train, test = cervix_cancer_knn_test, 
            cl = cacervix_train_label, k=3)

CrossTable(knn3, cacervix_test_label, 
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE)
cm = table(knn5, cacervix_test_label)
sum(diag(cm))/sum(cm) # Accuracy = 0.7727273

# k = 5
knn6 <- knn(train = cervix_cancer_knn_train, test = cervix_cancer_knn_test, 
            cl = cacervix_train_label, k=5)

CrossTable(knn3, cacervix_test_label, 
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE)
cm = table(knn6, cacervix_test_label)
sum(diag(cm))/sum(cm) # Accuracy = 0.7272727

# k = 15
knn7 <- knn(train = cervix_cancer_knn_train, test = cervix_cancer_knn_test, 
            cl = cacervix_train_label, k=15)

CrossTable(knn3, cacervix_test_label, 
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE)
cm = table(knn7, cacervix_test_label)
sum(diag(cm))/sum(cm) # Accuracy = 0.6818182

# k=1 has the best accuracy


##################################################################################################
# KMEANS
##################################################################################################

library("foreign")

cervix_cancer = read.csv("data/sobar-72.csv")
# Removing the classes
cervix_cancer_kmeans = cervix_cancer
cervix_cancer_kmeans$ca_cervix = NULL

# Nomrmalization of the data
cervix_cancer_kmeans1 = as.data.frame(lapply(cervix_cancer_kmeans[1:19], normalize))

# Using the first instance of each class as the k centers
kcenters = cervix_cancer_kmeans1[c(1,22),]

kmeans_model = kmeans(cervix_cancer_kmeans1, kcenters)

kmeans_model
kmeans_model$cluster
kmeans_model$tot.withinss
kmeans_model$centers

table(cervix_cancer$ca_cervix, kmeans_model$cluster)

# Plotting the cluster graph using 2 elements from the data frame
plot(cervix_cancer_kmeans1[c("empowerment_abilities", "empowerment_desires")], col = kmeans_model$cluster)
# plotting the cluster centers
points(kmeans_model$centers[,c("empowerment_abilities", "empowerment_desires")], col = 1:3, pch = 8, cex=2)

kmeans_model$withinss

# Using a different Approach
# Using the set see function
set.seed(5)

kmeans_model1 = kmeans(cervix_cancer_kmeans1, 2)

kmeans_model1
kmeans_model1$cluster
kmeans_model1$tot.withinss
kmeans_model1$centers

table(cervix_cancer$ca_cervix, kmeans_model1$cluster)

# Plotting the cluster graph using 2 elements from the data frame
plot(cervix_cancer_kmeans1[c("empowerment_abilities", "empowerment_desires")], col = kmeans_model1$cluster)
# plotting the cluster centers
points(kmeans_model1$centers[,c("empowerment_abilities", "empowerment_desires")], col = 1:3, pch = 8, cex=2)

kmeans_model1$withinss

