
#install.packages("MASS")

library(MASS)
attach(Boston)
#install.packages("ISLR") # installing the ISLR package
library(ISLR)


head(Boston)  #show head of dataset
dim(Boston)  #show dimensions of dataset
names(Boston) #column names
str(Boston)  #shows structure of the dataset 
nrow(Boston) #function shows the number of years 
ncol(Boston)  #function shows number of columns
summary(Boston) #shows summary of the statistics
#summary(Boston$crim) #shows summary of crime statistics in Boston


data(Auto)
head(Auto)
names(Auto)
summary(Auto)
summary(Auto$mpg)
fivenum(Auto$mpg)
boxplot(Auto$mpg)
hist(Auto$mpg)
summary(Auto$horsepower)
summary(Auto$weight)
fivenum(Auto$weight)
boxplot(Auto$weight)
mean(Auto$weight)
median((Auto$weight))

#EPI  <- read.csv('2010EPI_data.csv')
 
EPI <- read.csv(file.choose(), header = T)
summary(EPI)
fivenum(EPI,na.rm = TRUE)
boxplot(EPI)

hist(EPI)