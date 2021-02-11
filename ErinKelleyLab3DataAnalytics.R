library(ggplot2)


EPI_data <- read.csv("2010EPI_data.csv")
View(EPI_data)
attach(EPI_data) 	# sets the 'default' object
fix(EPI_data) 	# launches a simple data editor - test it!
EPI <- EPI_data$EPI		# prints out values EPI_data$EPI
View(EPI)
View(EPI)
tf <- is.na(EPI) 
EPI <- EPI[!tf]
View(EPI)
DALY <- EPI_data$DALY	# prints out values EPI_data$EPI
View(DALY)
tf1 <- is.na(DALY) 
DALY <- DALY[!tf1]
View(DALY)
BIODIVERSITY <- EPI_data$BIODIVERSITY	# prints out values EPI_data$EPI
View(BIODIVERSITY)
tf2 <- is.na(BIODIVERSITY) 
BIODIVERSITY <- BIODIVERSITY[!tf2]
View(BIODIVERSITY)




plot(ecdf(EPI), do.points=FALSE, verticals=TRUE)
plot(ecdf(EPI), do.points=TRUE, verticals=TRUE)
par(pty="s")
qqnorm(EPI); qqline(EPI)
x<-seq(30,95,1)
qqplot(qt(ppoints(250), df = 5), x, xlab = "Q-Q plot for tdsn")
qqline(x)
x2 <-seq(30,95,2)
x2
x2 <-seq(30,96,2)
x2
qqplot(qt(ppoints(250),df=5),x, xlab = "Q-Q plot")
qqline(x)


plot(ecdf(DALY), do.points=FALSE, verticals=TRUE)
plot(ecdf(DALY), do.points=TRUE, verticals=TRUE)
par(pty="s")
qqnorm(DALY); qqline(DALY)
x<-seq(30,95,1)
qqplot(qt(ppoints(250), df = 5), x, xlab = "Q-Q plot for tdsn")
qqline(x)
x2 <-seq(30,95,2)
x2
x2 <-seq(30,96,2)
x2
qqplot(qt(ppoints(250),df=5),x2, xlab = "Q-Q plot")
qqline(x2)



plot(ecdf(BIODIVERSITY), do.points=FALSE, verticals=TRUE)
plot(ecdf(BIODIVERSITY), do.points=TRUE, verticals=TRUE)
par(pty="s")
qqnorm(BIODIVERSITY); qqline(BIODIVERSITY)
x<-seq(30,95,1)
qqplot(qt(ppoints(250), df = 5), x, xlab = "Q-Q plot for tdsn")
qqline(x)
x2 <-seq(30,95,2)
x2
x2 <-seq(30,96,2)
x2
qqplot(qt(ppoints(250),df=5),x2, xlab = "Q-Q plot")
qqline(x2)


qqplot(EPI,DALY)
boxplot(EPI_data$EPI,EPI_data$DALY)

qqplot(EPI,BIODIVERSITY)

boxplot(EPI_data$EPI,EPI_data$BIODIVERSITY)

boxplot(EPI_data$DALY,EPI_data$BIODIVERSITY)





read.csv("~/Documents/teaching/DataAnalytics/
data/multivariate.csv")
attach(multivariate)
mm<-lm(Homeowners~Immigrant)
# Multivariate Regression
multivariate <-read.csv("~/Downloads/multivariate.csv")
head(multivariate)
attach(multivariate)
help(lm)
mm <-lm(Homeowners~Immigrant)
mm # mm here is a R object.
summary(mm)$coef # The output above shows the estimate of the regression beta




