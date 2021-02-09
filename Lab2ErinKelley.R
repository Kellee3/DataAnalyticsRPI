#install.packages("ggplot2")
library(ggplot2)

#data1 <-read.csv("GPW3_GRUMP_SummaryInformation_2010.csv")

#head(data1)  #show head of dataset
#dim(data1)  #show dimensions of dataset
#names(data1) #column names
#str(data1)  #shows structure of the dataset 
#nrow(data1) #function shows the number of years 
#ncol(data1)  #function shows number of columns
#summary(data1)

#ggplot(data = data1, mapping = aes(x = CountryEnglish, y = PopulationPerUnit)) + geom_point()


#ggplot(data = data1, mapping = aes(x =CountryEnglish), stat="count") + geom_histogram(binwidth = 0.25) 
#data()
#help("data")


EPI_data <- read.csv("2010EPI_data.csv")
View(EPI_data)
attach(EPI_data) 	# sets the 'default' object
fix(EPI_data) 	# launches a simple data editor - test it!
EPI <- EPI_data$EPI		# prints out values EPI_data$EPI
View(EPI)
tf <- is.na(EPI) 

E <- EPI[!tf]


View(E) 	# stats
summary(EPI)
fivenum(EPI)
View(EPI)

stem(EPI)		 # stem and leaf plot
hist(EPI)
hist(EPI, seq(30., 95., 1.0), prob=TRUE)
lines(density(EPI,na.rm=TRUE,bw=1.)) # or try bw="SJ"
rug(EPI) 
plot(ecdf(EPI), do.points=FALSE, verticals=TRUE) 

par(pty="s") 
qqnorm(EPI); qqline(EPI)
x<-seq(30,95,1)
qqplot(qt(ppoints(250), df = 5), x, xlab = "Q-Q plot for t dsn")
qqline(x)


DALY <- EPI_data$DALY	# prints out values EPI_data$EPI
View(DALY)
tf <- is.na(DALY) 

DALY <- DALY[!tf]


View(DALY) 	# stats
summary(DALY)
fivenum(DALY)
View(DALY)

stem(DALY)		 # stem and leaf plot
hist(DALY)
hist(DALY, seq(30., 95., 1.0), prob=TRUE)
lines(density(DALY,na.rm=TRUE,bw=1.)) # or try bw="SJ"
rug(DALY) 
plot(ecdf(DALY), do.points=FALSE, verticals=TRUE) 

par(pty="s") 
qqnorm(DALY); qqline(DALY)
q<-seq(30,95,1)
qqplot(qt(ppoints(250), df = 5), q, xlab = "Q-Q plot for t dsn")
qqline(q)

boxplot(EPI,DALY) 
qqplot(EPI,DALY)


EPILand<- EPI[!Landlock]
View(EPILand)
EPILand <- EPILand$EPI
Eland <- EPILand[!is.na(EPILand)]
hist(ELand)
hist(ELand, seq(30., 95., 1.0), prob=TRUE)


Eland <- EPILand[!is.na(EPILand)]
hist(ELand)
hist(ELand, seq(30., 95., 1.0), prob=TRUE)
