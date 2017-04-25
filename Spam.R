#Download Data Files:
#spamdata.csv:  
#spamnames.csv: 

#Load the two files into R:
spamdata<- read.csv("spamdata.csv",header=FALSE,sep=";")
spamnames<- read.csv("spamnames.csv",header=FALSE,sep=";")

#Set the names of the dataset dataframe:

names(spamdata) <- sapply((1:nrow(spamnames)),function(i) toString(spamnames[i,1]))


#make column y a factor variable for binary classification (spam or non-spam)
spamdata$y <- factor(spamdata$y)


#get a sample of 1000 rows
sample <- spamdata[sample(nrow(spamdata), 1000),]


#Set up the packages:

install.packages("caret", dependencies = c("Depends", "Suggests"))

require(caret)

install.packages("kernlab", dependencies = c("Depends", "Suggests"))

require(kernlab)

install.packages("doMC", dependencies = c("Depends", "Suggests"))

require(doParallel)


#Split the data in trainData and testData
trainIndex <- createDataPartition(sample$y, p = .8, list = FALSE, times = 1)
trainData <- sample[ trainIndex,]
testData <- sample[-trainIndex,]

#set up multicore environment
registerDoParallel(cores=5)


#Create the SVM model:

### finding optimal value of a tuning parameter
sigDist <- sigest(y ~ ., data = trainData, frac = 1)
### creating a grid of two tuning parameters, .sigma comes from the earlier line. we are trying to find best value of .C
svmTuneGrid <- data.frame(.sigma = sigDist[1], .C = 2^(-2:7))

x <- train(y ~ .,
           data = trainData,
           method = "",
           preProc = c("center", "scale"),
           tuneGrid = svmTuneGrid,
           trControl = trainControl(method = "repeatedcv", repeats = 5, classProbs =  FALSE))

#Evaluate the model
predict_spam <- predict(x,testData[,1:57])

acc <- confusionMatrix(predict_spam, testData$y)

write.csv(predict_spam, file = "Result.csv")
