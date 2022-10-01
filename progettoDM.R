### PROGETTO DATA MINING###

setwd("D:/data mining/ESAME")

library(readr)
AppleStore <- read_csv("D:/data mining/ESAME/AppleStore.csv")
# dataset AppleStore costituito da 7197 osservazioni e 17 variabili

str(AppleStore)
#tipologia variabili presenti nel dataset + loro descrizione (per ora le lasciamo così e non le cambiamo)
#Creiamo una nuova variabile dicotomica, che useremo poi come target. 
AppleStore$Class=ifelse(AppleStore$user_rating >= 4,1,0)
#se l'app ha più di 4 stelle allora Class=1, se no Class=0.
table(AppleStore$Class) #vediamo quante osservazioni sono sopra la soglia e quaanti sotto
prop.table(table(AppleStore$Class)) #qua otteniamo le percentuali
#eliminiamo dunque la vecchia variabile 'user_rating' con cui abbiamo generato il nostro target 'Class'
AppleStore <- AppleStore[ , -c(9)]

### CREIAMO IL DATASET DI SCORE ###
library(caret)
set.seed(107)
# create a random index for tyaking 90% of data stratified by target
Scoreindex <- createDataPartition(y = AppleStore$Class, p = .90, list = FALSE)

dataset <- AppleStore[ Scoreindex,]
score  <- AppleStore[-Scoreindex,]

score <- score[,-17]
#abbiamo così il dataset di score su cui andremo a fare le previsioni (senza la variabile target)

### VALORI MANCANTI ###
sapply(AppleStore, function(x)(sum(is.na(x))))
#non ci sono valori mancanti

### TRAIN E VALIDATION ###

library(car)
dataset$Class<-recode(dataset$Class, recodes="0='c0'; else='c1'")
str(dataset)
#trasformiamo la variabile target Class (numerica) in factor
dataset$Class=as.factor(dataset$Class)
str(dataset)
table(dataset$Class)

library(caret)
set.seed(1234)
# create a random index for tyaking 70% of data stratified by target
Trainindex <- createDataPartition(y = dataset$Class, p = .70, list = FALSE)

training <- dataset[ Trainindex,]
validation  <- dataset[-Trainindex,]
#così abbiamo diviso il dataset:
#70% delle osservazioni costituiscono il dataset di training
#30% delle osservazioni costituiscono il dataset di validation

### TREE ###

library(funModeling)
library(dplyr)

# Creiamo la tabella status per vedere meglio le variabili
status=df_status(training, print_results = F)

#eliminiamo le variabili con troppi livelli (o con un solo livello) e creiamo un nuovo dataset
training_tree <- training[ , -c(1:5)] 
#proviamo direttamente il deeper tree per vedere anche l'importanza delle variabili
library(rpart)
library(rpart.plot)
set.seed(1)
deeper.ct <- rpart(Class ~ ., data = training_tree, method = "class", cp = 0, minsplit = 1)
rpart.plot(deeper.ct)
b=data.frame(deeper.ct$frame$var)
table(b)
b1=data.frame(deeper.ct$variable.importance)
b1
#vediamo che le variabili hanno gradi di importanza notevolmente diversi, per questo provando a fare il default tree o il
#best tree il modello usa solo 1 variabile discriminante (rispettivamente: ver, user_rating_ver, ecc.)
#Eliminiamo allora le variabili ver user_rating_ver.
#Rimandiamo il deeper tree e vediamo ora che la var rating_count_tot è l'unica discriminante; la eliminiamo.
#Ora rating_count_ver ha un valore d'importanza molto più alto rispetto alle altre, ma provando a fare il best tree troviamo
#un risultato accettabile, quindi la teniamo.
#cambiamo le variabili
str(training_tree)
library(car)
training_tree$Class<-recode(training_tree$Class, recodes="c0='0'; else='1'")
training_tree$Class=as.numeric(training_tree$Class)
training_tree$ver=as.factor(training_tree$ver)
training_tree$cont_rating=as.factor(training_tree$cont_rating)
training_tree$prime_genre=as.factor(training_tree$prime_genre)
str(training_tree)
#così abbiamo solo variabili factor e numeriche
training_tree <- training_tree[ ,-c(4)]
training_tree <- training_tree[ ,-c(4)]
training_tree <- training_tree[ ,-c(2)] #
training_tree <- training_tree[ ,-c(2)] #rating_count_ver
#variabili da tenere nel training_tree: price, rating_count_ver, cont_rating, prime_genre, sup_devices_num, ipad, lang.num, vpp_lic
#leviamo le variabili user rating ver, ver e rating_count_tot, 
#perchè usate come uniche variabili discriminanti nell'albero (successivo) (una alla volta)
#Infatti nel deeper tree possiamo vedere che queste variabili hanno un'importanza molto maggiore rispetto alle altre
#e causano errori di overfitting

#Il nostro training-tree è quindi costituito da 9 variabili.

#default classification tree:
library(rpart)
library(rpart.plot)

set.seed(1234)  
default.ct <- rpart(Class ~ ., data = training_tree)
# plot tree
rpart.plot(default.ct)
rpart.plot(default.ct, type = 4, extra = 1)
rpart.plot(default.ct, type = 4, extra = 101,  split.font = 0.9, ycompress=FALSE, cex=.7)
#questo è il default tree
set.seed(1)
deeper.ct <- rpart(Class ~ ., data = training_tree, method = "class", cp = 0, minsplit = 1)
rpart.plot(deeper.ct) #illeggibile...
#questo è il deeper tree

#vediamo il default e il deeper tree a confronto (anche se poi ci converrà usare il best tree tramite il cp)
a=data.frame(default.ct$frame$var)
table(a)
b=data.frame(deeper.ct$frame$var)
table(b) #deeper.ct troppo complesso
#vediamo ora l'importanza delle variabili
b1=data.frame(deeper.ct$variable.importance)
b1
a1=data.frame(default.ct$variable.importance)
a1
#confrontiamo l'importanza delle variabili
deeper.ct$cptable
default.ct$cptable

#Facciamo il best tree
set.seed(1)
cv.ct <- rpart(Class ~ ., data = training_tree, method = "class", 
               cp = 0.00001, minsplit = 5, xval = 5)

printcp(cv.ct)
#qua abbiamo cambiato la penalizzazione (cp più alto)
#vediamo che l' xerror più basso ce l'abbiamo con cp=0.00327225, 6 split e quindi 7 foglie
set.seed(1)
pruned1 <- prune(cv.ct, 
                 cp = cv.ct$cptable[which.min(cv.ct$cptable[,"xerror"]),"CP"])

rpart.plot(pruned1, type = 4, extra = 1,   split.font = 0.9, ycompress=FALSE, cex=.7)
#questo è l'albero migliore che potremmo ottenere senza fare preprocessing (ma eliminando comunque alcune variabili,
#perchè se no l'albero ottenuto era inutile).
#Volendo possiamo rieseguire il processo del best tree eliminando pure la variabile rating_count_ver; in questo modo tramite il 
#deeper tree vediamo che tutte le variabili hanno circa la stessa importanza e quindi otteniamo un albero diviso bene da tutte 
#le variabili:
# - l'xerror più basso lo avremmo con un cp=0.00261780, 12 split, 13 foglie.


### RANDOM FOREST ###
library(funModeling)
library(dplyr)

status=df_status(training, print_results = F)
#come in TREE eliminiamo le variabili con troppi livelli (e currency che ha 1 solo livello)
training_rf <- training[ , -c(1:5)] 
training_rf <- training_rf[ , -c(2)]
training_rf <- training_rf[ , -c(4)]
#Quindi il dataset per la random forest è costituito da 10 variabili

library(caret)
seed <- 7
set.seed(seed)
metric <- "ROC"   # Kappa , AUC , Sens.....etc
control <- trainControl(method="cv", number=10, search="grid", summaryFunction = twoClassSummary, classProbs = TRUE)
tunegrid <- expand.grid(.mtry=c(1:5))
fit2 <- train(Class~., data=training_rf, method="rf", metric=metric, tuneGrid=tunegrid, ntree=250, trControl=control)
fit2
plot(fit2)
#vediamo i risultati di fit2
fit2$results
# final model and performance
getTrainPerf(fit2)
# metrics for each fold with best mtry
fit2$resample

#se vogliamo salvare le metriche
#z=data.frame(fit2$resample)
#head(z)

# cross validated confusion matrix
confusionMatrix(fit2) #Accuracy di 0.838 (non commentabile però perchè qua la soglia di default è 0.5)

#vediamo ora l'importanza delle variabili
Vimportance <- varImp(fit2)
head(Vimportance)
plot(Vimportance)

#volendo possiamo salvare le variabili più importanti in un dataframe a parte
VImP=as.data.frame(Vimportance$importance)
head(VImP)
dim(VImP)

#eliminiamo tutte le variabili con un'importanza minore del 10%
V=subset(VImP, Overall>10)
V2=t(V)
# target
y=training_rf[10] 
# select important vars
Xselected=training_rf[,colnames(V2)]
#selezioniamo nel datset solo le variabili presenti in V2
# add
training_rf=cbind(Xselected,y) #aggiungiamo le var selezionate al target
head(training_rf)
# checks
colnames(newDAta)
row.names(V)
#quindi questo datset è pronto a nuovi algoritmi per lavorarci su.

### PREPROCESSING ###

# nzv
library(funModeling)
library(dplyr)

status=df_status(training, print_results = F)
status%>%
  arrange(type, -unique)
nzv = nearZeroVar(training, saveMetrics = TRUE)
nzv
training_nzv <- training[ ,-c(5)] 
#via la variabile currency che ha zero variance
training_nzv <- training_nzv[ ,-c(15)]
#ci conviene togliere anche la var vpp_lic che ha varianza QUASI PARI a zero

# correlazioni
#se vogliamo fare preprocessing con caret dobbiamo dividere le variabili factor in più variabili numeriche
str(training)
#cambiamo le variabili chr in factor prima
library(car)
training1 <- training
training1$Class<-recode(training1$Class, recodes="c0='0'; else='1'")
training1$Class=as.numeric(training1$Class)
training1$ver=as.factor(training1$ver)
training1$cont_rating=as.factor(training1$cont_rating)
training1$prime_genre=as.factor(training1$prime_genre)
training1$track_name=as.factor(training1$track_name)
training1$currency=as.factor(training1$currency)
str(training1)

isfactor <- sapply(training1, function(x) is.factor(x))
isfactor

factordata <- training1[, isfactor]
str(factordata)

numeric <- sapply(training1, function(x) is.numeric(x))
numeric <-training1[, numeric]
str(numeric)

#Ora vediamo le collineraità
R=cor(numeric)
R
correlatedPredictors = findCorrelation(R, cutoff = 0.85, names = TRUE)
correlatedPredictors
#la variabile ID è troppo correlata con X1, ci conviene quindi eliminare una delle due
#vediamo le correlazioni anche graficamente:
library(PerformanceAnalytics)
chart.Correlation(numeric , histogram=TRUE, pch=22)

#Dall'analisi della collinearità quindi capiamo che ci conviene eliminare la variabile ID (tra le variabili numeriche)
#Decidiamo però di eliminare la variabile X1 invece (la seconda più correlata), perchè meno utile ai fini predittivi.
training_corr = training[ ,-c(1)]
#Ora passiamo all'analisi delle connessioni delle variabili factor
library(plyr)
b.table <- table(factordata$track_name,factordata$prime_genre)
numerosita=sum(b.table)
chi=chisq.test(b.table)
chi$statistic
chi$statistic/ (numerosita*min(nrow(b.table)-1,ncol(b.table)-1))
# track_name - currency : 0.999591
# track_name - ver : 0.9999901
# track_name - cont_rating : 0.9996658
# track_name - prime_genre : 1
#... PROBLEMA CON VARIABILI QUALITATIVE!!!

#trasformare le variabili factor in variabili dummy??

# trasformazione delle covariate

#proviamo a vedere se è possibile rendere più simmetriche le variabili di input
all=cbind(numeric,factordata)
head(all)
scaled_bc <- preProcess(all, method = c("scale", "BoxCox"))
scaled_bc
scaled_bc$bc #questi sono i lambda per le quattro variabili
all_bc=predict(scaled_bc, newdata = all)
head(all_bc)
#questo è il dataset con le variabili trsformate (prima standardizzazione, poi BoxCox)
#Vediamo come sono cambiate le variabili:
par(mfrow=c(4,2))
hist(all$X1)
hist(all_bc$X1)
hist(all$id)
hist(all_bc$id)
hist(all$size_bytes)
hist(all_bc$size_bytes)
hist(all$sup_devices.num)
hist(all_bc$sup_devices.num)
par(mfrow=c(1,1)) #per vedere i grafici singolarmente
#queste sono le 4 variabili trasformate: X1, id, size_bytes, sup_devices_num

#Ricontrolliamo la collinearità dopo aver trasformato le variabili:
chart.Correlation(all[,c(1:12)], histogram=TRUE, pch=22)
chart.Correlation(all_bc[,1:12] , histogram=TRUE, pch=22)
#aumenta la correlazione tra X1 e id (che tanto essendo già alta prima della trasformazione, ci portava a 
#escludere una delle due variabili)
training_transform = all_bc

#trasformazione delle covariate (metodo 2)

#proviamo a trasformare tutte le variabili factor in variabili dummy e rifare il preprocessing visto prima
#(poi per la creazione dei modelli classificativi, dovremo andare a ricreare le variabili factor con più livelli)
str(training)
library(car)
training$Class<-recode(training$Class, recodes="2='1'; else='0'")
training$Class=as.numeric(training$Class)
training$ver=as.factor(training$ver)
training$cont_rating=as.factor(training$cont_rating)
training$prime_genre=as.factor(training$prime_genre)
training$track_name=as.factor(training$track_name)
training$currency=as.factor(training$currency)
str(training)
training_dummy <- training[ ,-c(5)] #via currency perchè factor con un solo livello
training_dummy <- training_dummy[ ,-c(3)] #sarebbe inutile rendere dummy la variabile track_name, quindi la leviamo
dummies <- dummyVars(Class ~ ., data = training_dummy , fullRank = T, na.action = na.pass)
training_dummy = data.frame(predict(dummies, newdata = training_dummy))
training_dummy$Class <- training[ ,c(17)]
training_dummy$Class
#possiamo rifare i procedimenti di preprocessing su questo dataset...

# min-max
#controlliamo i boxplot delle variabili numeriche
boxplot(numeric$X1) #ok
boxplot(numeric$id) #ok
boxplot(numeric$size_bytes) #da migliorare
boxplot(numeric$price) #da migliorare
boxplot(numeric$rating_count_tot) #da migliorare
boxplot(numeric$rating_count_ver) #da migliorare
boxplot(numeric$user_rating_ver) #ok
boxplot(numeric$sup_devices.num) #non troppo male
boxplot(numeric$ipadSc_urls.num) #ok
boxplot(numeric$lang.num) #non troppo male
boxplot(numeric$vpp_lic) #???
#visto nel complesso:
boxplot(numeric)$out

numeric<- numeric[ ,-c(12)] #non consideriamo la variabile target
# ...

# optimal group
str(training)
#vogliamo ridurre i livelli delle variabili categoriali
#controlliamo i livelli delle variabili:
library(funModeling)
library(dplyr)
status=df_status(training, print_results = F)
#cerchiamo di ridurre i livelli di alcune variabili
#usiamo il metodo caret
training_opt <- training
library(rpart)
library(rpart.plot)

set.seed(1)  
grup.tree <- rpart(Class ~ factor(ver), cp=0, data = training_opt, method = "class")
grup.tree$cptable
plot(grup.tree)
#riduciamo i livelli della variabile ver
library(partykit)
z <- as.party(grup.tree)
training_opt$ver_opt=predict(z, training_opt, type = "node")
training_opt <- training_opt[ ,-c(10)]

set.seed(1)  
grup.tree <- rpart(Class ~ factor(rating_count_tot), cp=0, data = training_opt, method = "class")
grup.tree$cptable
plot(grup.tree)
#riduciamo i livelli della variabile rating_count_tot
library(partykit)
z <- as.party(grup.tree)
training_opt$rating_count_tot_opt=predict(z, training_opt, type = "node")
training_opt <- training_opt[ ,-c(7)]

#abbiamo ridotto i livelli delle variabili ver e rating_count_tot

# model selection

#Proviamo a fare model selection con il metodo BORUTA
library(Boruta)
set.seed(123)
boruta.train <- Boruta(Class~., data = training, doTrace = 1)

plot(boruta.train, xlab = "features", xaxt = "n", ylab="MDI")
#box verdi = variabili importanti
#box blu = variabili con MDI minimo
#box rossi = variabili non importanti
#box gialli = variabili al limite
print(boruta.train)
#l'unica variabile non importante risulta essere la variabile currency
#13 variabili sono importanti invece
boruta.metrics <- attStats(boruta.train)
table(boruta.metrics$decision)
training_boruta <- training[ ,-c(5)]
#eliminiamo la variabile currency


#Proviamo a fare model selection con il metodo LASSO
#ricordiamo che la lasso vuole una matrice come input, non un dataframe.
#Inoltre ci conviene eliminare le variabili non numeriche, perchè creano problemi
str(training)
x <- training[ , -c(3)]
x <- x[ , -c(4)]
x <- x[ , -c(8:10)]
str(x)
x$Class <- NULL
x <- as.matrix(x)
y <- as.numeric(training$Class) # became numeric target c1=2 e c0=1
table(y)

library(glmnet)
set.seed(11)
cv.glmnet_model <- cv.glmnet(x = x, y = y, family = "binomial")
# as glm, glmnet use as event highest y value: y=2=good
print(cv.glmnet_model)
plot(cv.glmnet_model)

head(cv.glmnet_model$glmnet.fit$beta)
dim(cv.glmnet_model$glmnet.fit$beta)

# lambda tuned
# this has minumum cv error
cv.glmnet_model$lambda.min
coef(cv.glmnet_model, s = "lambda.min") #qui leva solo la variabile id

# this has minumum cv error+1std.deviation(cv.error)
cv.glmnet_model$lambda.1se
coef(cv.glmnet_model, s = "lambda.1se") #qui tiene X1 , user_rating_ver , ipadSc_urls.num
# lambda.min, lambda.1se  best tuned lambda for penalization 

training_lasso_min <- training[ ,-c(2)]
training_lasso_1se <- training[ ,-c(2:8, 10:13, 15:16)]

# find predicted prob to be 2=good on train data x
predProb <- predict(cv.glmnet_model,   newx = x, type = "response")
head(predProb)

predProb1 <- predict(cv.glmnet_model,   newx = x, s = "lambda.1se", type = "response")
head(predProb1)
# predicted target on training data (x)
cv.glmnet_prediction_v <- predict(cv.glmnet_model, x, type="class")
head(cv.glmnet_prediction_v)

# confusion matrix of glmnet on training data (x)
table(y,cv.glmnet_prediction_v) #accuracy di 0.8062169

EP_good=predProb*645+234*(1-predProb) 
EP_bad=0

decision=ifelse(EP_good>0, 2,1)

decision_table=table(y,decision)
decision_table

### RANDOM FOREST SU TRAINING_OPT ### 
library(funModeling)
library(dplyr)

status2=df_status(training_opt, print_results = F)
#come in TREE eliminiamo le variabili con troppi livelli (e currency che ha 1 solo livello)
training_rf2 <- training_opt[ , -c(1:5)] 
str(training_rf2)
training_rf2$ver_opt <- as.numeric(training_rf2$ver_opt)
training_rf2$rating_count_tot_opt <- as.numeric(training_rf2$rating_count_tot_opt)
training_rf2$Class <- as.factor(training_rf2$Class)
training_rf2$Class <- recode_factor(training_rf2$Class, '1'='c0', '2'='c1')

set.seed(7)
metric <- "ROC"   # Kappa , AUC , Sens.....etc
control <- trainControl(method="cv", number=10, search="grid", summaryFunction = twoClassSummary, classProbs = TRUE)
tunegrid <- expand.grid(.mtry=c(1:5))
fit3 <- train(Class~., data=training_rf2, method="rf", metric=metric, tuneGrid=tunegrid, ntree=250, trControl=control)
fit3
plot(fit3)
#vediamo i risultati di fit2
fit3$results
# final model and performance
getTrainPerf(fit3)
# metrics for each fold with best mtry
fit3$resample

# cross validated confusion matrix
confusionMatrix(fit3) #Accuracy di 0.9224 (non commentabile però perchè qua la soglia di default è 0.5)

#vediamo ora l'importanza delle variabili
Vimportance2 <- varImp(fit3)
head(Vimportance2)
plot(Vimportance2)

#volendo possiamo salvare le variabili più importanti in un dataframe a parte
VImP2=as.data.frame(Vimportance2$importance)
head(VImP2)
dim(VImP2)

#eliminiamo tutte le variabili con un'importanza minore del 10%
V=subset(VImP2, Overall>10)
V2=t(V)
# target
y=training_rf2[10] 
# select important vars
Xselected=training_rf2[,colnames(V2)]
#selezioniamo nel datset solo le variabili presenti in V2
# add
training_rf2=cbind(Xselected,y) #aggiungiamo le var selezionate al target
head(training_rf2)
#quindi questo datset è pronto a nuov

## DATASET CON PREPROCESSING COMPLETO ##

# Il dataset che viene creato in questa sezione tiene conto dell'optimal grouping, della
# trasformazione delle variabili, della nzv e della collinearità.

training_pre_completo=training_opt
# Tolgo X1, currency, vpp_lic (per preprocessing nzv e correlazioni)
training_pre_completo=training_pre_completo[,-c(1,5,14)]

# Trasformo le variabili character in factor, e le variabili opt. grouping da character a num
training_pre_completo$track_name=as.factor(training_pre_completo$track_name)
training_pre_completo$cont_rating=as.factor(training_pre_completo$cont_rating)
training_pre_completo$prime_genre=as.factor(training_pre_completo$prime_genre)
training_pre_completo$ver_opt=as.numeric(training_pre_completo$ver_opt)
training_pre_completo$rating_count_tot_opt=as.numeric(training_pre_completo$rating_count_tot_opt)


isfactor <- sapply(training_pre_completo, function(x) is.factor(x))
isfactor

factordata <- training_pre_completo[, isfactor]
str(factordata)

numeric <- sapply(training_pre_completo, function(x) is.numeric(x))
numeric <-training_pre_completo[, numeric]
str(numeric)

numeric_pre_completo=numeric[,-1]

all=cbind(numeric,factordata)
head(all,n=10)

head(all)

# decide to scale data (x/sigma) and do boxcox for all numeric x
scaled_bc <- preProcess(all, method = c("scale", "BoxCox"))

scaled_bc
# var transformed with  a lambda
scaled_bc$bc

# add scaled data to dataset
all_bc=predict(scaled_bc, newdata = all)
head(all_bc)


# effects of the transformations
par(mfrow=c(2,2))
hist(all$id)
hist(all_bc$id)
hist(all$size_bytes)
hist(all_bc$size_bytes)
hist(all$sup_devices.num)
hist(all_bc$sup_devices.num)
hist(all$ver_opt)
hist(all_bc$ver_opt)
hist(all$rating_count_tot_opt)
hist(all_bc$rating_count_tot_opt)
par(mfrow=c(1,1))

### NB ###

str(training) #la variabile class deve essere factor
training$Class <- as.factor(training$Class) 
#proviamo a lanciare il modello senza fare alcun preprocessing (sappiamo però che non ci sono missing values)
library(klaR)                                             
naive_all1 <- NaiveBayes(Class ~ ., data = training, usekernel = FALSE)
#non funziona...
x = training[,-17]
y = training$Class
library(e1071)
naive_all1 = train(x,y,'nb',trControl=trainControl(method='cv',number=10))
naive_all1
#risultati strani... prima ci conviene risolvere i problemi di preprocessing per il NB: missing, nzv, collinearità

# NB-nzv
str(training_nzv)
library(klaR)                                             
naive_all1 <- NaiveBayes(Class ~ ., data = training_nzv, usekernel = FALSE)
naive_all1$apriori  
naive_all1$tables$price #una variabile a caso come esempio
naive_all1$tables #tutte le variabili

# NB-collinearità
str(training_corr)
#dobbiamo prima ritrasformare la variabile Class in factor
training_corr$Class=as.factor(training_corr$Class)
str(training_corr)
training_corr <- training_corr[ ,-c(4)] #via la variabile currency
naive_all1 <- NaiveBayes(Class ~ ., data = training_corr, usekernel = FALSE)
naive_all1$tables
#levando la variabile currency funziona, se no no

#NB-covariate trasformate
str(training_transform)
#trasformiamo la variabile Class in factor
training_transform$Class=as.factor(training_transform$Class)
naive_all1 <- NaiveBayes(Class ~ ., data = training_transform, usekernel = FALSE)

#NB-rf model selection
str(training_rf)
#trasformiamo la variabile Class in factor
naive_all1 <- NaiveBayes(Class ~ ., data = training_rf, usekernel = FALSE)
naive_all1$tables

pred <- predict(naive_all1, training_rf, type="class")
# predicted y
head(pred$class)
# predicted probs
head(pred$posterior)

# confusion matrix on train
table(pred=pred$class, true=training_rf$Class)/nrow(training_rf)
# Incredibly not bad..but model can overfit on other data!!!!!


### PROVA CON PIU' MODELLI INSIEME ###

#Proviamo a comparare i vari modelli usando il dataset ottenuto facendo model selection con la random forest
set.seed(1234)
library(caretEnsemble)
classifiers<-c("nnet","regLogistic","LogitBoost","nb","lda","knn","kknn","rf","C5.0","glm") #possiamo decidere quali modelli lanciare qui (qua non conosciamo kknn e c5.0, gli altri sì)
models<-caretList(x=training_rf[,1:4],
                  y=training_rf[,5],
                  trControl=trainControl(method="cv",number=10,classProbs=TRUE,summaryFunction=twoClassSummary),
                  metric="ROC",methodList=classifiers)

class(models)
# it's list
ls(models)
#i risultati:
results <- resamples(models)
results
#vediamo graficamente i risultati dei vari modelli per compararli
bwplot(results)


#Proviamo a fare la stessa cosa usando il dataset training_rf2 ------> Mattia


#Proviamo a comparare i vari modelli usando il dataset ottenuto facendo model selection con il preprocessing completo
str(all_bc)
#trasformiamo tutte le variabili in numeriche
all_bc$cont_rating <- as.numeric(all_bc$cont_rating)
all_bc$prime_genre <- as.numeric(all_bc$prime_genre)
#leviamo le variabili con troppi livelli...
all_bc <- all_bc[ ,-c(11)]
all_bc <- all_bc[ ,-c(1)]
all_bc <- all_bc[ ,-c(1)]

set.seed(1234)
library(caret)
library(caretEnsemble)
classifiers<-c("nnet","regLogistic","LogitBoost","nb","lda","knn","kknn","rf","C5.0","glm") #possiamo decidere quali modelli lanciare qui 
models<-caretList(x=all_bc[,1:10],
                  y=all_bc[,11],
                  trControl=trainControl(method="cv",number=10,classProbs=TRUE,summaryFunction=twoClassSummary),
                  metric="ROC",methodList=classifiers)

class(models)
# it's list
ls(models)
#i risultati:
results <- resamples(models)
results
#vediamo graficamente i risultati dei vari modelli per compararli
bwplot(results)


## RANDOM FOREST:
library(caret)
set.seed(1234)
metric <- "AUC"   # Kappa , AUC , Sens.....etc
control <- trainControl(method="cv", number=10, search="grid", summaryFunction = twoClassSummary, classProbs = TRUE)
#la summaryFunction serve per disattivare il default (l'accuracy) e attivare la nuova metrica specificata prima
tunegrid <- expand.grid(.mtry=c(1:5))
fit4 <- train(Class~., data=all_bc, method="rf", metric=metric, tuneGrid=tunegrid, ntree=250, trControl=control)

fit4
plot(fit4)
fit4$results
getTrainPerf(fit4)
mean(fit4$resample$ROC)
confusionMatrix(fit4)
Vimportance4 <- varImp(fit4) #salviamo l'oggetto come lista
head(Vimportance4)
plot(Vimportance4)




## NEURAL NETWORKS:
library(caret)
set.seed(7)
metric <- "Accuracy"
ctrl = trainControl(method="cv", number=5, search = "grid")
# remember THIS: insted of using  y ~ ??????. data=train
# if you use    x, y, from x group remove the target!!!! 
nnetFit_glm <- train(training_rf2[-6], training_rf2$Class,
                     method = "nnet",
                     preProcess = "range", 
                     metric=metric, trControl=ctrl,
                     trace = TRUE, # use true to see convergence
                     maxit = 300)

print(nnetFit_glm)
plot(nnetFit_glm)
getTrainPerf(nnetFit_glm) 

# cross vaidated confusion matrix
confusionMatrix(nnetFit_glm)  #Accuracy di 0.92

#Proviamo a vedere come funziona lo stesso modello sul dataset training_rf (ovvero senza optimal group)
set.seed(7)
metric <- "Accuracy"
ctrl = trainControl(method="cv", number=5, classProbs = TRUE, search = "grid")
nnetFit_def <- train(training_rf[-5], training_rf$Class,
                     method = "nnet",
                     preProcess = "range", 
                     metric=metric, trControl=ctrl,
                     trace = TRUE, # use true to see convergence
                     maxit = 300)

print(nnetFit_def)
plot(nnetFit_def)

confusionMatrix(nnetFit_def) #Accuracy di 0.84, più bassa rispetto a prima













