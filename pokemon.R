library(ggplot2)
library(dplyr)
library(grid)
library(gridExtra)
library(caret)
library(caTools)
library(randomForest)
library(gbm)
library(tidyr)
set.seed(123)


combat <- read.csv('combats.csv')
tests <- read.csv('tests.csv')
pokemon <- read.csv('pokemon.csv')
types <- read.csv('types.csv', row.names = 1)

#Rename columns correctly

colnames(pokemon) <- c( "ID" , "Name"    ,   "Type1"   ,  "Type2"  ,   "HP"    ,     "Attack"    ,
                        "Defense"  ,  "Sp.Atk"   , "Sp.Def"  ,  "Speed"   ,   "Generation",
                        "Legendary" )

pokemon$Type1 <- as.character(pokemon$Type1)
pokemon$Type2 <- as.character(pokemon$Type2)
pokemon$Type2[pokemon$Type2 == ''] <- 'None'


#Visualizations

ggplot(pokemon, aes(x=Type1,y=Type2)) +
   geom_jitter(aes(colour = Type1)) +
   theme(axis.text.x = element_text(angle = 90,hjust=1))

t1Att <- ggplot(pokemon, aes(x=Type1, y=Attack)) +
  geom_boxplot(aes(colour = Type1)) +
  theme(axis.text.x = element_text(angle=90,hjust=1), legend.position = 'none')
t1Def <- ggplot(pokemon, aes(x=Type1, y=Defense)) +
  geom_boxplot(aes(colour = Type1)) +
  theme(axis.text.x = element_text(angle=90,hjust=1),legend.position = 'none')
t1SpAtt <- ggplot(pokemon, aes(x=Type1, y=Sp.Atk)) +
  geom_boxplot(aes(colour = Type1)) +
  theme(axis.text.x = element_text(angle=90,hjust=1),legend.position = 'none')
t1SpDef <- ggplot(pokemon, aes(x=Type1, y=Sp.Def)) +
  geom_boxplot(aes(colour = Type1)) +
  theme(axis.text.x = element_text(angle=90,hjust=1),legend.position = 'none')
t1HP <- ggplot(pokemon, aes(x=Type1, y=HP)) +
  geom_boxplot(aes(colour = Type1)) +
  theme(axis.text.x = element_text(angle=90,hjust=1),legend.position = 'none')
t1Speed <- ggplot(pokemon, aes(x=Type1, y=Speed)) +
  geom_boxplot(aes(colour = Type1)) +
  theme(axis.text.x = element_text(angle=90,hjust=1),legend.position = 'none')
grid.arrange(t1HP, t1Att,t1Def, t1Speed, t1SpAtt, t1SpDef, nrow = 2)

t2Att <- ggplot(pokemon, aes(x=Type2, y=Attack)) +
  geom_boxplot(aes(colour = Type2)) +
  theme(axis.text.x = element_text(angle=90,hjust=1), legend.position = 'none')
t2Def <- ggplot(pokemon, aes(x=Type2, y=Defense)) +
  geom_boxplot(aes(colour = Type2)) +
  theme(axis.text.x = element_text(angle=90,hjust=1),legend.position = 'none')
t2SpAtt <- ggplot(pokemon, aes(x=Type2, y=Sp.Atk)) +
  geom_boxplot(aes(colour = Type2)) +
  theme(axis.text.x = element_text(angle=90,hjust=1),legend.position = 'none')
t2SpDef <- ggplot(pokemon, aes(x=Type2, y=Sp.Def)) +
  geom_boxplot(aes(colour = Type2)) +
  theme(axis.text.x = element_text(angle=90,hjust=1),legend.position = 'none')
t2HP <- ggplot(pokemon, aes(x=Type2, y=HP)) +
  geom_boxplot(aes(colour = Type2)) +
  theme(axis.text.x = element_text(angle=90,hjust=1),legend.position = 'none')
t2Speed <- ggplot(pokemon, aes(x=Type2, y=Speed)) +
  geom_boxplot(aes(colour = Type2)) +
  theme(axis.text.x = element_text(angle=90,hjust=1),legend.position = 'none')

grid.arrange(t2HP, t2Att,t2Def, t2Speed, t2SpAtt, t2SpDef, nrow = 2)


#Feature Engineering

df <- combat %>% left_join(pokemon, c("First_pokemon" = "ID") ) %>%
  left_join(pokemon, c("Second_pokemon" = "ID"))
colnames(df)[4:14] <- paste('P1_' , colnames(pokemon)[2:12],sep = '')
colnames(df)[15:25] <- paste('P2_',colnames(pokemon)[2:12],sep = '')

#Binary 1/0 for if pokemon1 wins
df$P1_Win <- ifelse(df$First_pokemon == df$Winner, 1,0)
df$P1_Win <- as.factor(df$P1_Win)


#Type Multiplier for pokemon 1
df$P1T1_Multi <- 0
df$P1T2_Multi <- 0
df$P1T1_Multi <- apply(df, 1, function(x) (types[x[5],x[16]] * types[x[5], x[17]]) )
df$P1T2_Multi <- apply(df, 1, function(x) (types[x[6],x[16]] * types[x[6], x[17]]) )

#Difference of pokemon's stats
mat1 <- as.matrix(df[,7:12], ncol=6)
mat2 <- as.matrix(df[,18:23], ncol=6)

Diff_Mat <- as.data.frame(mat1 - mat2)
colnames(Diff_Mat) <- paste("Diff_", colnames(pokemon)[5:10], sep = '')
rm(mat1,mat2)

#Binary if Legendary
df$P1_Legendary <- ifelse(df$P1_Legendary == "True", 1, 0)
df$P2_Legendary <- ifelse(df$P2_Legendary == "True", 1, 0)

#Final 
df2 <- df[, c(4,13:15,24:28)]
df2 <- cbind(df2, Diff_Mat)
df2 <- df2[, c(1,4,7:15,2:3,5:6)]
df2[, c(3,12:15)] <- lapply(df2[, c(3,12:15)], factor)

#Modeling

#Train Test Split
split <- sample.split(df2$P1_Win, SplitRatio = .8)
train <- subset(df2, split == TRUE)
test <- subset(df2, split == FALSE)

#Logistic Regression
#Full Model
log_reg <- glm(as.formula(paste(colnames(train)[3],"~", paste(colnames(train)[4:15], collapse = '+'), sep = '')), 
               train, family = binomial(link = 'logit'))
summary(log_reg)

#Reduced Model
log_reg2 <- glm(P1_Win ~ 1, train, family = binomial(link = 'logit'))

#Model Selection: Results return full model every time AIC 28680
step(log_reg, direction = "backward")
step(log_reg2, scope = list(lower=log_reg2,upper=log_reg),direction = 'forward')
step(log_reg2, scope = list(lower=log_reg2,upper=log_reg),direction = 'both')

#Model Check
#Logistic Regression gives an accuracy of 88.52%
prob_pred <- predict(log_reg, newdata = test, type = 'response')
y_pred <- ifelse(prob_pred >= 0.5, 1, 0)

cm <- table(test[,3], y_pred)
cm


#Random Forest: 95.87%
forest <- randomForest(as.formula(paste(colnames(train)[3],"~", paste(colnames(train)[4:15], collapse = '+'), sep = '')),
                       train, ntree = 100)
importance(forest)
summary(forest)
plot(forest)
varImpPlot(forest)

prob_pred_rf <- predict(forest, newdata = test, type = 'response')


cm_rf <- table(test[,3], prob_pred_rf)
cm_rf
sum(diag(cm_rf))/sum(cm_rf)

#Add features to the tests dataset
tests <- tests %>% left_join(pokemon, c("First_pokemon" = "ID") ) %>%
  left_join(pokemon, c("Second_pokemon" = "ID"))
colnames(tests)[3:13] <- paste('P1_' , colnames(pokemon)[2:12],sep = '')
colnames(tests)[14:24] <- paste('P2_',colnames(pokemon)[2:12],sep = '')


tests$P1T1_Multi <- 0
tests$P1T2_Multi <- 0
tests$P1T1_Multi <- apply(tests, 1, function(x) (types[x[4],x[15]] * types[x[4], x[16]]) )
tests$P1T2_Multi <- apply(tests, 1, function(x) (types[x[5],x[15]] * types[x[5], x[16]]) )


mat3 <- as.matrix(tests[,6:11], ncol=6)
mat4 <- as.matrix(tests[,17:22], ncol=6)

Diff_Mat2 <- as.data.frame(mat3 - mat4)
colnames(Diff_Mat2) <- paste("Diff_", colnames(pokemon)[5:10], sep = '')
rm(mat3,mat4)

tests$P1_Legendary <- ifelse(tests$P1_Legendary == "True", 1, 0)
tests$P2_Legendary <- ifelse(tests$P2_Legendary == "True", 1, 0)

#Final 
final_test <- tests[, c(3,12:14,23:26)]
final_test <- cbind(final_test, Diff_Mat2)
final_test <- final_test[, c(1,4,7:14,2:3,5:6)]
final_test[, c(11:14)] <- lapply(final_test[, c(11:14)], factor)

prob_pred_rf_2 <- predict(forest, newdata = final_test, type = 'response')
final_test$P1_Win <- prob_pred_rf_2

results <- final_test[, c(1,2,15)]
results <- results %>% left_join(pokemon, by = c("P1_Name" = "Name")) %>% 
  left_join(pokemon, by = c("P2_Name" = "Name"))
results <- results[, c(1:3,5:6)]
colnames(results)[4:5] <- paste("P1_", colnames(pokemon)[3:4], sep = '') 
results$P1_Win <- as.numeric(results$P1_Win) - 1

#Best pokemon by Win %
blah <- results %>% group_by(P1_Name) %>% count(P1_Name, P1_Type1, P1_Type2) 
blah2 <- results %>% group_by(P1_Name) %>% summarise(Win = sum(P1_Win))
poke_performance <- blah %>% left_join(blah2, by = c('P1_Name' = 'P1_Name')) %>% mutate(Win = Win/n)

#Best Type
#Type 1
blah <- results %>% group_by(P1_Type1) %>% count(P1_Type1)
best_type <- results %>% group_by(P1_Type1) %>% summarise_at(.vars = 'P1_Win', sum) %>% left_join(blah, by = c('P1_Type1' = 'P1_Type1'))
best_type$Win_Pct<- best_type$P1_Win / best_type$n
#Type 2
blah2 <- results %>% group_by(P1_Type2) %>% count(P1_Type2)
best_type2 <- results %>% group_by(P1_Type2) %>% summarise_at(.vars = 'P1_Win', sum) %>% left_join(blah2, by = c('P1_Type2' = 'P1_Type2'))
best_type2$Win_Pct<- best_type2$P1_Win / best_type2$n

btype1 <- ggplot(best_type, aes(P1_Type1, Win_Pct)) + 
  geom_bar(stat = 'identity', aes(fill= P1_Type1))
btype2 <- ggplot(best_type2, aes(P1_Type2, Win_Pct)) + 
  geom_bar(stat = 'identity', aes(fill= P1_Type2))


