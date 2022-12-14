#Plantilla para el preprocesado de datos
#Importar el dataset
dataset=read.csv('Data.csv')

#Tratamiento en los valores NA
dataset$Age=ifelse(is.na(dataset$Age),
                   ave(dataset$Age,FUN = function(x) mean(x,na.rm = TRUE)),
                   dataset$Age)

dataset$Salary=ifelse(is.na(dataset$Salary),
                   ave(dataset$Salary,FUN = function(x) mean(x,na.rm = TRUE)),
                   dataset$Salary)

dataset$Country=factor(dataset$Country,
                       levels = c("France","Spain","Germany"),
                       labels = c(1,2,3))

dataset$Purchased=factor(dataset$Purchased,
                         levels = c("No","Yes"),
                         labels = c(0,1))