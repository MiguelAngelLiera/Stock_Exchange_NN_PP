datos <- read.csv("D:/MALIERA/Documents/[00] Stock_Exchange/Stock_Exchange_NN_PP/src/COMI Historical Data.csv", header = TRUE, sep = ",")
cierre2 <- datos[["Cierre"]]

write.csv(cierre2, file = "cierre2.csv", row.names = FALSE)
