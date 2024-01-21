datos <- read.csv("/home/miguel/Documentos/Stock_Exchange_NN_PP/src/EGX 30 Historical DataW.csv", header = TRUE, sep = ",")
cierre2 <- datos[["Price"]]


write.csv(cierre2, file = "/home/miguel/Documentos/Stock_Exchange_NN_PP/src/C_Egipto_semanal.csv", row.names = FALSE)
