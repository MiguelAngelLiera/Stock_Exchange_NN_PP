datos <- read.csv("~/Documentos/articulos/analisis/Datos históricos GFINBURO.csv", header = TRUE, sep = ",")
cierre <- datos[["Cierre"]]

write.csv(cierre, file = "cierre.csv", row.names = FALSE)
