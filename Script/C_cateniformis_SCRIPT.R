#20/08/2025
#Modelamiento para distribución espacial de Cedrelinga cateniformis en el Perú
#empleandos diversos algoritmos de apredizaje automático

library(readxl)
library(plotrix)
library(RColorBrewer)
library(sf)
library(sp)
library(raster)
library(terra)
library(caTools)
library(caret)
library(PerformanceAnalytics)
library(pROC)
library(xlsx)
library(corrplot)
library(corrr)
library(writexl)
library(officer)
library(flextable)
library(dplyr)
library(keras)
library(xgboost)
library(keras)
library(reticulate)
library(catboost)
install_keras()
library(reticulate)
reticulate::py_install("catboost")
catboost <- import("catboost")

setwd("E:/PRO_FORESTAL/Distribución_Tornillo/")

# CARGAR VARIABLES RASTER Y PUNTOS DE PRESENCIA
li <- list.files("Variables/", pattern = ".tif$", full.names = TRUE)
variables <- stack(li)
presencia <- st_read("Variables/Presencia_Tornillo.shp")
presencia <- st_transform(presencia, crs(variables))

#-------------------CORRELACIÓN DE VARIABLES (1)------------------#
# Convertir sf a data.frame y extraer coordenadas
coords <- st_coordinates(presencia)
datafra <- as.data.frame(coords)

# EXTRAER COORDENADAS
coords <- st_coordinates(presencia)
coords_df <- as.data.frame(coords)

# EXTRAER VALORES RASTER EN LOS PUNTOS
pres.covs <- raster::extract(variables, coords_df)
pres.covs <- as.data.frame(pres.covs)
pres.covs <- na.omit(pres.covs)
pres.covs <- unique(pres.covs)

# GUARDAR COMO CSV
write.table(pres.covs, file = "Cedrelinga.csv", sep = ",", row.names = FALSE)

# MATRIZ DE CORRELACIÓN
mat_cor <- cor(pres.covs)

# VERIFICAR si hay columnas con varianza cero (para evitar errores en cor())
mat_cor <- cor(pres.covs[, sapply(pres.covs, function(x) sd(x) != 0)])

# Graficar
png("correlacion_ellipse.png", units = "cm", width = 38, height = 30, res = 300)
corrplot(
  mat_cor,
  method = "ellipse",
  type = "lower",
  tl.cex = 1.2,
  cl.cex = 1.2,
  number.cex = 1.2,
  p.mat = fake_pmat,
  sig.level = 0.05,
  insig = "label_sig",
  pch = "*",
  pch.cex = 2.5,
  pch.col = "black"
)
dev.off()

#-------------------------LLAMADO DE VARIABLES NUEVAS----------------------------#
#LLAMADO DE VARIABLES
li <- list.files("Variables/", pattern = ".tif$", full.names = TRUE)

#EXCLUIR LAS VARIABLES CON ALTA CORRELACIÓN
excluir <- c("Apparent Density.tif", "Clay content.tif", "Coarse Fragments.tif", "pH Soil.tif")
li <- li[!basename(li) %in% excluir]

#CARGAR VARIABLES EN AGRUPAMIENTO
variables <- stack(li)
names(variables)

presencia <- st_read("Variables/Presencia_Tornillo.shp")
presencia <- st_transform(presencia, crs(variables))

#------------------------MODELAMIENTO---------------------#

# EXTRAER VALORES DE PIXELES (PRESENCIA Y PSEUDO-AUSENCIAS)
val_pres <- raster::extract(variables, presencia)
data_pres <- as.data.frame(val_pres)
data_pres$clase <- 1

pseudo_aus <- randomPoints(variables[[1]], n = 1000, p = st_coordinates(presencia))
val_aus <- raster::extract(variables, pseudo_aus)
data_aus <- as.data.frame(val_aus)
data_aus$clase <- 0

# UNIR Y PREPARAR DATOS
datos <- na.omit(rbind(data_pres, data_aus))
X <- as.matrix(scale(datos[, 1:(ncol(datos)-1)]))
y <- as.numeric(datos$clase)

# SEPARAR EN TRAIN Y VALIDACIÓN
idx <- sample(1:nrow(X), size = 0.7 * nrow(X))
X_train <- X[idx, ]; y_train <- y[idx]
X_valid <- X[-idx, ]; y_valid <- y[-idx]

###########################################################################
# MODELO 1: RED NEURONAL (BACKPROPAGATION)
model_nn <- keras_model_sequential() %>%
  layer_dense(units = 10, activation = "relu", input_shape = ncol(X_train)) %>%
  layer_dense(units = 1, activation = "sigmoid")

model_nn %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.01),
  loss = "binary_crossentropy",
  metrics = "accuracy"
)

model_nn %>% fit(X_train, y_train, epochs = 100, batch_size = 32, validation_split = 0.2, verbose = 0)

library(reticulate)
py_run_string("from tensorflow.keras.utils import plot_model; print('plot_model disponible')")
reticulate::py_install(c("pydot", "graphviz"), pip = TRUE, force = TRUE)
system("dot -V")
# Importar el módulo de utilidades de Keras desde TensorFlow
tf <- import("tensorflow")
utils <- tf$keras$utils

###########################################################################
# MODELO 2: XGBOOST
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dvalid <- xgb.DMatrix(data = X_valid)

params_xgb <- list(objective = "binary:logistic", eval_metric = "logloss", max_depth = 4, eta = 0.1)
xgb_model <- xgb.train(params = params_xgb, data = dtrain, nrounds = 100, verbose = 0)

###########################################################################
# MODELO 3: CATBOOST
# Crear Pools
pool_train <- catboost$Pool(data = X_train, label = y_train)
pool_valid <- catboost$Pool(data = X_valid, label = y_valid)

# Crear lista de parámetros como lista R
params_cb <- list(
  iterations = 100L,
  learning_rate = 0.1,
  depth = 4L,
  loss_function = 'Logloss',
  verbose = 0L
)

# Crear el modelo
model_cb <- do.call(catboost$CatBoostClassifier, params_cb)

# Entrenar el modelo
model_cb$fit(pool_train)

###########################################################################
# PREDICCIONES Y AUC PARA TRAIN Y VALID
# BACKPROPAGATION
pred_nn_train <- as.numeric(predict(model_nn, X_train))
pred_nn_valid <- as.numeric(predict(model_nn, X_valid))

# XGBOOST
pred_xgb_train <- as.numeric(predict(xgb_model, dtrain))
pred_xgb_valid <- as.numeric(predict(xgb_model, dvalid))

# CATBOOST
pred_cb_train <- as.numeric(model_cb$predict_proba(pool_train)[,2])
pred_cb_valid <- as.numeric(model_cb$predict_proba(pool_valid)[,2])

# CURVAS ROC
roc_nn_train <- roc(y_train, pred_nn_train)
roc_nn_valid <- roc(y_valid, pred_nn_valid)

roc_xgb_train <- roc(y_train, pred_xgb_train)
roc_xgb_valid <- roc(y_valid, pred_xgb_valid)

roc_cb_train <- roc(y_train, pred_cb_train)
roc_cb_valid <- roc(y_valid, pred_cb_valid)

# MOSTRAR AUC
cat("Backpropagation - AUC train:", round(auc(roc_nn_train), 3),
    "valid:", round(auc(roc_nn_valid), 3), "\n")

cat("XGBoost        - AUC train:", round(auc(roc_xgb_train), 3),
    "valid:", round(auc(roc_xgb_valid), 3), "\n")

cat("CatBoost       - AUC train:", round(auc(roc_cb_train), 3),
    "valid:", round(auc(roc_cb_valid), 3), "\n")


###########################################################################
# PREDICCIÓN ESPACIAL SOBRE RASTER
valores_raster <- as.data.frame(getValues(variables))
valores_raster_scaled <- scale(valores_raster)
valid_idx <- complete.cases(valores_raster_scaled)
X_pred <- as.matrix(valores_raster_scaled[valid_idx, ])

# BACKPROP
pred_map_nn <- predict(model_nn, X_pred)

# XGBOOST
dX_pred <- xgb.DMatrix(data = X_pred)
pred_map_xgb <- predict(xgb_model, dX_pred)

# CATBOOST
pool_pred <- catboost$Pool(data = X_pred)
pred_map_cb <- model_cb$predict_proba(pool_pred)[,2]

# CREAR RASTERS
r_base <- raster(variables[[1]])
values(r_base) <- NA

r_nn <- r_xgb <- r_cb <- r_base
values(r_nn)[valid_idx]  <- pred_map_nn
values(r_xgb)[valid_idx] <- pred_map_xgb
values(r_cb)[valid_idx]  <- pred_map_cb

# VISUALIZAR
plot(r_nn, main = "Tornillo (Backpropagation)")
plot(r_xgb, main = "Tornillo (XGBOOST)")
plot(r_cb, main = "Tornillo (CATBOOST)")

# EXPORTAR RASTERS
writeRaster(r_nn,  "Tornillo_Prob_Backprop.tif",  overwrite = TRUE)
writeRaster(r_xgb, "Tornillo_Prob_XGBoost.tif",   overwrite = TRUE)
writeRaster(r_cb,  "Tornillo_Prob_CatBoost.tif",  overwrite = TRUE)

#ENSAMBLADO DE LOS 3 MODELOS GENERADOS 

modelo_1 <- raster("Tornillo_Prob_Backprop.tif")
modelo_2 <- raster("Tornillo_Prob_XGBoost.tif")
modelo_3 <- raster("Tornillo_Prob_CatBoost.tif")

Modelos_prom <- (modelo_1+modelo_2+modelo_3)/3
plot(Modelos_prom)
writeRaster(Modelos_prom,  "Tornillo_Promedio.tif",  overwrite = TRUE)

#-------------------------IMPORTANCIA DE VARIABLES-----------------------------#
# IMPORTANCIA DE VARIABLES - XGBOOST
importance_matrix_xgb <- xgb.importance(model = xgb_model)
importance_matrix_xgb$Importance <- round(100 * importance_matrix_xgb$Gain / sum(importance_matrix_xgb$Gain), 2)
print(importance_matrix_xgb[, c("Feature", "Importance")])

# IMPORTANCIA DE VARIABLES - CATBOOST
importancia_cb <- model_cb$get_feature_importance(pool_train, type = "FeatureImportance")
nombres_cb <- colnames(X_train)
df_importancia_cb <- data.frame(Variable = nombres_cb,
                                Importancia = round(100 * importancia_cb / sum(importancia_cb), 2))
print(df_importancia_cb)

# FUNCIÓN PARA IMPORTANCIA POR PERMUTACIÓN (BACKPROPAGATION)
perm_importance_nn <- function(model, X_valid, y_valid, auc_original) {
  importancia <- numeric(ncol(X_valid))
  for (i in seq_len(ncol(X_valid))) {
    X_permutado <- X_valid
    X_permutado[, i] <- sample(X_permutado[, i])  # permutar columna
    pred <- as.numeric(predict(model, X_permutado))
    auc_perm <- auc(y_valid, pred)
    importancia[i] <- auc_original - auc_perm
  }
  importancia <- round(100 * importancia / sum(importancia), 2)
  data.frame(Variable = colnames(X_valid), Importancia = importancia)
}

# AUC original del modelo
auc_nn <- auc(roc_nn_valid)

# IMPORTANCIA
importancia_nn <- perm_importance_nn(model_nn, X_valid, y_valid, auc_nn)
print(importancia_nn)

#------------------------GRÁFICO DE CONTRIBUCIÓN----------------------#

datos <- read_excel("CONTRIBUCIÓN.xlsx", sheet = "XGBOOST")
datos <- datos[order(datos$Contribución), ]
valores <- datos$Contribución
nombres <- datos$Variable
colores <- colorRampPalette(brewer.pal(9, "Greens"))(length(valores))

##############GRÁFICO CONTRIBUCIÓN###################
png("contribucion_XGBOOST.png", width = 1200, height = 800, res = 150)

# Configuración general
par(
  mar = c(6, 12, 5, 4),
  family = "serif",
  cex.axis = 1,
  font.axis = 1 
)

# Crear gráfico 
bp <- barplot(
  valores,
  names.arg = rep("", length(valores)),
  horiz = TRUE,
  col = colores,
  border = "black",
  xlab = "",        
  main = "",
  axes = TRUE,
  xlim = c(0, max(valores) + 12),
  width = 1.5                    
)

# Etiquetas de variables 
text(
  x = -2,
  y = bp,
  labels = nombres,
  xpd = TRUE,
  adj = 1,
  font = 1,
  family = "serif",
  cex = 1
)

# Eje X con valores normales
axis(1, lwd = 2, col = "black", col.axis = "black", font = 1)

# Etiqueta eje X
mtext("Contribution (%)", side = 1, line = 4, font = 2, cex = 1.5, family = "serif")

# Título
title(main = "Importance of Variables", cex.main = 2, font.main = 2, family = "serif", line = 3)

# Texto al final de las barras
text(
  x = valores + 3,
  y = bp,
  labels = paste0(round(valores, 1), "%"),
  font = 1,
  family = "serif",
  cex = 1
)

dev.off()


#-------------------------GRÁFICA AUC-----------------------------#
# Crear archivo PNG
output_file <- "AUC_combinados_modelos_CORREGIDO.png"
png(output_file, units = 'cm', width = 21, height = 6, res = 1200)

# Configuración de la figura
par(
  mfrow = c(1, 3),
  mar = c(4, 4, 2, 1) + 0.1,
  family = "serif"
)

# Parámetros generales
line_width <- 1
axis_size <- 1.2
label_size <- 1.2
title_size <- 1.5
legend_size <- 1

## 1. Backpropagation
plot(roc_nn_train, col = "blue", lwd = line_width,
     xlim = c(1, 0), ylim = c(0, 1), legacy.axes = TRUE,
     xlab = "", ylab = "True Positive Rate",
     cex.lab = label_size, cex.axis = axis_size)
mtext("False Positive Rate (1 - Specificity)", side = 1, line = 2.5, cex = 0.7)
lines.roc(roc_nn_valid, col = "green", lwd = line_width)
legend("bottomright", inset = c(0, -0.05), 
       legend = c(
         paste("Training =", round(auc(roc_nn_train), 2)),
         paste("Testing  =", round(auc(roc_nn_valid), 2))),
       col = c("blue", "green"), lty = 1, lwd = line_width,
       cex = legend_size, bty = "n", xpd = TRUE)
title("Backpropagation", font.main = 2, cex.main = title_size)

## 2. XGBoost
plot(roc_xgb_train, col = "blue", lwd = line_width,
     xlim = c(1, 0), ylim = c(0, 1), legacy.axes = TRUE,
     xlab = "", ylab = "True Positive Rate",
     cex.lab = label_size, cex.axis = axis_size)
mtext("False Positive Rate (1 - Specificity)", side = 1, line = 2.5, cex = 0.7)
lines.roc(roc_xgb_valid, col = "green", lwd = line_width)
legend("bottomright", inset = c(0, -0.05), 
       legend = c(
         paste("Training =", round(auc(roc_xgb_train), 2)),
         paste("Testing  =", round(auc(roc_xgb_valid), 2))),
       col = c("blue", "green"), lty = 1, lwd = line_width,
       cex = legend_size, bty = "n", xpd = TRUE)
title("XGBoost", font.main = 2, cex.main = title_size)

## 3. CatBoost
plot(roc_cb_train, col = "blue", lwd = line_width,
     xlim = c(1, 0), ylim = c(0, 1), legacy.axes = TRUE,
     xlab = "", ylab = "True Positive Rate",
     cex.lab = label_size, cex.axis = axis_size)
mtext("False Positive Rate (1 - Specificity)", side = 1, line = 2.5, cex = 0.7)
lines.roc(roc_cb_valid, col = "green", lwd = line_width)
legend("bottomright", inset = c(0, -0.05), 
       legend = c(
         paste("Training =", round(auc(roc_cb_train), 2)),
         paste("Testing  =", round(auc(roc_cb_valid), 2))),
       col = c("blue", "green"), lty = 1, lwd = line_width,
       cex = legend_size, bty = "n", xpd = TRUE)
title("CatBoost", font.main = 2, cex.main = title_size)

# Cerrar gráfico
dev.off()

#-------------------------ESTADÍSTICA DE VALIDACIÓN-----------------------------#

###########################################################
################ESTADÍSTICA DE F1-SCORE####################
###########################################################

# Revisar formas
print(length(y_train)); print(str(y_train))
print(length(pred_nn_train)); print(str(pred_nn_train))

# Forzar vectores planos y numéricos
y_train_vec <- as.numeric(drop(y_train))
y_valid_vec <- as.numeric(drop(y_valid))

pred_nn_train_vec <- as.numeric(drop(pred_nn_train))
pred_nn_valid_vec <- as.numeric(drop(pred_nn_valid))

pred_xgb_train_vec <- as.numeric(drop(pred_xgb_train))
pred_xgb_valid_vec <- as.numeric(drop(pred_xgb_valid))

pred_cb_train_vec <- as.numeric(drop(pred_cb_train))
pred_cb_valid_vec <- as.numeric(drop(pred_cb_valid))


calcula_metricas <- function(y_true, y_prob, modelo_nombre = "") {
  y_true <- as.numeric(drop(y_true))
  y_prob <- as.numeric(drop(y_prob))
  
  if (length(y_true) != length(y_prob)) {
    stop("Las longitudes de y_true y y_prob no coinciden: ",
         length(y_true), " vs ", length(y_prob))
  }
  
  roc_obj <- roc(y_true, y_prob)
  # Tomar el primer umbral óptimo (Youden's J) de forma segura
  umbral_vec <- as.numeric(coords(roc_obj, "best", ret = "threshold", transpose = FALSE))
  umbral <- umbral_vec[1]
  
  y_pred <- ifelse(y_prob >= umbral, 1, 0)
  
  tp <- sum(y_true == 1 & y_pred == 1)
  tn <- sum(y_true == 0 & y_pred == 0)
  fp <- sum(y_true == 0 & y_pred == 1)
  fn <- sum(y_true == 1 & y_pred == 0)
  
  precision <- ifelse((tp + fp) == 0, NA, tp / (tp + fp))
  recall <- ifelse((tp + fn) == 0, NA, tp / (tp + fn))
  accuracy <- (tp + tn) / (tp + tn + fp + fn)
  f1 <- ifelse(is.na(precision) | is.na(recall) | (precision + recall) == 0,
               NA,
               2 * precision * recall / (precision + recall))
  
  data.frame(
    Modelo = modelo_nombre,
    Umbral = round(umbral, 3),
    Precision = round(precision, 3),
    Recall = round(recall, 3),
    F1 = round(f1, 3),
    Accuracy = round(accuracy, 3),
    AUC = round(auc(roc_obj), 3)
  )
}

met_nn_train <- calcula_metricas(y_train, pred_nn_train, "Backpropagation (train)")
met_nn_valid <- calcula_metricas(y_valid, pred_nn_valid, "Backpropagation (valid)")

met_xgb_train <- calcula_metricas(y_train, pred_xgb_train, "XGBoost (train)")
met_xgb_valid <- calcula_metricas(y_valid, pred_xgb_valid, "XGBoost (valid)")

met_cb_train <- calcula_metricas(y_train, pred_cb_train, "CatBoost (train)")
met_cb_valid <- calcula_metricas(y_valid, pred_cb_valid, "CatBoost (valid)")

resultados <- rbind(
  met_nn_train, met_nn_valid,
  met_xgb_train, met_xgb_valid,
  met_cb_train, met_cb_valid
)
print(resultados)
write_xlsx(resultados, "metricas_modelos_train_valid.xlsx")

###########################################################
############ANOVA (Prueba de T-student)####################
###########################################################

# Comparaciones pareadas de AUC con DeLong
test_xgb_vs_cb <- roc.test(roc_xgb_valid, roc_cb_valid, method = "delong", paired = TRUE)
test_xgb_vs_nn <- roc.test(roc_xgb_valid, roc_nn_valid, method = "delong", paired = TRUE)
test_cb_vs_nn  <- roc.test(roc_cb_valid,  roc_nn_valid, method = "delong", paired = TRUE)

# Mostrar resultados (diferencia de AUC, estadístico z, p-valor)
print(test_xgb_vs_cb)
print(test_xgb_vs_nn)
print(test_cb_vs_nn)

n_boot <- 1000
n_valid <- length(y_valid)

# Almacena AUC bootstrap
auc_boot <- data.frame(
  nn = numeric(n_boot),
  xgb = numeric(n_boot),
  cb = numeric(n_boot)
)

for (i in seq_len(n_boot)) {
  idx <- sample(seq_len(n_valid), replace = TRUE)
  y_boot <- y_valid[idx]
  
  pred_nn_b <- pred_nn_valid[idx]
  pred_xgb_b <- pred_xgb_valid[idx]
  pred_cb_b <- pred_cb_valid[idx]
  
  auc_boot$nn[i]  <- auc(roc(y_boot, pred_nn_b))
  auc_boot$xgb[i] <- auc(roc(y_boot, pred_xgb_b))
  auc_boot$cb[i]  <- auc(roc(y_boot, pred_cb_b))
}

# Paired t-tests entre modelos
tt_nn_vs_xgb <- t.test(auc_boot$nn, auc_boot$xgb, paired = TRUE)
tt_nn_vs_cb  <- t.test(auc_boot$nn, auc_boot$cb, paired = TRUE)
tt_xgb_vs_cb <- t.test(auc_boot$xgb, auc_boot$cb, paired = TRUE)

# Diferencia de medias, t, p, intervalo
resumen_comparaciones <- list(
  "NN vs XGBoost" = tt_nn_vs_xgb,
  "NN vs CatBoost" = tt_nn_vs_cb,
  "XGBoost vs CatBoost" = tt_xgb_vs_cb
)



