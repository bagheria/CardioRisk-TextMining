# lstm variations

library(keras)
source("prep.R")

# type of all env variables
# eapply(.GlobalEnv,typeof)

load(file = "unravelficd.Rda")
dtm <- create_dtm_matrix(unravel.ficd)
tfidf <- create_tfidf_matrix(dtm)

dim(dtm)
dim(tfidf)
dim(unravel.ficd)
# unravel.ficd$ICD10_Code_first[0:5]
tr_classes <- as.factor(unravel.ficd$ICD10_Code_first)
X_bag_of_words <- as.matrix(dtm)
X_tfidf <- as.matrix(tfidf)


maxlen <- 80 # length of each document
# x_train <- X_bag_of_words
# y_train <- tr_classes

y_binary = to_categorical(as.numeric(tr_classes)) # one-hot encoding

cat('Pad sequences (samples x time)\n')
xb_train <- pad_sequences(X_bag_of_words, maxlen = maxlen)
cat('x_train shape:', dim(xb_train), '\n')

max_features <- dim(dtm)[2]

cat('Build model...\n')
model <- keras_model_sequential()

model %>%
  layer_embedding(input_dim = max_features, output_dim = 100) %>% 
  bidirectional(layer_lstm(units = 100, 
                           dropout = 0.2, 
                           recurrent_dropout = 0.2)) %>%
  layer_flatten() %>% 
  layer_dense(units = 3, activation = 'softmax')

summary(model)

# Try using different optimizers and different optimizer configs
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

cat('Train...\n')
model %>% keras::fit(
  x = xb_train, 
  y = y_binary,
  batch_size = 64,
  epochs = 50,
  # validation_data = list(x_test = xb_train, y_test = tr_classes)
)

scores <- model %>% evaluate(xb_train, y_binary)
cat('Test score:', scores[[1]])
cat('Test accuracy', scores[[2]])

prediction <- model %>% predict_classes(xb_train)
# levels(tr_classes)
t <- table(true = as.numeric(tr_classes), pred = prediction)

