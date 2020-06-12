library(keras)
source("prep.R")


dim(dtm)
tr_classes <- as.factor(X_training_BOW$MACEstat)
table(tr_classes)
lstm_X_bag_of_words <- as.matrix(dtm)

maxlen <- 300 # length of each document
# x_train <- X_bag_of_words
# y_train <- tr_classes

y_binary = to_categorical(as.numeric(tr_classes)) # one-hot encoding

cat('Pad sequences (samples x time)\n')
xb_train <- pad_sequences(lstm_X_bag_of_words, maxlen = maxlen)
cat('x_train shape:', dim(xb_train), '\n')

max_features <- dim(dtm)[2]

cat('Build model...\n')

input_1 <- layer_input(1)
input_2 <- layer_input(1)

vec_1 <- input_1 %>%
  layer_embedding(input_dim = max_features, output_dim = 100) %>% 
  bidirectional(layer_lstm(units = 100, 
                           dropout = 0.2, 
                           recurrent_dropout = 0.2)) %>% 
  layer_flatten()

vec_2 <- input_2 %>% layer_dense(units = 15, activation = 'relu')

#input2 %>% layer_dense(units = 8, activation = 'relu') %>% 
# layer_dense(units = 3, activation = 'softmax')

concat_inputs <- layer_concatenate(list(vec_1, vec_2))
out <- concat_inputs %>% layer_dense(units = 1, activation = 'softmax')

model <- keras_model(inputs = list(input_1, input_2),
                     outputs = out)

summary(model)

model %>% save_model_hdf5("my_model.h5")

# Try using different optimizers and different optimizer configs
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

cat('Train...\n')
model %>% fit(
  inputs = xb_train,
  outputs = y_binary,
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

