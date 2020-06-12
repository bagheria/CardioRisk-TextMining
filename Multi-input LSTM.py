from keras.layers import Input, Embedding, LSTM, Dense, concatenate, Bidirectional
from keras.models import Model, Sequential
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, \
    confusion_matrix, auc, roc_curve, zero_one_loss, accuracy_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence


def read_r_object(rdatafile):
    # RDatafile is a string: "D:/.../file.RData"
    import rpy2.robjects as robjects
    # from rpy2.robjects import pandas2ri
    robjects.r['load'](rdatafile)
    lstm_x_bow = robjects.r['lstm_X_bag_of_words']
    x_training_bow = robjects.r['X_training_BOW']
    return lstm_x_bow, x_training_bow


def read_data(file):
    xbo = pd.read_csv(file, encoding='latin-1')
    df = pd.DataFrame(xbo)
    return df


def split_dataframe(data, selected_columns):
    first_part = data[selected_columns]
    second_part = data[data.columns.difference(selected_columns)]
    return first_part, second_part


def train_test(data, labels, validation_split=0.2):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    n_validation_samples = int(validation_split * data.shape[0])
    x_train = data[:-n_validation_samples]
    y_train = labels[:-n_validation_samples]
    x_test = data[-n_validation_samples:]
    y_test = labels[-n_validation_samples:]

    return x_train, y_train, x_test, y_test


def train_test_(data, validation_split=0.25):
    n_validation_samples = int(validation_split * data.shape[0])
    train_set = data[:-n_validation_samples]
    test_set = data[-n_validation_samples:]

    return train_set, test_set


def dl_multi_input(n_input_variables=100, n_add_variables=5, n_output=1):
    reports_input = Input(shape=(n_input_variables,), dtype='int32', name='reports_input')
    x = Embedding(output_dim=512, input_dim=10000, input_length=n_input_variables)(reports_input)
    lstm_out = Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2))(x)
    reports_output = Dense(n_output, activation='sigmoid', name='reports_output')(lstm_out)

    medical_vars_input = Input(shape=(n_add_variables,), name='medical_vars_input')
    x = concatenate([lstm_out, medical_vars_input])

    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    final_output = Dense(n_output, activation='sigmoid', name='final_output')(x)
    model = Model(inputs=[reports_input, medical_vars_input], outputs=[reports_output, final_output])

    return model


def dl_model(data, labels):
    n_input_variables = data.shape[1]
    n_output = labels.ndim
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=n_input_variables))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(n_output, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(data, labels, epochs=20, batch_size=32)
    return model


def compile_model(model, optim, l_weights):
    model.compile(optimizer=optim,
                  loss={'final_output': 'binary_crossentropy', 'reports_output': 'binary_crossentropy'},
                  loss_weights={'final_output': l_weights[0], 'reports_output': l_weights[1]},
                  metrics=['accuracy'])


def performance(true_labels, pred_labels):
    confusion_matrix(true_labels, pred_labels)
    print(classification_report(true_labels, pred_labels))
    fpr, tpr, thresholds = roc_curve(true_labels, pred_labels)
    print('auc = ', auc(fpr, tpr))
    print('zero_one_loss = ', zero_one_loss(true_labels, pred_labels))
    print('accuracy_score = ', accuracy_score(true_labels, pred_labels))


def main():
    # set seed for reproducibility
    np.random.seed(321)
    csvfile = "D:/.../file.csv"
    xray_data = read_data(csvfile)
    # flag is either 'text' or 'dataframe', showing the format of the text part of the data.
    flag = 'text'
    cols = ['Unnamed: 0', 'MACEstat', 'AGE', 'SEX', 'SMOKING', 'SYST', 'DM', 'HDL',
            'TOTCHOLEST', 'Mdrd', 'EVERcvd', 'EVERstroke', 'EVERpad', 'EVERaaa', 'YrSinceDiagnosisCVD']
    # train and test
    if flag == 'dataframe':
        xray_train, xray_test = train_test_(xray_data)
        clin_vars_train, reports_train = split_dataframe(xray_train, cols)
        labels_train = clin_vars_train['MACEstat']
        clin_vars_train = clin_vars_train[clin_vars_train.columns.difference(['Unnamed: 0', 'MACEstat'])]
        clin_vars_test, reports_test = split_dataframe(xray_test, cols)
        labels_test = clin_vars_test['MACEstat']
        clin_vars_test = clin_vars_test[clin_vars_test.columns.difference(['Unnamed: 0', 'MACEstat'])]
    else:  # flag is 'text'
        texts = xray_data['text']
        labels = xray_data['MACEstat']
        clin_vars = xray_data[xray_data.columns.difference(['Unnamed: 0', 'MACEstat', 'text'])]
        tokenizer = Tokenizer(num_words=30000, lower=True)
        tokenizer.fit_on_texts(texts)
        texts_seq = tokenizer.texts_to_sequences(texts)
        texts_seq = np.asarray(texts_seq)
        # labels = pd.get_dummies(labels)
        # labels = np.asarray(labels)

        # shuffle indices before train test split
        indices = np.arange(texts_seq.shape[0])
        np.random.shuffle(indices)
        texts_seq = texts_seq[indices]
        labels = labels[indices]
        clin_vars = clin_vars.loc[indices]  # clin_vars is a dataframe

        texts_seq = sequence.pad_sequences(texts_seq)
        reports_train, reports_test = train_test_(texts_seq)
        labels_train, labels_test = train_test_(labels)
        clin_vars_train, clin_vars_test = train_test_(clin_vars)

    # model 1
    model_misslr = dl_model(clin_vars_train[['AGE', 'SEX']], labels_train)
    labels_pred = model_misslr.predict_classes(clin_vars_test[['AGE', 'SEX']])
    performance(labels_test, labels_pred)

    # model 2
    model_clin_vars = dl_model(clin_vars_train, labels_train)
    labels_pred = model_clin_vars.predict_classes(clin_vars_test)
    performance(labels_test, labels_pred)

    # model 3 AND 4
    # clin_vars_train = clin_vars_train[['AGE', 'SEX']]
    # clin_vars_test = clin_vars_test[['AGE', 'SEX']]

    xray_lstm_model = dl_multi_input(n_input_variables=reports_train.shape[1],
                                     n_add_variables=clin_vars_train.shape[1],
                                     n_output=labels_train.ndim)
    xray_lstm_model.summary()

    loss_weights = np.array([1, 0.2])
    optimizer = 'adam'
    compile_model(model=xray_lstm_model,
                  optim=optimizer,
                  l_weights=loss_weights)

    xray_lstm_model.fit({'reports_input': reports_train, 'medical_vars_input': clin_vars_train},
                        {'final_output': labels_train, 'reports_output': labels_train},
                        epochs=20, batch_size=64)

    labels_pred_final, labels_pred_reports = xray_lstm_model.predict([reports_test, clin_vars_test])
    pl_final = (labels_pred_final >= 0.5).astype(np.int)
    performance(labels_test, pl_final)


if __name__ == '__main__':
    main()
