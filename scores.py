from tensorflow.keras import Input, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import  BinaryCrossentropy
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_log_error
from sklearn.metrics import accuracy_score
from tensorflow.keras import Input, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_log_error


#implementation from https://github.com/ydataai/ydata-synthetic/blob/dev/examples/timeseries/TimeGAN_Synthetic_stock_data.ipynb

def RNN_prediction(units,input_size):
    opt = Adam(name='AdamOpt')
    loss = MeanAbsoluteError(name='MAE')
    model = Sequential()
    model.add(GRU(units=units,
                  name=f'RNN_1'))
    model.add(Dense(units=input_size,
                    activation='sigmoid',
                    name='OUT'))
    model.compile(optimizer=opt, loss=loss)
    return model

def predictive_score_metrics(real_data,synth_data):

    real_data=np.asarray(real_data)
    n_events = len(real_data)
    seq_len = len(real_data[0][:,0])
    input_dim = len(real_data[0][0,:])
    synth_data = np.asarray(synth_data[:n_events])
    

    #Split data on train and test
    idx = np.arange(n_events)
    n_train = int(.75*n_events)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    #Define the X for synthetic and real data
    X_real_train = real_data[train_idx, :seq_len-1, :]
    X_synth_train = synth_data[train_idx, :seq_len-1, :]

    X_real_test = real_data[test_idx, :seq_len-1, :]
    y_real_test = real_data[test_idx, -1, :]

    #Define the y for synthetic and real datasets
    y_real_train = real_data[train_idx, -1, :]
    y_synth_train = synth_data[train_idx, -1, :]

    ts_real = RNN_prediction(2,input_dim)
    early_stopping = EarlyStopping(monitor='val_loss',patience=3)

    real_train = ts_real.fit(x=X_real_train,
                          y=y_real_train,
                          validation_data=(X_real_test, y_real_test),
                          epochs=200,
                          batch_size=128,
                          callbacks=[early_stopping],
                          verbose=0)
    
    ts_synth = RNN_prediction(12,input_dim)
    synth_train = ts_synth.fit(x=X_synth_train,
                          y=y_synth_train,
                          validation_data=(X_real_test, y_real_test),
                          epochs=200,
                          batch_size=128,
                          callbacks=[early_stopping],
                          verbose=0)
    
    real_predictions = ts_real.predict(X_real_test)
    synth_predictions = ts_synth.predict(X_real_test)

    metrics_dict = {'r2': [r2_score(y_real_test, real_predictions),
                       r2_score(y_real_test, synth_predictions)],
                'MAE': [mean_absolute_error(y_real_test, real_predictions),
                        mean_absolute_error(y_real_test, synth_predictions)]}

    return metrics_dict

def get_predictive_score(args,samples_gen):

    samples_gen = samples_gen.cpu()
    samples_gen = np.array(samples_gen)
    if args.generator != "crnn":
      samples_gen = np.squeeze(samples_gen, axis=2)
      samples_gen = np.transpose(samples_gen, (0, 2, 1))
    seq_len = args.seq_length
    features = args.features

    if args.dataset_type == "Sine":
        dataX = sine_data_generation(10000, seq_len ,features)
        pred = predictive_score_metrics(dataX,samples_gen[:10000])
    
    elif args.dataset_type == "Gaus":
        dataX = gaus_data_loading(seq_len,args.gaus_phi,args.gaus_sigma ,3000,features)
        pred = predictive_score_metrics(dataX,samples_gen[:3000])

    elif args.dataset_type == "Stock":
        dataX = google_data_loading (seq_len)
        pred = predictive_score_metrics(dataX,samples_gen)

    elif args.dataset_type == "Chickenpox":
        dataX = chickenpox_data_loading (seq_len)
        pred = predictive_score_metrics(dataX,samples_gen)

    elif args.dataset_type == "Energy":
        dataX = energy_data_loading (seq_len)
        pred = predictive_score_metrics(dataX,samples_gen)

    return pred
    
def RNN_discriminator(units):
    opt = Adam(name='AdamOpt')
    loss = BinaryCrossentropy()
    model = Sequential()
    model.add(GRU(units=units,
                  name=f'RNN_1'))
    model.add(Dense(units=1,
                    activation='sigmoid',
                    name='OUT'))
    model.compile(optimizer=opt, loss=loss)
    return model

def discriminative_score_metricsNEW(real_data,synth_data):
    #Prepare the dataset for the regression model
    real_data=np.asarray(real_data)
    n_events = len(real_data)
    seq_len = len(real_data[0][:,0])
    input_dim = len(real_data[0][0,:])
    synth_data = np.asarray(synth_data[:n_events])
    
    Y = np.concatenate((np.ones(n_events), np.zeros(n_events)), axis = 0)
    X = np.concatenate((real_data, synth_data), axis = 0)

    #Split data on train and test
    idx = np.arange(n_events)
    np.random.shuffle(idx)
    n_train = int(.75*n_events)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    X_train = X[train_idx, :, :]
    Y_train = Y[train_idx]

    X_test = X[test_idx, :, :]
    Y_test = Y[test_idx]
    #n_epochs = int(150000/n_events)
    n_epochs = 3

    ts_dis = RNN_discriminator(1)
    early_stopping = EarlyStopping(monitor='val_loss',patience=2)

    discrim = ts_dis.fit(x=X_train,
                          y=Y_train,
                          validation_data=(X_test, Y_test),
                          epochs=n_epochs,
                          batch_size=128,
                          callbacks=[early_stopping],
                          verbose=0)
    
    Y_pred = ts_dis.predict(X_test)
    Acc = accuracy_score((Y_pred>0.5),Y_test)
    #Acc = mean_absolute_error(Y_pred,Y_test)

    return Acc

    
def get_discriminative_score(args,samples_gen):

    samples_gen = samples_gen.cpu()
    samples_gen = np.array(samples_gen)
    if args.generator != "crnn":
      samples_gen = np.squeeze(samples_gen, axis=2)
      samples_gen = np.transpose(samples_gen, (0, 2, 1))
    seq_len = args.seq_length
    features = args.features

    if args.dataset_type == "Sine":
        dataX = sine_data_generation(10000, seq_len, features)
        disc = discriminative_score_metricsNEW(dataX,samples_gen)

    elif args.dataset_type == "Gaus":
        dataX = gaus_data_loading(seq_len,args.gaus_phi,args.gaus_sigma ,3000,features)
        disc = discriminative_score_metricsNEW(dataX,samples_gen[:3000])

    elif args.dataset_type == "Stock":
        dataX = google_data_loading (seq_len)
        disc = discriminative_score_metricsNEW(dataX,samples_gen[:len(dataX)])


    elif args.dataset_type == "Chickenpox":
        dataX = chickenpox_data_loading(seq_len)
        disc = discriminative_score_metricsNEW(dataX,samples_gen[:len(dataX)])

    elif args.dataset_type == "Energy":
        dataX = energy_data_loading (seq_len)
        disc = discriminative_score_metricsNEW(dataX,samples_gen[:len(dataX)])
        #make sure datasets are same size

    return disc
