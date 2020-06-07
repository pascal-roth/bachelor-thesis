import fc_pre_processing_load
import fc_post_processing

mechanism_input = 'cai'
equivalence_ratio = [0.0]
pressure = [0]
temperature = [0]
pode = [0]
number_net = '001'
number_test_run = '000'

# get the model
model, criterion, features, labels, x_scaler, y_scaler, _, _, number_train_run = fc_post_processing. \
    load_checkpoint(number_net)
print('Model loaded, begin to load data ...')

# get train and test samples
x_train, y_train = fc_pre_processing_load.load_samples(mechanism_input, number_train_run,
                                                       equivalence_ratio, pressure, temperature,
                                                       pode, features, labels, select_data='exclude',
                                                       category='train')

x_test, y_test = fc_pre_processing_load.load_samples(mechanism_input, number_test_run,
                                                     equivalence_ratio, pressure,
                                                     temperature, pode, features, labels,
                                                     select_data='exclude', category='test')

#  Load  test tensors
test_loader = fc_pre_processing_load.load_dataloader(x_test, y_test, split=False,
                                                     x_scaler=x_scaler, y_scaler=y_scaler, features=None)

print('Data loaded!')

# calculate accuracy
acc_mean = fc_post_processing.calc_acc(model, criterion, test_loader, y_scaler)
print('The mean accuracy with a 5% tolerance is {}'.format(acc_mean))

# normalize the data
x_train, _ = fc_pre_processing_load.normalize_df(x_train, scaler=x_scaler)
y_train, _ = fc_pre_processing_load.normalize_df(y_train, scaler=y_scaler)

x_test, _ = fc_pre_processing_load.normalize_df(x_test, scaler=x_scaler)
y_test, _ = fc_pre_processing_load.normalize_df(y_test, scaler=y_scaler)

# plot the output of NN and reactor together with the closest parameter in the training set (data between the
# interpolation took place)
fc_post_processing.plot_data(model, x_train, y_train, x_test, y_test, x_scaler,
                             y_scaler, number_net, plt_nbrs=True, features=features)