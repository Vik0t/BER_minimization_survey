#!/mnt/storage/home/ossidelnikov/anaconda/bin/python

from mxnet import nd, gluon, init, autograd, gpu
from mxnet.gluon import nn
import mxnet as mx

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import pandas as pd

from os import listdir
from os import makedirs
import time
from random import randint
import datetime
import json
import sys
import shutil

import matplotlib.pyplot as plt
import seaborn as sns
import os

#========================
#gpus = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(',')]
#print(gpus)
#========================

full_time = time.time()
mode_1 = int(sys.argv[1]) 
run_1 = int(sys.argv[2])

mode = mode_1
run = run_1

print(mode)
print(run)

power = 1

if run == 1:
    radius_array = [30, 30, 30]
    neuron_array = [96, 96, 96]
if run == 2:
    radius_array = [30, 30, 30]
    neuron_array = [96, 96, 96]
if run == 3:
    radius_array = [30, 30, 30]
    neuron_array = [96, 96, 96]
if run == 4:
    radius_array = [30, 30, 30]
    neuron_array = [96, 96, 96]
if run == 5:
    radius_array = [30, 30, 30]
    neuron_array = [96, 96, 96]
if run > 5:
    radius_array = [30, 30, 30]
    neuron_array = [96, 96, 96]

root = '/mnt/storage/home/ossidelnikov/Stepan/1pol/'

rawdata_path = root + 'Data/Results/1/20/' + str(power) + '.000000/Symbols_1m_1ch_PR_'
results_path = root + 'NN/2022/Logs/Complex/' + str(run) + '/' + str(mode) + '/'

# dirlist = [ item for item in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, item)) ]

gray_symbols_16qam = np.sqrt(0.1) * np.array(
                    [1+1j, 1+3j, 1-1j, 1-3j, 
                    3+1j, 3+3j, 3-1j, 3-3j, 
                    -1+1j, -1+3j, -1-1j, -1-3j,
                    -3+1j, -3+3j, -3-1j, -3-3j])

swap_64_to_128_complex = False

if swap_64_to_128_complex:
    complex_t = np.complex128
    real_t = np.float64
else:
    complex_t = np.complex64
    real_t = np.float32
#========================

# 1: Data import
data_size = 524288
data_files = 64
train_portion = 0.50
train_files = int(data_files * train_portion)
test_files = data_files - train_files

# 2.1: Create model
radius = radius_array[mode]
sample_size = data_size - 2 * radius

shape_size = 4 * radius + 2
neuron_hidden_layer = neuron_array[mode]

init_seed = randint(0, 100000000)

# 3: Fit model
epochs = 500

if run == 2:
    epochs = 1000
if run == 3:
    epochs = 1500
if run == 4:
    epochs = 2000
if run == 5:
    epochs = 2500
if run > 5:
    epochs = 20000

minibatch_multiplier = 1
batch_size = data_size * minibatch_multiplier

lrate = 0.001
if run == 13:
    lrate = 0.01
if run == 15:
    lrate = 0.005
if run == 17:
    lrate = 0.0005

save_model_best = True
model_filename = "net.params"

continue_train = False # True
model_path = results_path
# model_path = results_path + dirlist[-1] + '/'

calc_ber = True

if run == 1:
    run_options = "LR 0.001 500 epochs"
if run == 2:
    run_options = "LR 0.001 1000 epochs"
if run == 3:
    run_options = "LR 0.001 1500 epochs"
if run == 4:
    run_options = "LR 0.001 2000 epochs"
if run == 5:
    run_options = "LR 0.001 2500 epochs"
if run == 6:
    run_options = "Shodimost' Decay --- Start LR 0.001 Step 50 Tol 1e-3 Stop 1e-4 Train poertion 0.5"
if run == 7:
    run_options = "Shodimost' Decay --- Start LR 0.001 Step 100 Tol 1e-3 Stop 1e-4 Train poertion 0.5"
if run == 8:
    run_options = "Shodimost' Decay --- Start LR 0.001 Step 100 Tol 5e-4 Stop 1e-4 Train poertion 0.5"
if run == 9:
    run_options = "Shodimost' Decay --- Start LR 0.001 Step 100 Tol 1e-4 Stop 1e-4 Train poertion 0.5"
if run == 10:
    run_options = "Shodimost' Decay --- Start LR 0.001 Step 100 Tol 5e-5 Stop 1e-4 Train poertion 0.5"
if run == 11:
    run_options = "Shodimost' Decay --- Start LR 0.001 Step 100 Tol 1e-5 Stop 1e-4 Train poertion 0.5"
if run == 12:
    run_options = "Shodimost' Decay --- Start LR 0.001 Step 200 Tol 1e-4 Stop 1e-4 Train poertion 0.5"
if run == 13:
    run_options = "Shodimost' Decay --- Start LR 0.01 Step 100 Tol 1e-4 Stop 1e-4 Train poertion 0.5"
if run == 14:
    run_options = "Shodimost' Decay --- Start LR 0.001 Step 100 Tol 1e-4 Stop 5e-5 Train poertion 0.5"
if run == 15:
    run_options = "Shodimost' Decay --- Start LR 0.005 Step 100 Tol 1e-4 Stop 1e-4 Train poertion 0.5"
if run == 16:
    run_options = "Shodimost' Decay --- Start LR 0.001 Step 100 Tol 1e-4 Stop 5e-4 Train poertion 0.5"
if run == 17:
    run_options = "Shodimost' Decay --- Start LR 0.0005 Step 100 Tol 1e-4 Stop 1e-4 Train poertion 0.5"
if run == 18:
    run_options = "Shodimost' Decay --- Start LR 0.001 Step 150 Tol 1e-4 Stop 1e-4 Train poertion 0.5"
if run == 19:
    run_options = "Shodimost' Decay --- Start LR 0.001 Step 250 Tol 1e-4 Stop 1e-4 Train poertion 0.5"
if run == 20:
    run_options = "Shodimost' Decay --- Start LR 0.001 Step 300 Tol 1e-4 Stop 1e-4 Train poertion 0.5"
if run == 21:
    run_options = "Shodimost' Decay --- Start LR 0.001 Step 350 Tol 1e-4 Stop 1e-4 Train poertion 0.5"
if run == 22:
    run_options = "Shodimost' Decay --- Start LR 0.001 Step 250 Tol 2e-4 Stop 1e-4 Train poertion 0.5"
if run == 23:
    run_options = "Shodimost' Decay --- Start LR 0.001 Step 250 Tol 5e-5 Stop 1e-4 Train poertion 0.5"
if run == 24:
    run_options = "Shodimost' Decay --- Start LR 0.001 Step 250 Tol 1e-4 Stop 2e-4 Train poertion 0.5"
if run == 25:
    run_options = "Shodimost' Decay --- Start LR 0.001 Step 250 Tol 1e-4 Stop 5e-5 Train poertion 0.5"
if run == 26:
    run_options = "Shodimost' Decay --- Start LR 0.001 Step 400 Tol 1e-4 Stop 1e-4 Train poertion 0.5"
if run == 27:
    run_options = "Shodimost' Decay --- Start LR 0.001 Step 250 Tol 1e-4 Stop 1e-4 Train poertion 0.5 + Tol * Lrate"
if run == 28:
    run_options = "Shodimost' Decay --- Start LR 0.001 Step 250 Tol 1e-5 Stop 1e-4 Train poertion 0.5 + Tol * Lrate"
if run == 29:
    run_options = "Shodimost' Decay --- Start LR 0.001 Step 50 Tol 1e-5 Stop 1e-4 Train poertion 0.5 + Tol * Lrate"
if run == 30:
    run_options = "Shodimost' Decay --- Start LR 0.001 Step 200 Tol 1e-4 Stop 1e-4 Train poertion 0.5 + Tol * Lrate"
if run == 31:
    run_options = "Shodimost' Decay --- Start LR 0.001 Step 300 Tol 1e-4 Stop 1e-4 Train poertion 0.5 + Tol * Lrate"
if run == 32:
    run_options = "Shodimost' Decay --- Start LR 0.001 Step 250 Tol 2e-4 Stop 1e-4 Train poertion 0.5 + Tol * Lrate"
if run == 33:
    run_options = "Shodimost' Decay --- Start LR 0.001 Step 250 Tol 5e-5 Stop 1e-4 Train poertion 0.5 + Tol * Lrate"
if run == 34:
    run_options = "Shodimost' Decay --- Start LR 0.001 Step 350 Tol 1e-4 Stop 1e-4 Train poertion 0.5 + Tol * Lrate"
if run == 35:
    run_options = "Shodimost' Decay --- Start LR 0.001 Step 250 Tol 1e-4 Stop 2e-4 Train poertion 0.5 + Tol * Lrate"
if run == 36:
    run_options = "Shodimost' Decay --- Start LR 0.001 Step 250 Tol 1e-4 Stop 5e-5 Train poertion 0.5 + Tol * Lrate"

print('Seed:', init_seed)
print('Batch size:', batch_size)

mx.random.seed(init_seed)

#========================

def symbols_to_codes_v2(data_complex):
    codes = np.zeros(len(data_complex), dtype=np.int32)
    for i in range(len(data_complex)):
        codes[i] = np.argmin(abs(gray_symbols_16qam - data_complex[i]))
    
    return codes

def ber_by_codes(tx, rx):
    diff = tx ^ rx
    errors = 0
    for error in diff:
        while error:
            error &= error - 1
            errors +=1
            
    return 0.25 * errors / len(diff)

def create_metadata():
    metadata = dict( (name, eval(name)) for name in ['init_seed', 'data_size','data_files','train_portion', 'train_files', 'test_files',
                                     'radius', 'neuron_hidden_layer', 'lrate', 'epochs', 'minibatch_multiplier', 'batch_size', 'continue_train', 'swap_64_to_128_complex'] )
    now = datetime.datetime.now()
    date_and_time = now.strftime('%Y%m%d_%H%M')
    file_signature = date_and_time + '_' + str(init_seed)
    dir_path = results_path + file_signature + '/'
    makedirs(dir_path)
    
    f = open(dir_path + 'metadata.md', 'w')
    f.write(now.strftime('%Y-%m-%d %H:%M') + '\n\n')
    f.write('Complex NN\n\n')
    f.write('{}\n\n'.format(run_options))
    
    for x, y in metadata.items():
        f.write('{}: {}\n'.format(x, y))
    f.close()
    
    return dir_path

#========================

def dataloader_CNN(X, y, batch, radius, enableShuffle):
    size = X.shape[1]
    batch = min(batch, size)
    d = size / batch
    if d%1 > 0.2:
        batch = np.floor(size / np.ceil(d)).astype(int)
    batch = batch - batch%2
    number_of_bathes = np.floor(size / batch).astype(int)
    
    data = nd.empty((number_of_bathes, 2, batch))
    label = nd.empty((number_of_bathes, 2, batch - 2*radius))
    for i in range(number_of_bathes):
        data[i,:,:] = nd.array(X[:,i*batch:(i+1)*batch])
        for j in range(2):
            label[i,j:(j+1),:] = nd.array(y[j:(j+1),i*batch + radius:(i+1)*batch - radius])
    dataset = gluon.data.dataset.ArrayDataset(data, label)
    return gluon.data.DataLoader(dataset, batch_size=1, shuffle=enableShuffle), batch, number_of_bathes

#========================

time_start = time.time()

X_train = np.zeros((train_files * data_size, 2), dtype=real_t)
y_train = np.zeros((train_files * data_size, 2), dtype=real_t)
y_train_ber = np.zeros((train_files * sample_size, 2), dtype=real_t)
X_test = np.zeros((test_files * data_size, 2), dtype=real_t)
y_test = np.zeros((test_files * data_size, 2), dtype=real_t)
y_test_ber = np.zeros((test_files * sample_size, 2), dtype=real_t)

columns = ['tx_re', 'tx_im', 'rx_re', 'rx_im']

for i in range(train_files):
    data = pd.read_csv(rawdata_path + str(i + 1) + '.csv', names=columns).to_numpy(dtype=real_t) 
    X_train[i * data_size:(i + 1) * data_size,:] = data[:,2:4]
    y_train[i * data_size:(i + 1) * data_size] = data[:,0:2]
    y_train_ber[i * sample_size:(i + 1) * sample_size] = data[radius:data_size-radius,0:2]
    
for i in range(test_files):
    data = pd.read_csv(rawdata_path + str(train_files + i + 1) + '.csv', names=columns).to_numpy(dtype=real_t) 
    X_test[i * data_size:(i + 1) * data_size,:] =data[:,2:4]
    y_test[i * data_size:(i + 1) * data_size] = data[:,0:2]
    y_test_ber[i * sample_size:(i + 1) * sample_size] = data[radius:data_size-radius,0:2]

del data

X_train_loader = np.transpose(X_train)
y_train_loader = np.transpose(y_train)
X_test_loader = np.transpose(X_test)
y_test_loader = np.transpose(y_test)

[train_dataloader_CNN, batch_train, number_of_bathes_train] = dataloader_CNN(X_train_loader, y_train_loader, batch_size, radius, True)
[test_dataloader_CNN, batch_test, number_of_bathes_test]  = dataloader_CNN(X_test_loader, y_test_loader, batch_size, radius, False)

del X_train_loader, y_train_loader, X_test_loader, y_test_loader

time_end = time.time()
print('Time elapsed: {}'.format(time_end - time_start))

print(X_test.shape)
print(y_test.shape)

#========================

class ComplexDenseCNN_v1(gluon.Block):
    def __init__(self, channels, units, inputs, kernel_size, strides=1, **kwargs):
        super(ComplexDenseCNN_v1, self).__init__()
        self.units = units
        self.inputs = inputs
        self.channels = channels
        with self.name_scope():
            self.conv_dense_re = gluon.nn.Conv1D(channels=units, in_channels=inputs, groups=1, kernel_size=kernel_size, strides=strides, activation=None, use_bias=False)
            self.conv_dense_im = gluon.nn.Conv1D(channels=units, in_channels=inputs, groups=1, kernel_size=kernel_size, strides=strides, activation=None, use_bias=False)

    def forward(self, z):
        u = self.conv_dense_re(z[:,0:self.inputs,:]) - self.conv_dense_im(z[:,self.inputs:2*self.inputs,:])
        v = self.conv_dense_im(z[:,0:self.inputs,:]) + self.conv_dense_re(z[:,self.inputs:2*self.inputs,:])
        
        return nd.concat(u, v, dim=1)

#========================

class KerrActivationEnhanced_1ch_v1(gluon.Block):
    def __init__(self, units, **kwargs):
        super(KerrActivationEnhanced_1ch_v1, self).__init__()
        self.units = units
        with self.name_scope():
            self.conv_intra = self.params.get('conv_intra', allow_deferred_init=True, shape=(1,))

    def forward(self, z):
        power = self.conv_intra.data() * (nd.square(z[:,0:self.units,:]) +  nd.square(z[:,self.units:,:]))
        
        cos = nd.cos(power)
        sin = nd.sin(power)

        u = cos * z[:,0:self.units,:] - sin * z[:,self.units:,:]
        v = cos * z[:,self.units:,:] + sin * z[:,0:self.units,:]

        return nd.concat(u, v, dim=1)
        
        
class KerrActivationEnhanced_1ch_v2(gluon.Block):
    def __init__(self, units, **kwargs):
        super(KerrActivationEnhanced_1ch_v2, self).__init__()
        self.units = units
        with self.name_scope():
            self.conv_intra = gluon.nn.Conv1D(channels=units, in_channels=units, groups=units, kernel_size=1, strides=1, activation=None, use_bias=False)

    def forward(self, z):
        power = self.conv_intra(nd.square(z[:,0:self.units,:]) +  nd.square(z[:,self.units:,:]))

        cos = nd.cos(power)
        sin = nd.sin(power)

        u = cos * z[:,0:self.units,:] - sin * z[:,self.units:,:]
        v = cos * z[:,self.units:,:] + sin * z[:,0:self.units,:]
        
        return nd.concat(u, v, dim=1)

#========================

ctx = gpu(0)

net = gluon.nn.Sequential()
with net.name_scope():
    net.add(ComplexDenseCNN_v1(channels=2,units=neuron_hidden_layer,inputs=1,kernel_size=2*radius+1))
    net.add(KerrActivationEnhanced_1ch_v1(units=neuron_hidden_layer))
    net.add(ComplexDenseCNN_v1(channels=2,units=neuron_hidden_layer,inputs=neuron_hidden_layer,kernel_size=1))
    net.add(KerrActivationEnhanced_1ch_v1(units=neuron_hidden_layer))
    net.add(ComplexDenseCNN_v1(channels=2,units=1,inputs=neuron_hidden_layer,kernel_size=1))
    net.add(KerrActivationEnhanced_1ch_v1(units=1))

mse = gluon.loss.L2Loss()

model_saved = False
min_loss = 100
decay_steps = 50
tolerance = 1e-3
stopLR = 1e-4

if run == 7 or run == 8 or run == 9 or run == 10 or run == 11 or run == 13 or run == 14 or run == 15 or run == 16 or run == 17:
    decay_steps = 100
if run == 12 or run == 30:
    decay_steps = 200
if run == 18:
    decay_steps = 150
if run == 19 or run == 22 or run == 23 or run == 24 or run == 25 or run == 27 or run == 28 or run == 32 or run == 33 or run == 35 or run == 36:
    decay_steps = 250
if run == 20 or run == 31:
    decay_steps = 300
if run == 21 or run == 34:
    decay_steps = 350
if run == 26:
    decay_steps = 400

if run == 8:
    tolerance = 5e-4
if run == 9 or run == 12 or run == 13 or run == 14 or run == 15 or run == 16 or run == 17 or run == 18 or run == 19 or run == 20 or run == 21 or run == 24 or run == 25 or run == 26 or run == 27  or run == 30 or run == 31 or run == 34 or run == 35 or run == 36:
    tolerance = 1e-4
if run == 10 or run == 23 or run == 33:
    tolerance = 5e-5
if run == 11 or run == 28 or run == 29:
    tolerance = 1e-5
if run == 22 or run == 32:
    tolerance = 2e-4


if run == 14 or run == 25 or run == 36:
    stopLR = 5e-5
if run == 16:
    stopLR = 5e-4
if run == 24  or run == 35:
    stopLR = 2e-4

steps_wo_min = 0

if continue_train:
    net.load_parameters(model_path + model_filename, ctx=ctx)
    
    f = open(model_path + 'learning_rate.dat', 'r')
    lrate = float(f.read())
    f.close()
    f = open(model_path + 'reached_loss.dat', 'r')
    min_loss = float(f.read())
    f.close()
    f = open(model_path + 'steps_wo_min.dat', 'r')
    steps_wo_min = float(f.read())
    f.close()
else:
    net.initialize(init=mx.init.Normal(sigma=0.05), ctx=ctx)
    
trainer = gluon.Trainer(net.collect_params(), 'Adam', {'learning_rate': lrate})
if continue_train:
    trainer.load_states(model_path + 'optimizer.state')

#========================

dir_path = create_metadata()

filename = os.path.join(dir_path, model_filename)
filename_optimizer = os.path.join(dir_path, 'optimizer.state')

f = open(dir_path + 'learning_rate.dat', 'w')
f.write('{}\n'.format(lrate))
f.close()

f = open(dir_path + 'reached_loss.dat', 'w')
f.write('{}\n'.format(min_loss))
f.close()

f = open(dir_path + 'nn_train.dat', 'a+')

glob_start = time.time()

for epoch in range(epochs):
    train_loss = nd.zeros(1, ctx=ctx)
    tic = time.time()
    for data, label in train_dataloader_CNN:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = mse(output, label)
            loss = nd.mean(loss)
        loss.backward()
        trainer.step(1)
        train_loss += loss.mean().asscalar()
    
    if (epoch) % 5 == 0:
        print('epoch', epoch + 1, '-- loss', train_loss.asscalar(), '-- time', time.time()-tic)
        f.write('epoch {} -- loss {} -- time {}\n'.format(epoch + 1, train_loss.asscalar(), time.time()-tic))
        
    if min_loss > train_loss.asscalar():
        # if (min_loss - train_loss.asscalar())/min_loss < tolerance:
        if (min_loss - train_loss.asscalar())/min_loss < tolerance * lrate / 0.001:
            steps_wo_min += 1
        else:
            steps_wo_min = 0
        lose_p = (min_loss - train_loss.asscalar())/min_loss
        min_loss = train_loss.asscalar()
        if save_model_best:
            net.save_parameters(filename)
            trainer.save_states(filename_optimizer)
            flr = open(dir_path + 'reached_loss.dat', 'w')
            flr.write('{}\n'.format(min_loss))
            flr.close()
            model_saved = True
            print('epoch', epoch + 1, '-- Model saved', '-- loss', train_loss.asscalar(), '-- decay steps', steps_wo_min, '-- tolerance ', lose_p)
            f.write('epoch {} -- Model saved -- loss {} -- decay steps {} -- tolerance {}\n'.format(epoch + 1, train_loss.asscalar(), steps_wo_min, lose_p))
    else:
        steps_wo_min += 1
        
    if steps_wo_min >= decay_steps:
        steps_wo_min = 0
        lrate /= 2
        flr = open(dir_path + 'learning_rate.dat', 'w')
        flr.write('{}\n'.format(lrate))
        flr.close()
        if lrate < stopLR:
            print('Train finished!')
            break
        trainer.set_learning_rate(lrate)
        print('epoch', epoch + 1, '-- Learning rate', lrate)
        f.write('epoch {} -- Learning rate {}\n'.format(epoch + 1, lrate))
        
    if time.time()-full_time > 72000:
        print('Time!', run, mode)
        f.write('Time!')
        break

f.close()
output.wait_to_read() 

if model_saved == False:
    shutil.rmtree(dir_path)
    f = open(model_path + 'steps_wo_min.dat', 'w')
    f.write('{}\n'.format(steps_wo_min))
    f.close()
    flr = open(model_path + 'learning_rate.dat', 'w')
    flr.write('{}\n'.format(lrate))
    flr.close()
else:
    f = open(dir_path + 'steps_wo_min.dat', 'w')
    f.write('{}\n'.format(steps_wo_min))
    f.close()

glob_end = time.time()

print(time.time() - glob_start)

#========================
if calc_ber:

    glob_end = time.time()
    
    batch_size_n = batch_test - 2*radius

    rx_test = symbols_to_codes_v2(X_test[:,0] + 1j * X_test[:,1])
    tx_test_pred = nd.empty([1, 2, test_files * sample_size], ctx=ctx)
    
    for batch_idx, data in enumerate(test_dataloader_CNN):
        data[0] = data[0].as_in_context(ctx)
        output = net(data[0])
        if (output.shape[2] < batch_size_n):
            tx_test_pred[:,:,batch_idx*batch_size_n:] = output
        else:
            tx_test_pred[:,:,batch_idx*batch_size_n:(batch_idx+1)*batch_size_n] = output

    tx_test_pred = tx_test_pred.as_in_context(mx.cpu(0)).asnumpy()
    tx_test = symbols_to_codes_v2(y_test[:,0] + 1j * y_test[:,1])
    tx_test_nn = symbols_to_codes_v2(y_test_ber[:,0] + 1j * y_test_ber[:,1])
    ber_before_nn = ber_by_codes(tx_test, rx_test)
    ber_after_nn = ber_by_codes(tx_test_nn, symbols_to_codes_v2(tx_test_pred[0,0,:] + 1j * tx_test_pred[0,1,:]))

    # Train BER

    batch_size_n = batch_train - 2*radius

    rx_train = symbols_to_codes_v2(X_train[:,0] + 1j * X_train[:,1])
    tx_train_pred = nd.empty([1, 2, train_files * sample_size], ctx=ctx)
    y_train_shuffled = nd.empty([1, 2, train_files * sample_size], ctx=ctx)
    for batch_idx, data in enumerate(train_dataloader_CNN):
        data[0] = data[0].as_in_context(ctx)
        output = net(data[0])
        y_train_shuffled[:,:,batch_idx*batch_size_n:(batch_idx+1)*batch_size_n] = data[1]
        if (output.shape[2] < batch_size_n):
            tx_train_pred[:,:,batch_idx*batch_size_n:] = output
        else:
            tx_train_pred[:,:,batch_idx*batch_size_n:(batch_idx+1)*batch_size_n] = output

    tx_train_pred = tx_train_pred.as_in_context(mx.cpu(0)).asnumpy()
    y_train_shuffled = y_train_shuffled.as_in_context(mx.cpu(0)).asnumpy()
    tx_train = symbols_to_codes_v2(y_train[:,0] + 1j * y_train[:,1])
    tx_train_nn = symbols_to_codes_v2(y_train_ber[:,0] + 1j * y_train_ber[:,1])
    tx_train_nn_shuffled = symbols_to_codes_v2(y_train_shuffled[0,0,:] + 1j * y_train_shuffled[0,1,:])
    ber_before_nn_train = ber_by_codes(tx_train, rx_train)
    ber_after_nn_train = ber_by_codes(tx_train_nn_shuffled, symbols_to_codes_v2(tx_train_pred[0,0,:] + 1j * tx_train_pred[0,1,:]))

    ###############################################################################
    
    f = open(dir_path + 'metadata.md', 'a')
    f.write('\nTime elapsed: {}\n'.format(glob_end - glob_start))
    
    f.write('Epochs to decay: {}\n'.format(decay_steps))
    if epochs > 0:
        f.write('\nEpochs total: {}\n'.format(epoch + 1))
    else:
        f.write('\nEpochs total: {}\n'.format(0))
    f.write('Minimum loss: {}\n'.format(min_loss))
    
    f.write('\nTest BER:\n==================\n\n')
    f.write('\nBER before NN: {}\n'.format(ber_before_nn))
    f.write('BER after NN: {}\n'.format(ber_after_nn))

    f.write('\nTrain BER:\n==================\n\n')
    f.write('BER before NN: {}\n'.format(ber_before_nn_train))
    f.write('BER after NN: {}\n'.format(ber_after_nn_train))

    f.close()

    print(time.time() - glob_start)