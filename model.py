import gc

import random as rd
import pandas as pd
import matplotlib as mt
import numpy as np
import scipy as sp
from functools import partial as ft_partial
import csv

#print("Garbage collection thresholds:", gc.get_threshold())

#--- Activation functions ---#
class neuron_numpy_f():
    def _sigmoid_f(self, x):
        return (np.reciprocal(1 + np.exp(-x)))
    def D_sigmoid_f(self, x):
        return (self._sigmoid_f(x) * (1 - self._sigmoid_f(x)))

    def _tanh_f(self, x):
        return (np.tanh(x))
    def D_tanh_f(self, x):
        return (1 - (self._tanh_f(x) ** 2))
    
    def _softplus_f(self, x):
        return(np.log(1 + np.exp(x)))
    def D_softplus_f(self, x):
        return (self._sigmoid_f(x))

    def _relu_f(self, x):
        return np.maximum(x, 0)
    def D_relu_f(self, x):
        return (self._relu_f(np.sign(x)))

    def _softmax_f(self, x):
        pass

    # Not activatable
    def _binary_f(self, x, threshold = 0):
        return(np.maximum(np.sign(x - threshold), 0))

#--- Loss functions ---#
class loss_numpy_f():
    def _mean_sqred_errf(self, y_output, y_target):
        return (((y_output - y_target) ** 2).mean())

    def D_mean_sqred_errf(self, y_output, y_target):
        return (2 * (y_output - y_target))

    # Only for measuring loss
    def _rms_errf(self, y_output, y_target):
        return (np.sqrt(self._mean_sqred_errf(y_output, y_target)))


#--- gradient descent functions ---#
class SGD():
    def _SGD(self, param, D_param):
        return (param + (-D_param))

class SGD_momentum():
    beta = 0
    prev_del_Z = []

    def __init__(self, beta):
        self.beta = beta

    def _SGD_momentums(self):
        pass

class gradient_f():
    del_Z = []

    def __init__(self, grad_func, args = ()):
        self.grad_func = grad_func
        self.args = args

        if(self.grad_func == "_SGD"):
            self.instance = SGD()
        elif(self.grad_func == "_SGD_momentum"):
            if(len(args) == 1 and isinstance(args[0], float) and args[0] < 1):
                self.instance = SGD_momentum(*args)
            else:
                self.error_exit("1")
        else:
            self.error_exit("0")

    # compute the gradient matrix for a layer
    def actv_grad_f(self, grad_size, all_layers, layer, Weight, Bias, A0):
        del_B = np.divide(self.del_Z[all_layers - layer - 1], grad_size)
        del_W = np.divide(np.outer(self.del_Z[all_layers - layer - 1].T, A0).T, grad_size)

        # activate the respective gradient algorithm
        return (self.instance.__getattribute__(self.grad_func)(Weight, del_W), self.instance.__getattribute__(self.grad_func)(Bias, del_B))

    def error_exit(self, exit_num = "-1"):
        if(exit_num == "0"):
            print(self.grad_func,"not in gradient function list")
        if(exit_num == "1"):
            print("Arguments not given properly")
        else:
            print("Unknown cause of error")
        exit()

# Use seeded random numbers
def rd_seed(num):
    rd.seed(num)
    np.random.seed(num)

# simple artificial NN
class Simple_NN():
    W = []
    B = []
    trained = 0.0
    alpha = 0.95 # learning rate
    precision = 0.1 # tells how precise the network will be after trained->  trained *(1 - prec * 100) = actual_accuracy

    # initial setup
    '''
    num_input -> number of input nodes
    num_output -> number of output nodes
    output_type -> mapping, mathematical, probabilistic
    '''
    def __init__(self, num_input, num_output, output_type = "mapping", alpha = 0.95, prec = 0.1):
        self.num_input = num_input
        self.num_output = num_output
        self.output_type = output_type
        self.alpha = alpha
        self.precision = prec

    # in case of improper inputs
    def ANN_err_exits(self, exit_num = "-1", args = ()):
        if(exit_num == "0"):
            print("Setup inputs of NN are invalid")
        if(exit_num == "0a"):
            print("Setup of output_type of NN is invalid")
        if(exit_num == "1"):
            print("Layers and number of cells per layer don't match")
        elif(exit_num == "1a"):
            print("Activation function not properly mentioned directly or in a list")
        elif(exit_num == "1ab"):
            print("Layers and number of activation functions per layer don't match")
        elif(exit_num == "1ac"):
            print("For all neurons to have same activation function, use string directly. ex: '_sigmoid_f'")
        elif(exit_num == "1ad"):
            print("Activation function must be string. ex: ['_sigmoid_f', '_relu_f']")
        elif(exit_num == "1b"):
            print(*args,"not in Activation functions list.")
        elif(exit_num == "1c"):
            print("Loss function must be string. '_mean_sqred_errf'")
        elif(exit_num == "1ca"):
            print(*args,"not in Loss functions list")
        elif(exit_num == "2a"):
            print("Input/output, Data should be in list or tuple form")
        elif(exit_num == "2b"):
            print("Provide at least 10 data samples")
        elif(exit_num == "2c"):
            print("Epoch or batch_size not set correctly")
        elif(exit_num == "2d"):
            print("Data cannot be trained on, check input and output dimensions of inp/outp data")
        elif(exit_num == "3a"):
            print("Error in test data")
        else:
            print("Unknown cause of exit")
        exit()

    # to set the activation function(s) for all the neurons
    def set_neuron_actv_f(self, num_hidden, neuron_func):
        temp_neuron_f = neuron_numpy_f()
        all_neuron_funcs = dir(temp_neuron_f)

        self.actv_funcs = {}
        self.actv_funcs_ord = [-1]

        if(isinstance(neuron_func, str)):
            _neuron_func = neuron_func
            D_neuron_func = "D" + _neuron_func

            if(_neuron_func in all_neuron_funcs and D_neuron_func in all_neuron_funcs):
                self.actv_funcs_ord.append(_neuron_func)
                self.actv_funcs[_neuron_func] = getattr(temp_neuron_f, _neuron_func)
                self.actv_funcs[D_neuron_func] = getattr(temp_neuron_f, D_neuron_func)
            else:
                self.ANN_err_exits("1b", (_neuron_func, D_neuron_func))

        elif(isinstance(neuron_func, (list, tuple))):
            if(len(neuron_func) == num_hidden):
                self.actv_funcs_ord[0] = num_hidden

                for n_funcs in range(num_hidden):
                    _neuron_func = neuron_func[n_funcs]
                    
                    if(isinstance(_neuron_func, str)):
                        D_neuron_func = "D" + _neuron_func

                        if(_neuron_func in all_neuron_funcs and D_neuron_func in all_neuron_funcs):
                            self.actv_funcs_ord.append(_neuron_func)
                            if(_neuron_func not in self.actv_funcs):
                                self.actv_funcs[_neuron_func] = getattr(temp_neuron_f, _neuron_func)
                                self.actv_funcs[D_neuron_func] = getattr(temp_neuron_f, D_neuron_func)
                        else:
                            self.ANN_err_exits("1b", (_neuron_func, D_neuron_func))
                    else:
                        self.ANN_err_exits("1ad")
            elif(len(neuron_func) == 1 and num_hidden != 1):
                self.ANN_err_exits("1ac")
            else:
                self.ANN_err_exits("1ab")
        else:
            self.ANN_err_exits("1a")

        # set the activations of last layer
        if(self.output_type == "mapping"):
            self.actv_funcs_ord.append("_sigmoid_f")
            if("_sigmoid_f" not in self.actv_funcs):
                self.actv_funcs["_sigmoid_f"] = getattr(temp_neuron_f, "_sigmoid_f")
                self.actv_funcs["D_sigmoid_f"] = getattr(temp_neuron_f, "D_sigmoid_f")
        elif(self.output_type == "mathematical"):
            self.actv_funcs_ord.append("_relu_f")
            if("_relu_f" not in self.actv_funcs):
                self.actv_funcs["_relu_f"] = getattr(temp_neuron_f, "_relu_f")
                self.actv_funcs["D_relu_f"] = getattr(temp_neuron_f, "D_relu_f")
        #elif(self.output_type == "probabilistic"):
        else:
            self.ANN_err_exits("0a")

        # set the activation functions and its nth derivatives
        self.actv_f = ft_partial(self.layer_actv_f, D = "")
        self.D_actv_f = ft_partial(self.layer_actv_f, D = "D")

    # if multiple layers have different activation function
    '''
    x -> value to be activated
    layer_n -> layer number
    D -> for which differential to use of that layer's function
    ex: D_f -> 1st differential, DD_f -> 2nd differential
    '''
    def layer_actv_f(self, x, layer_n, D):
        if(self.actv_funcs_ord[0] == -1):
            if(layer_n < self.num_hidden_layers - 1):
                layer_n = 0
            else:
                layer_n = 1

        return self.actv_funcs[D + self.actv_funcs_ord[layer_n + 1]](x)

    # to set the loss function
    def set_neuron_loss_f(self, loss_func):
        if(isinstance(loss_func, str)):
            temp_loss_funcs = loss_numpy_f()
            all_loss_funcs = dir(temp_loss_funcs)
            _loss_func = loss_func
            D_loss_func = "D" + _loss_func

            if(_loss_func in all_loss_funcs and D_loss_func in all_loss_funcs):
                self.n_loss_f = getattr(temp_loss_funcs, _loss_func)
                self.D_n_loss_f = getattr(temp_loss_funcs, D_loss_func)
            else:
                self.ANN_err_exits("1ca")
        else:
            self.ANN_err_exits("1c")
        pass

    # add layers given num of layer and cells per layer
    def add_hidden_layers(self, num_hidden, num_cells, neuron_func = "_sigmoid_f", loss_func = "_mean_sqred_errf"):
        if(num_hidden != len(num_cells)):
            self.ANN_err_exits("1")

        # prepare the activation and loss function
        self.set_neuron_actv_f(num_hidden, neuron_func)
        self.set_neuron_loss_f(loss_func)

        # store total hidden layers
        self.num_hidden_layers = num_hidden + 1

        # total cells and layers used
        total_layer = self.num_hidden_layers + 1
        total_cells = []
        total_cells.append(self.num_input)
        total_cells.extend(num_cells)
        total_cells.append(self.num_output)

        # create the random weights
        for i in range(1, total_layer):
            self.W.append(self.alpha * np.random.rand(total_cells[i - 1], total_cells[i]))
            self.B.append(self.alpha * np.random.rand(1, total_cells[i]))

    # feed front (returns next layer values) (activated/ non-activated)
    def feed_front(self, input_n, layer_i = 0, layer_f = -1, actv_signal = True):
        result = input_n
        if(layer_f == -1):
            layer_f = self.num_hidden_layers

        for layer in range(layer_i,layer_f):
            result = result @ self.W[layer] + self.B[layer]
            if(actv_signal):
                result = self.actv_f(x = result, layer_n = layer)
        return result

    # for training 
    '''
    samp = ("plot" / "data", samp_num (int))
    msg = "Test"/"Notest" 
    '''
    def NN_train(self, input_data, output_data, epoch = 1000, batch_size = 10, grad_func = "_SGD", samp = (), msg = "Notest"):
        if(not(isinstance(input_data, (list, tuple)) or isinstance(output_data, (list, tuple)))):
            self.ANN_err_exits("2a")
        elif(len(input_data) < 10):
            self.ANN_err_exits("2b")
        elif(not (isinstance(epoch, int) and isinstance(batch_size, int)) or (epoch <= 0 or batch_size <= 0)):
            self.ANN_err_exits("2c")
        
        # prepare the gradient object
        n_grad_obj = gradient_f(grad_func)

        temp_data, temp_out, test_data, test_out = [], [], [], []
        n = round(0.9 * len(input_data))

        temp_data = input_data[:n]
        temp_out = output_data[:n]
        test_data = input_data[n:]
        test_out = output_data[n:]

        data_0 = np.array(temp_data[0], ndmin=2)
        target_0 = np.array(temp_out[0], ndmin=2)

        if(len(data_0.flatten()) != self.num_input or (self.feed_front(data_0)).shape != target_0.shape or len(target_0.flatten()) != self.num_output):
            self.ANN_err_exits("2d")

        # if sample is asked
        samples, samp_size = [], 0
        if(isinstance(samp, (tuple, list)) and len(samp) == 2 and isinstance(samp[1], int) and samp[1] >= 1):
            samp_size = int(epoch / batch_size * samp[1])
            if(samp_size > 100):
                samp_size = int(epoch / 100)
        else:
            samp_size = 0

        # for each epoch train NN using backprop
        for i in range(epoch):
            batch = [] 
            for d in range(batch_size):
                t = rd.randint(0, n - 1)
                batch.append([np.array(temp_data[t], ndmin=2), np.array(temp_out[t], ndmin=2)])
            self.back_prop(batch, batch_size, n_grad_obj)

            if(samp_size != 0 and i % samp_size == 0):
                samples.append(self.n_loss_f(self.feed_front(data_0), target_0))

        # For plotting or printing the data of samples
        if(samp_size != 0 and len(samp) == 2 and isinstance(samp[0], str)):
            print("Loss function used:", self.n_loss_f.__name__)
            if(samp[0] == "plot"):
                pass
            else:
                for sample_num in range(len(samples)):
                    print(f"epoch {(sample_num + 1) * samp_size}:",samples[sample_num])

        # to signal next phase
        print("training done")

        # do the testing if needed
        self.NN_test(test_data, test_out, msg)

    # update weights and biases
    def back_prop(self, batch, batch_size, grad_obj):
        for b in range(batch_size):
            input_n, output_n = batch[b][0], batch[b][1]
            result = input_n

            # list to hold all pre_activated values
            pre_act_val = []
            pre_act_val.append(result)

            # step by step non-activated feed front
            for layer in range(self.num_hidden_layers):
                result = self.feed_front(result, layer, layer + 1, False)
                pre_act_val.append(result)
                result = self.actv_f(result, layer)

            # Loss in outputs and targets
            dL_dy = self.D_n_loss_f(result, output_n)

            # empty the del_Z matrix list
            grad_obj.del_Z = []

            # weights and Bias update depending on the gradient descent algorithm
            for layer in range(self.num_hidden_layers):
                if(len(grad_obj.del_Z) == 0):
                    Zf = np.multiply(dL_dy, self.D_actv_f(pre_act_val[self.num_hidden_layers - layer], self.num_hidden_layers - layer - 1))
                    grad_obj.del_Z.append(Zf)
                else:
                    Kn = grad_obj.del_Z[layer - 1] @ self.W[self.num_hidden_layers - layer].T
                    Zn = np.multiply(Kn, self.D_actv_f(pre_act_val[self.num_hidden_layers - layer], self.num_hidden_layers - layer - 1))
                    grad_obj.del_Z.append(Zn)
                    self.W[self.num_hidden_layers - layer], self.B[self.num_hidden_layers - layer] = grad_obj.actv_grad_f(batch_size, self.num_hidden_layers, self.num_hidden_layers - layer, self.W[self.num_hidden_layers - layer], self.B[self.num_hidden_layers - layer], self.actv_f(pre_act_val[self.num_hidden_layers - layer], self.num_hidden_layers - layer - 1))

            # starting layer update (no activation of input)
            self.W[0], self.B[0] = grad_obj.actv_grad_f(batch_size / self.alpha, self.num_hidden_layers, 0, self.W[0], self.B[0], pre_act_val[0])

    # for testing
    def NN_test(self, test_data, test_out, msg):
        count = 0.0
        tc = 0
        len_t = len(test_data) if(len(test_data) > 0) else 0
        tcase = 20 if (len_t > 20) else len_t

        if(len_t == 0):
            self.ANN_err_exits("3a")

        for t in range(len_t):
            test_out[t] = np.array(test_out[t], ndmin= 2)
            output_val = self.feed_front(np.array(test_data[t], ndmin= 2))

            if(self.n_loss_f(output_val, test_out[t]) < self.precision):
                count = count + 1

            if(msg == "Test" and tc < tcase and rd.randint(0,1)):
                if(self.output_type == "mapping"):
                    temp_func_holder = neuron_numpy_f()
                    output_val = temp_func_holder._binary_f(output_val, 2 * self.precision)
                print(f"o/P->{output_val} targ->{test_out[t]}") 
                tc += 1 

        self.trained = (count / len_t * 100) * (1 - self.precision) if (0 < self.precision < 1) else 0
        self.trained = round(self.trained, 3)
        print(f"testing done, trained = {self.trained}%")

    # print the weight snd biases ("All", "Weights", "Biases") are msg param
    def print_brain(self, msg = "All"):
        print("Brain parameters:")
        if(msg == "All" or msg == "Weights"):
            print("Weights: ")
            for w in range(self.num_hidden_layers):
                print(f"{w + 1} layer\n{self.W[w]} -> {self.W[w].shape}")
        print()
        if(msg == "All" or msg == "Biases"):
            print("Biases")
            for b in range(self.num_hidden_layers):
                print(f"{b + 1} layer {self.B[b]} -> {self.B[b].shape}")
        print()

# Convolutional NN
class Convolutional_NN(neuron_numpy_f):
    W = []
    B = []
    op_order = []   # to store Kernels and Samplers
    trained = 0
    alpha = 0.95    # learning rate
    precision = 0.1 # tells how precise the network will be after trained->  trained *(1 - prec * 100) = actual_accuracy

    _feed_layer = False

    # initial setup
    def __init__(self, size_input, size_output, output_type, alpha = 0.95):
        self.size_input = size_input
        self.alpha = alpha

    def add_Kernel(self):
        pass

    def add_sampler(self):
        pass

    def set_feed_layer(self):
        self._feed_layer = True

# used to create the type of neural network
class ml_brain():
    def __init__(self, NN_type, ml_arg):
        self.NN_type = NN_type
        if(self.NN_type == "simple"):
            self.instance = Simple_NN(*ml_arg)
        elif(self.NN_type == "convolutional"):
            self.instance = Convolutional_NN(*ml_arg)

    def __getattr__(self, atr_name):
        return self.instance.__getattribute__(atr_name)
    
    def save_brain(self, name):
        model_name = name + self.NN_type
        '''with open(model_name+"_param.txt", "w") as fparam:
            fparam.write("NN_type = "+str(self.NN_type))
            fparam.write(self.save_param())
            fparam.close()'''
        print(model_name+"_param successfully saved.")

    def get_output(self, input_val):
        if(self.instance.trained > 0):
            input_val = np.array(input_val, ndim = 2)
            output_val = self.feed_front(input_val)

            print(f"output of {input_val} =", output_val)
            return output_val
        else:
            print("not trained")

    def load_brain(self, fpath):
        model_name = ""
        with open(fpath, "r") as fparam:
            fparam.close()
        print(model_name+"_param successfully loaded.")            

inp = 6
outp = 3
inp_data = []
outp_data = []

#seed_num = 100
#rd_seed(seed_num)

for i in range(2000):
    inp_data.append([]) # simple NN trains on data in form of 2x2 lists inside a list -> 3d list
    outp_data.append([]) # corresponding outputs should be in -> [[x]] {1x1}  [[1,2],[1,2]] {2x2}  [[1,2]] {1x2}
    num = []
    for j in range(2):
        num.append(rd.randint(10,50))
    res = num[0] ^ num[1]

    for p in range(len(num)):
        bn = num[p]
        for k in range(outp):
            if(bn % 2):
                inp_data[i].append(1)
            else:
                inp_data[i].append(0)
            bn //= 2

            if(p == 0):
                if(res % 2):
                    outp_data[i].append(1)
                else:
                    outp_data[i].append(0)
                res //= 2

def main():
    br = ml_brain("simple", (inp, outp))
    br.add_hidden_layers(2, [4,4], "_sigmoid_f")
    br.NN_train(inp_data, outp_data, 5000, 25, msg = "Test")
    #br.print_brain()

main()