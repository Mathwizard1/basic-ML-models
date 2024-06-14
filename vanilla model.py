import random as rd
from math import exp
import csv
import numpy as np

class neuron_f():
    def sigmoid_f(self, x):
        return (1 / (1 + exp(-x)))
    
    def D_sigmoid_f(self, x):
        return (exp(-x) / ((1 + exp(-x)) ** 2))
    
    def binary_f(self, x, threshold = 0):
        if(x > threshold):
            return 1
        else:
            return 0

    def diff_square(self, x, y):
        return ((x - y) ** 2)
    
    def matrix_mult_2d(self, m1, m2):
        m1_r = len(m1)
        m2_r= len(m2)
        m2_c = len(m2[0])

        #print(m1,"@",m2)
        #if(m1_c == m2_r):
            #print(f"possible {m1_r}x{m1_c} @ {m2_r}x{m2_c} = {m1_r}x{m2_c}")

        prod = []

        for i in range(m1_r):
            prod.append([])
            for j in range(m2_c):
                res = 0
                for k in range(m2_r):
                    res += m1[i][k] * m2[k][j]
                prod[i].append(res)
                #print(f"prod[{i}][{j}] = {prod[i][j]}")

        #print(prod)
        return prod
    
    def rms_err(self, output, target):
        net_err = 0
        n = len(output[0][0]) if (type(output[0][0]) == type([])) else 1
        if(n != 1):
            for i in range(n):
                net_err += self.diff_square(output[0][0][i], target[0][0][i])
        else:
            net_err = self.diff_square(output[0][0], target[0][0])
        return ((net_err ** 0.5) / n)

# find SGD with momentum optimisation?
class Simple_NN(neuron_f):
    W = []
    B = []
    alpha = 0.95
    trained = 0
    precision = 0.1 # tells how precise the network will be after trained->  trained *(1 - prec * 100) = actual_accuracy

    # initial setup
    def __init__(self, num_input, num_output, alpha = 0.95):
        self.num_input = num_input
        self.num_output = num_output
        self.alpha = alpha

    # add layers given num of layer and cells per layer
    def add_hidden_layers(self, num_hidden, num_cells):
        if(num_hidden != len(num_cells)):
            print("Layers and number of cells per layer don't match")
            exit()

        self.num_hidden_layers = num_hidden
        self.num_h_cells = num_cells

        total_layer = self.num_hidden_layers + 2
        total_cells = []
        total_cells.append(self.num_input)
        total_cells.extend(num_cells)
        total_cells.append(self.num_output)

        for i in range(1, total_layer):
            val = {"weights":[], "bias":[]}
            for j in range(total_cells[i - 1]):
                val["weights"].append([])
                for k in range(total_cells[i]):
                    val["weights"][j].append(self.alpha * rd.random())
                    if(j == 0):
                        val["bias"].append(rd.random())
            self.W.append(val["weights"])
            self.B.append(val["bias"])

    # feed front
    def feed_front(self, input_val):
        temp_val = input_val
        for i in range(len(self.W)):
            temp_val = self.matrix_mult_2d(temp_val, self.W[i])
            for j in range(len(temp_val[0])):
                temp_val[0][j] = self.sigmoid_f(temp_val[0][j] + self.B[i][j]) 
        return temp_val

    # for testing
    def NN_test(self, test_data, test_out, msg, bin_o = "Bin", tcase = 20):
        count = 0.0
        tc = 0
        len_t = len(test_data) if(len(test_data) > 0) else 0
        tcase = 20 if (tcase > len_t) else tcase

        if(len_t == 0):
            print("Error in testing")
            exit()

        for t in range(len_t):
            output_val = self.feed_front(test_data[t])
            if(self.rms_err(output_val, test_out[t]) <= self.precision):
                count = count + 1
            if(tc < tcase and msg == "Test" and rd.randint(1,4) == 2 or self.rms_err(output_val, test_out[t]) <= self.precision):
                tc = tc + 1
                n = len(output_val[0])
                if(bin_o == "Bin"):
                    if(n != 1):
                        for i in range(n):
                            output_val[0][i] = self.binary_f(output_val[0][i], 0.5)
                    else:
                        output_val[0][0] = self.binary_f(output_val[0][0], 0.5)
                print(f"o/P->{output_val} targ->{test_out[t]}")  

        self.trained = (count / len_t * 100) * (1 - self.precision) if (self.precision < 1) else 0
        print(f"testing done, trained = {self.trained}%")

    # for training msg = "Test"/"Notest" tescase = ("Nobin"/"Bin", num of test cases to see < 20) Use "Nobin"
    def NN_train(self, input_val, output_val, epoch = 1000, batch_size = 10, samp = 0, prec = 0.1, msg = "Notest", testcase = ("Bin", 20)):
        self.precision = prec

        temp_data, temp_out, test_data, test_out = [], [], [], []
        n = round(0.9 * len(input_val))

        temp_data = input_val[:n]
        temp_out = output_val[:n]

        test_data = input_val[n:]
        test_out = output_val[n:]

        for i in range(epoch):
            batch = [] 
            for d in range(batch_size):
                t = rd.randint(0, n - 1)
                batch.append([temp_data[t], temp_out[t]])
            self.back_prop(batch, batch_size)
            if(samp >= 1 and i % int(samp * batch_size) == 0):
                print("sample:",self.rms_err(self.feed_front(temp_data[0]), temp_out[0]))

        print("training done")
        self.NN_test(test_data, test_out, msg, *testcase)

    # recursive chain rule for calculating delta
    def chain_delweight(self, dCdy, layer, node, number, pre_act_vals):
        val = 0
        if(layer + 1 == len(self.W)):
                #print(f"layer {layer+1},node {node+1},weight {number + 1}","-> itself")
                val = dCdy[number] * self.D_sigmoid_f(pre_act_vals[layer + 1][number])
        else:
            for i in range(len(pre_act_vals[layer + 1])):
                if(i == number):
                    for j in range(len(self.W[layer + 1][i])):
                        #print(f"layer {layer+1},node {node+1},weight {number + 1}","->", end = " ")
                        val = val + self.D_sigmoid_f(pre_act_vals[layer + 1][number]) * self.W[layer + 1][i][j] * self.chain_delweight(dCdy, layer + 1, i, j, pre_act_vals)
        return val

    # update weights and biases
    def back_prop(self, batch, batch_size):
        for b in range(batch_size):
            temp_val, target_val = batch[b][0], batch[b][1]
            #print(temp_val, target_val)

            pre_act_val = []
            pre_act_val.append(temp_val[0])
            for i in range(len(self.W)):
                temp_val = self.matrix_mult_2d(temp_val, self.W[i])
                pre_act_val.append([])
                for j in range(len(temp_val[0])):
                    pre_act_val[i + 1].append(temp_val[0][j] + self.B[i][j])
                    temp_val[0][j] = self.sigmoid_f(temp_val[0][j] + self.B[i][j])    

            # diff in outputs and targets
            dCdy = [2 * (temp_val[0][i] - target_val[0][i]) for i in range(len(temp_val[0]))]

            # weights and Bias
            layers = len(self.W)
            for i in range (layers):
                for j in range(len(self.W[i])):
                    for k in range(len(self.W[i][j])):
                        # one weight at a time update
                        wijk = self.W[i][j][k]

                        # find the delta
                        del_bik = del_wijk = self.chain_delweight(dCdy, i, j, k, pre_act_val)

                        if(i == 0):
                            del_wijk = del_wijk * pre_act_val[i][j] 
                        else:
                            del_wijk = del_wijk * self.sigmoid_f(pre_act_val[i][j])

                        self.W[i][j][k] = wijk - self.alpha * del_wijk / batch_size
                        #print(f"\ndone: {i + 1} layer, {j + 1} node, {k+1} weight {wijk} -> {self.W[i][j][k]}\n")
                        if(j == 0):
                            bik = self.B[i][k]
                            self.B[i][k] = self.B[i][k] - self.alpha * del_bik / batch_size
                            #print(f"{i + 1} layer, {k + 1} node bias {bik} -> {self.B[i][k]}")

    # print the weight snd biases
    def print_brain(self, msg = "All"):
        if(msg == "All" or msg == "Weights"):
            print("weights: ")
            for w in self.W:
                w_r = len(w)
                w_c = len(w[0])
                print(w,f"= {w_r}x{w_c}")
        if(msg == "All" or msg == "Biases"):
            print("biases")
            for b in self.B:
                b_r = len(b)
                print(b,f"= {b_r}")
        print()

# used to create the type of network
class ml_brain():
    def __init__(self, NN_type, ml_arg):
        self.NN_type = NN_type
        if(self.NN_type == "simple"):
            self.instance = Simple_NN(*ml_arg)

    def __getattr__(self, atr_name):
        return self.instance.__getattribute__(atr_name)
    
    def save_brain(self):
        model_name = self.NN_type+ (f"_{self.instance.num_input}x{self.instance.num_output}")
        '''with open(model_name+"_param.txt", "w") as fparam:
            fparam.write("NN_type = "+str(self.NN_type))
            fparam.write(self.save_param())
            fparam.close()'''
        print(model_name+"_param successfully saved.")

    def get_output(self, input_val):
        if(self.trained):
            output_val = self.feed_front(input_val)
            n = len(output_val[0])
            if(n != 1):
                for i in range(n):
                    output_val[0][i] = self.binary_f(output_val[0][i], 0.5)
            else:
                output_val[0][0] = self.binary_f(output_val[0][0], 0.5)
            print(f"output of {input_val} =", output_val)
            return output_val
        else:
            print("not trained")

    def get_brain(self, fpath):
        model_name = ""
        with open(fpath, "r") as fparam:
            fparam.close()
        print(model_name+"_param successfully loaded.")            

outp = 1
inp = 2

data = []
out = []

# xor of two numbers in bit form and output in bit form
'''for i in range(1000):
    data.append([[]]) # simple NN trains on data in form of 2x2 lists inside a list -> 3d list
    out.append([[]]) # corresponding outputs should be in -> [[x]] {1x1}  [[1,2],[1,2]] {2x2}  [[1,2]] {1x2}
    num = []
    for j in range(2):
        num.append(rd.randint(10,50))
    res = num[0] ^ num[1]

    for p in range(len(num)):
        bn = num[p]
        for k in range(outp):
            if(bn % 2):
                data[i][0].append(1)
            else:
                data[i][0].append(0)
            bn //= 2

            if(p == 0):
                if(res % 2):
                    out[i][0].append(1)
                else:
                    out[i][0].append(0)
                res //= 2

br = ml_brain("simple", (inp, outp))
br.instance.add_hidden_layers(1,[1])
br.NN_train(data, out, 9000, 10, samp = 1, msg = "Test", testcase = ("Nobin", 10))'''
#br.print_brain()




