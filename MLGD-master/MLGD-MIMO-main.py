# import numpy as np
# import random
# import matplotlib.pyplot as plt
# import torch
# import pickle
import numpy as np
import torch
import torch.nn as nn
from utils_9_6 import *
from timeit import default_timer as timer
import os
import xlwt


# H的随机性
# 用 Schulz iterative approach 代替WMMSE
# LAGD的更新本身就可以考虑梯度更新的方式比如加动量,尝试一下
# 用元学习多步更新LAGD
# 离线训练


# 做一个缩小变化的 LAGD_lr
# 加一个随机初始化的值，加一个随机更新的梯度方向标签


print_step = 100

K = 4  # users number, 4 8 10 20 30 ... 70
SNR = 10 # 信噪比
sigma = 1 # nosie power 的开方

PT = int(10 ** (SNR / 10)) # total power

epsilon = 1e-5
d = 2
Nr = 2  # a user equipped with Nr receive antennas, fix to 2
Nt = 8  # a BS equipped with Nt transmit antennas, 8 16 32 64 128 256
# Imax = 100  # The maximum number of iterations
err = float("inf")

rate = []
rate_np = []
test_num = 50 # epoch
max_iterations = 2000 # 500
WMMSE_iterations = 10 # 20
Scale_V_steps = 1 #迭代n步投影一次

# Network settings
network = 'FCN' # 网络类型选择 ‘FCN’ or ‘LSTM’
N_layers = 1 # 网络隐藏层层数
input_size_V = Nt * Nr * 2  # 可以是 K, Nt, Nr, 2这四个数相乘的任意组合, 默认Nt * Nr * 2 (不乘K表示权值共享、并行计算)
hidden_size_V = 50  #(input_size_V) // 2  # 网络隐藏层神经元个数，可以设小一点
output_size_V = input_size_V
optimizer_lr_V = 1e-3  # LAGD网络的更新学习率
LAGD_lr = 1e-1   # LAGD 每更新一步的步长
rho = 0 #1e-3
beta = 0.9
gradient_old = torch.ones(1).to(device)
gradient = torch.ones(1).to(device)
# gradient = gradient.to(device)
# gradient_old = gradient_old.to(device)
Si = torch.ones(1).to(device)
# Si = Si.to(device)
e=1e-5

col1 = 1
col2 = 2
list1 = [[1 for j in range(50)] for i in range(2)]  # 创建一个二维列表
wb = xlwt.Workbook()  # 一个实例
sheet = wb.add_sheet("Sheet1")  # 工作簿名称
style = "font:colour_index black;"  # 设置样式
black_style = xlwt.easyxf(style)
sheet.write(0,1,"LAGD")
sheet.write(0,2,"WMMSE")
for i in range(1,51):
    sheet.write(i, 0, str(i))  # 不设置格式
    wb.save('LAGD_SNR_'+ str(SNR)+'dB'+'_K_'+str(K)+'_Nt_'+str(Nt)+'_Nr_'+str(Nr)+'.xls')  # 最后一定要保存，否则无效


Initial_V_tc_mode = "inherited" # 初始化的V采用 固定的 "fixed", 承接的 "inherited", 随机的 "random"
Learning_strategy = "LAGD"   # V的更新方式  学习更新一步到位"learn_one_step", 学习辅助更新 "LAGD", 学习辅助与WMMSE交替 "LAGD_WMMSE", 学习辅助与GD交替 "LAGD_GD",
LAGD_Strategy = "LAGD" # 当Learning_strategy='LAGD'时, 可选择"LAGD", "LAGD_Adagard", "LAGD_RMSProp", "LAGD_Adam", "LAGD_momentum"


IF_random_label = False  #随机梯度标签 "random_label"
random_label_load_path = "./WMMSE_K_"+ str(K) +"_Nr_" + str(Nr) + "_Nt_" + str(Nt) + "/" + "5dB" # "_dB" or "_-_dB"
random_inner_loop = 20
meta_loop = 1 #外循环迭代n次，LAGD网络参数更新1次，配合承接模式使用
IF_LAGD_solution = True

IF_WMMSE_solution = False
WMMSE_i = 2 # LAGD_WMMSE交替更新的情况 WMMSE的更新次数
LAGD_i = 50  # LAGD_WMMSE交替更新的情况 LAGD的更新次数

IF_GD_solution = False
GD_j = 2 # LAGD_GD交替更新的情况 GD的更新次数
LAGD_j = 2  # LAGD_GD交替更新的情况 LAGD的更新次数

rate_iterations = np.zeros([max_iterations, test_num])
rate_itera_WMMSE = np.zeros([test_num])

class LSTM_Optimizee_V(torch.nn.Module):
    def __init__(self, N_layers):
        super(LSTM_Optimizee_V,self).__init__()
        self.lstm=torch.nn.LSTM(input_size_V,hidden_size_V, N_layers)
        self.out=torch.nn.Linear(hidden_size_V,output_size_V)
    def forward(self, gradient, state_h, state_c):
            gradient = gradient.unsqueeze(0).unsqueeze(0)
            if state_h is None:
                state_h = torch.zeros(N_layers, 1, hidden_size_V).to(device)
                state_c = torch.zeros(N_layers, 1, hidden_size_V).to(device)
            update, (state_h, state_c) = self.lstm(gradient, (state_h, state_c))
            update = self.out(update)
            # update = update.squeeze(0)
            return update, (state_h, state_c)


def LAGD_Optimizee_V(N):
    model=nn.Sequential()
    model.add_module('linear_input', nn.Linear(input_size_V, hidden_size_V))
    model.add_module('relu', nn.ReLU())
    for n in range(N):
        model.add_module('linear_hidden' + str(n), nn.Linear(hidden_size_V, hidden_size_V))
        model.add_module('relu', nn.ReLU())

    model.add_module('linear_output', nn.Linear(hidden_size_V, output_size_V))
    model.add_module('relu', nn.ReLU())

    return model


def generate_random_V_tc():

    V_tc_random = torch.randn([K, Nt, Nr, 2]).to(device)

    tc_trace1 = torch.sum(torch.stack([tc_trace(tc_mul(V_tc_random[i], tc_CT(V_tc_random[i]))) for i in range(K)]), dim=0)
    power1 = tc_trace1[0]
    alpha = torch.sqrt(PT / power1)
    V_tc_random = alpha * V_tc_random

    return V_tc_random


# Update of Uk, Wk, Vk Numpy
def update_Uk(A, U, V, H, K, Nr):
    for k in range(K):
        A_item = np.zeros((Nr, Nr), complex)
        for m in range(K):
            A_item += H[k].dot(V[m]).dot(V[m].conjugate().T).dot(H[k].conjugate().T)
        A[k] = sigma ** 2 / PT * np.sum([np.trace(np.dot(V[i], V[i].conjugate().T)) for i in range(K)]) * np.eye(
            Nr) + A_item
        U[k] = np.linalg.inv(A[k]).dot(H[k]).dot(V[k])
    return U


def update_Wk(U, V, W_old, W, K, d):
    for k in range(K):
        E = np.eye(d) + 0j * np.eye(d) - (U[k].conjugate().T).dot(H[k]).dot(V[k])
        W_old[k] = W[k]
        W[k] = np.linalg.inv(E)
    return W, W_old


def update_Vk(V, W, U, H, K, Nt, sigma):
    for k in range(K):
        B_item = np.zeros((Nt, Nt), complex)
        for m in range(K):
            B_item += (H[m].conjugate().T).dot(U[m]).dot(W[m]).dot(U[m].conjugate().T).dot(H[m])
        B = B_item + sigma ** 2 / PT * np.sum(
            [np.trace(U[i].dot(W[i]).dot(U[i].conjugate().T)) for i in range(K)]) * np.eye(Nt)
        V[k] = np.linalg.inv(B).dot(H[k].conjugate().T).dot(U[k]).dot(W[k])
    # Scale
    alpha = np.sqrt(PT / np.sum([np.trace(np.dot(V[i], V[i].conjugate().T)) for i in range(K)]))
    for k in range(K):
        V[k] = alpha * V[k]
    return V

# Update of Uk, Wk, Vk Tensor complexity


def tc_WSR_compute(A_tc, V_tc, H_tc, K, Nr):

    Eye = torch.cat(((torch.eye(Nr, Nr).unsqueeze(-1)).to(device), (torch.zeros(Nr, Nr).unsqueeze(-1)).to(device)), 2)
    WSR = 0
    for k in range(K):
        A_item_tc = torch.zeros(Nr, Nr, 2).to(device)

        for r in range(K):
            if r != k:
                A_item_tc += tc_mul(tc_mul(tc_mul(H_tc[k], V_tc[r]), tc_CT(V_tc[r])), tc_CT(H_tc[k]))

        A_tc[k] = sigma ** 2 * Eye + A_item_tc
        S_tc = torch.log(torch.sum(tc_det2(tc_mul(tc_mul(tc_mul(tc_mul(H_tc[k], V_tc[k]), tc_CT(V_tc[k])), tc_CT(H_tc[k])), tc_inv(A_tc[k])) + Eye)))

        WSR = WSR + S_tc

    return WSR


def WSR_compute(V, H, K, Nt, Nr):

    H_real = torch.from_numpy(H.real).unsqueeze(-1)
    H_imag = torch.from_numpy(H.imag).unsqueeze(-1)
    H_np_tc = torch.cat((H_real, H_imag), 3)
    H_np_tc = H_np_tc.to(device)

    V_np_tc = torch.zeros(K, Nt, Nr, 2)
    for i in range(K):
        V_real = torch.from_numpy(V[i].real).unsqueeze(-1)
        V_imag = torch.from_numpy(V[i].imag).unsqueeze(-1)
        V_np_tc[i] = torch.cat((V_real, V_imag), 2)
    V_np_tc = V_np_tc.to(device)

    Eye = torch.cat(((torch.eye(Nr, Nr).unsqueeze(-1)).to(device), (torch.zeros(Nr, Nr).unsqueeze(-1)).to(device)), 2)
    WSR = 0
    for k in range(K):
        A_item_tc = torch.zeros(Nr, Nr, 2).to(device)

        for r in range(K):
            if r != k:
                A_item_tc += tc_mul(tc_mul(tc_mul(H_np_tc[k], V_np_tc[r]), tc_CT(V_np_tc[r])), tc_CT(H_np_tc[k]))

        A_tc = sigma ** 2 * Eye + A_item_tc
        S_tc = torch.log(torch.sum(tc_det2(tc_mul(tc_mul(tc_mul(tc_mul(H_np_tc[k], V_np_tc[k]), tc_CT(V_np_tc[k])), tc_CT(H_np_tc[k])), tc_inv(A_tc)) + Eye)))

        WSR = WSR + S_tc

    return WSR




def tc_update_U(A_tc, U_tc, V_tc, H_tc, K, Nr):

    for k in range(K):

        A_item_tc= torch.zeros(Nr, Nr, 2).to(device)
        # A_item_tc1 = torch.zeros(Nr, Nr, 2)

        for m in range(K):

            A_item_tc += tc_mul(tc_mul(tc_mul(H_tc[k], V_tc[m]), tc_CT(V_tc[m])), tc_CT(H_tc[k]))

        tc_trace1 = torch.sum(torch.stack([tc_trace(tc_mul(V_tc[i], tc_CT(V_tc[i]))) for i in range(K)]), dim=0)
        power1 = tc_trace1[0]
        Eye = torch.cat(((torch.eye(Nr, Nr).unsqueeze(-1)).to(device), (torch.zeros(Nr, Nr).unsqueeze(-1)).to(device)), 2)

        A_tc[k] = sigma ** 2 / PT * power1 * Eye + A_item_tc

        U_tc[k] = tc_mul(tc_mul(tc_inv(A_tc[k]), H_tc[k]), V_tc[k])



    return U_tc


def tc_update_Wk(U_tc, V_tc, W_old_tc, W_tc, K, d):

    for k in range(K):

        Eye = torch.cat((torch.eye(d,d).unsqueeze(-1).to(device), torch.zeros(d,d).unsqueeze(-1).to(device)), 2)

        E_tc = Eye - tc_mul(tc_mul(tc_CT(U_tc[k]), H_tc[k]), V_tc[k])
        W_old_tc[k] = W_tc[k]
        W_tc[k] = tc_inv(E_tc)

    return W_tc, W_old_tc


def tc_update_Wk_Schulz(U_tc, V_tc, W_old_tc, W_tc, K, d):

    for k in range(K):

        Eye = torch.cat((torch.eye(d,d).unsqueeze(-1).to(device), torch.zeros(d,d).unsqueeze(-1).to(device)), 2)

        E_tc = Eye - tc_mul(tc_mul(tc_CT(U_tc[k]), H_tc[k]), V_tc[k])
        W_old_tc[k] = W_tc[k]
        # W_tc[k] = tc_inv(E_tc)
        W_tc[k] = tc_mul(W_tc[k].detach(), (2 * Eye - tc_mul(E_tc, W_tc[k].detach())))

    return W_tc, W_old_tc


def tc_update_Vk(V_tc, W_tc, U_tc, H_tc, K, Nt, sigma):

    def tc_trace(A):

        if A.ndim == 3:
            A_real = A[:, :, 0]
            A_imag = A[:, :, 1]

            trace_real = torch.trace(A_real)
            trace_imag = torch.trace(A_imag)

            trace = torch.cat((trace_real.unsqueeze(-1), trace_imag.unsqueeze(-1)), 0)

        return trace

    for k in range(K):
        B_item_tc = torch.zeros(Nt, Nt, 2).to(device)
        for m in range(K):
            B_item_tc += tc_mul(tc_mul(tc_mul(tc_mul(tc_CT(H_tc[m]), U_tc[m]), W_tc[m]), tc_CT(U_tc[m])), H_tc[m])

        tc_trace2 = torch.sum(torch.stack([tc_trace(tc_mul(tc_mul(U_tc[i], W_tc[i]), tc_CT(U_tc[i]))) for i in range(K)]), dim=0)
        power2 = tc_trace2[0]
        Eye = torch.cat((torch.eye(Nt, Nt).unsqueeze(-1).to(device), torch.zeros(Nt, Nt).unsqueeze(-1).to(device)), 2)

        B_tc = B_item_tc + sigma ** 2 / PT * power2 * Eye

        V_tc[k] = tc_mul(tc_mul(tc_mul(tc_inv(B_tc), tc_CT(H_tc[k])), U_tc[k]), W_tc[k])

    # Scale
    tc_trace1 = torch.sum(torch.stack([tc_trace(tc_mul(V_tc[i], tc_CT(V_tc[i]))) for i in range(K)]), dim = 0)
    power1 = tc_trace1[0]

    alpha = torch.sqrt(PT / power1)
    for k in range(K):
        V_tc[k] = alpha * V_tc[k]

    return V_tc

MSEloss = torch.nn.MSELoss().to(device)

if network == 'FCN':
    optimizee_V = LAGD_Optimizee_V(N_layers).to(device)
if network == 'LSTM':
    optimizee_V = LSTM_Optimizee_V(N_layers).to(device)
adam_optimizer_V = torch.optim.Adam(optimizee_V.parameters(), lr=optimizer_lr_V)  # update optimizee with adam

print("optimizee_V is created successfully !")
print(optimizee_V)


for num_test in range(test_num):

    if network == 'FCN':
        optimizee_V = LAGD_Optimizee_V(N_layers).to(device)
    if network == 'LSTM':
        optimizee_V = LSTM_Optimizee_V(N_layers).to(device)
    adam_optimizer_V = torch.optim.Adam(optimizee_V.parameters(), lr=optimizer_lr_V)  # update optimizee with adam

    # initial H Tensor complexity
    mean1 = np.zeros(Nr)
    cov1 = np.eye(Nr)
    data = np.zeros((Nr, Nt)) + 1j * np.zeros((Nr, Nt))
    H = np.zeros((K, Nr, Nt)) + 1j * np.zeros((K, Nr, Nt))
    # V_tensor_grad = torch.zeros(K, Nt, Nr, 2)

    for i in range(K):
        data1 = np.random.multivariate_normal(mean1, cov1, Nt)
        data2 = np.random.multivariate_normal(mean1, cov1, Nt)
        data[0, :] = data1[:, 0] + 1j * data1[:, 1]
        data[1, :] = data2[:, 0] + 1j * data2[:, 1]
        H[i, :, :] = data;

    # filename = save_variable(H, 'H.txt')
    # H = load_variavle('H.txt')

    #initial A、U、W, W_old
    A = []
    U = []
    W = []
    V = []
    W_old = []

    # initial A、U、W, W_old, H Tensor complexity

    H_real = torch.from_numpy(H.real).unsqueeze(-1)
    H_imag = torch.from_numpy(H.imag).unsqueeze(-1)
    H_tc = torch.cat((H_real, H_imag), 3)
    H_tc = H_tc.to(device)

    A_tc = []
    S_tc = []
    U_tc = []
    W_tc = []
    V_tc = []
    W_old_tc = []
    for k in range(K):
        # Numpy
        A.append(np.zeros((Nr,Nr),complex))

        U.append(np.zeros((Nr,d),complex))
        W.append(np.zeros((d,d),complex))
        W_old.append(np.eye(d)+complex("inf"))

        V.append(np.conj(np.matmul(np.linalg.pinv(np.matmul(H[k], np.conj(H[k]).T)), H[k])).T)

        # Tensor
        A_tc.append(torch.zeros((Nr, Nr, 2), dtype = torch.float64).to(device))
        S_tc.append(torch.zeros((1), dtype=torch.float64).to(device))
        U_tc.append(torch.zeros((Nr, d, 2), dtype = torch.float64).to(device))
        W_tc.append(torch.zeros((d, d, 2), dtype = torch.float64).to(device))
        W_old_tc.append(torch.zeros((d, d, 2), dtype = torch.float64).to(device))

        # V.append( np.sqrt(1/2)*(np.random.randn(Nt,Nr)+1j*np.random.randn(Nt,Nr)) )

        V_tc.append(tc_CT(tc_mul(tc_inv(tc_mul(H_tc[k], tc_CT(H_tc[k]))), H_tc[k])))


    # Scale V tensor complexity

    tc_trace1 = torch.sum(torch.stack([tc_trace(tc_mul(V_tc[i], tc_CT(V_tc[i]))) for i in range(K)]), dim = 0)
    tc_trace_np = np.sum([np.trace(np.dot(V[i], V[i].conjugate().T)) for i in range(K)])

    power1 = tc_trace1[0]

    alpha = torch.sqrt(PT / power1)
    alpha_np = np.sqrt(PT / np.sum([np.trace(np.dot(V[i], V[i].conjugate().T)) for i in range(K)]))
    # alpha = torch.sqrt(PT / torch.sum(torch.stack([torch.trace(tc_mul(V_tc[i], tc_CT(V_tc[i]))) for i in range(K)])))

    for k in range(K):  # Scale V_k
        V[k] = alpha_np * V[k]

        V_tc[k] = alpha * V_tc[k]

        # a = tc_mul(H_tc[k], tc_CT(H_tc[k]))

    iteration = []
    sum_rate = []
    sum_rate_np = []
    t = 0

    V_tc = torch.stack(V_tc)

    for k in range(K):
        # A_compare = compare_tc_np(V_tc[k], V[k])
        U_tc[k].requires_grad = False
        W_tc[k].requires_grad = False
    V_tc.requires_grad = True

    V_init_tc = V_tc
    V_init_tc_input = torch.reshape(V_init_tc.clone().detach(), [K * Nt * Nr * 2]).to(device)
    V_init_tc_input = V_init_tc_input.type(torch.float32)
    # Accumulated_loss = torch.zeros(0).to(device)

    V_init_tc.requires_grad = False

    print("\n Epoch", '%.0f' % num_test, ": Start WMMSE Updating:")
    start = timer()
    # ######################################  WMMSE (Numpy)  ##############################################
    # for t in range(WMMSE_iterations):
    #     U = update_Uk(A, U, V, H, K, Nr)
    #     W, W_old = update_Wk(U, V, W_old, W, K, d)
    #     V = update_Vk(V, W, U, H, K, Nt, sigma)
    #
    #     WSR_np = np.sum(np.fromiter([np.log(np.linalg.det(W[k])) for k in range(K)], complex))
    #     WSR_np = WSR_np.real
    #     sum_rate_np.append(WSR_np)
    #     time = timer() - start
    #     ratio = (t+1) / WMMSE_iterations
    #     percent = str(t+1)+'/'+str( WMMSE_iterations)
    #     progress(ratio, percent)
    #     print('->  step :', t + 1, 'WSR(WMMSE) =', '%.2f' % WSR_np, 'time =',
    #           '%.2f' % time, end='')

    for t in range(WMMSE_iterations):
        U = update_Uk(A, U, V, H, K, Nr)
        W, W_old = update_Wk(U, V, W_old, W, K, d)
        V = update_Vk(V, W, U, H, K, Nt, sigma)

        WSR_np_new = WSR_compute(V, H, K, Nt, Nr)

        WSR_np = np.sum(np.fromiter([np.log(np.linalg.det(W[k])) for k in range(K)], complex))
        WSR_np = WSR_np.real
        sum_rate_np.append(WSR_np)
        time = timer() - start
        ratio = (t+1) / WMMSE_iterations
        percent = str(t+1)+'/'+str( WMMSE_iterations)
        progress(ratio, percent)
        print('->  step :', t + 1, 'WSR(WMMSE) =', '%.2f' % WSR_np, 'WSR(WMMSE)_new =', '%.2f' % WSR_np_new, 'time =',
              '%.2f' % time, end='')

    # #################### 初始化V   固定的 "fixed", 承接的 "inherited", 随机的 "random" #######################
    if Initial_V_tc_mode == "fixed":
        V_tc = V_init_tc.clone().detach().requires_grad_(True).to(device)
    elif Initial_V_tc_mode == "inherited":
        V_tc = V_tc.clone().detach().requires_grad_(True).to(device)
    elif Initial_V_tc_mode == "random":
        V_tc_random = generate_random_V_tc()
        V_tc = V_tc_random.clone().detach().requires_grad_(True).to(device)

    U_tc = tc_update_U(A_tc, U_tc, V_tc, H_tc, K, Nr)
    W_tc, W_old_tc = tc_update_Wk(U_tc, V_tc, W_old_tc, W_tc, K, d)


    WSR_tc = torch.stack([torch.log(torch.sum(tc_det2(W_tc[k]))) for k in range(K)])
    WSR = torch.sum(WSR_tc)  # , dim = 0
    loss = -WSR
    loss.backward(retain_graph=True)

    # LAGD update

    V_tc_grad = torch.reshape(V_tc.grad.clone().detach(), [K, Nt * Nr * 2]).to(device)


    print("\n Epoch", '%.0f' % num_test, ": Start LAGD Updating:")
    start = timer()
    # while (err > epsilon) or t > Imax:
    for t in range(max_iterations):

        if t >= WMMSE_iterations:
            sum_rate_np.append(WSR_np)

        # #######################################  LAGD (Tensor)  #########################################

        # 初始化V   固定的 "fixed", 承接的 "inherited", 随机的 "random" ########################################
        if Initial_V_tc_mode == "fixed":
            V_tc = V_init_tc.clone().detach().requires_grad_(True).to(device)
        elif Initial_V_tc_mode == "inherited":
            V_tc = V_tc.clone().detach().requires_grad_(True).to(device)
        elif Initial_V_tc_mode == "random":
            V_tc_random = generate_random_V_tc()
            V_tc = V_tc_random.clone().detach().requires_grad_(True).to(device)

        if IF_random_label == True:

            # ################################ Compute WSR and gradient of V ##################################

            U_tc = tc_update_U(A_tc, U_tc, V_tc, H_tc, K, Nr)
            W_tc, W_old_tc = tc_update_Wk(U_tc, V_tc, W_old_tc, W_tc, K, d)
            # W_tc, W_old_tc = tc_update_Wk_Schulz(U_tc, V_tc, W_old_tc, W_tc, K, d)

            WSR_tc = torch.stack([torch.log(torch.sum(tc_det2(W_tc[k]))) for k in range(K)])
            WSR = torch.sum(WSR_tc)  # , dim = 0
            loss = -WSR
            loss.backward(retain_graph=True)

            V_tc_grad = torch.reshape(V_tc.grad.clone().detach(), [int((K * Nt * Nr * 2)/input_size_V), input_size_V]).to(device)


            ######################################### V update ###############################################

            if network == 'LSTM':
                state_h = None
                state_c = None
                V_tc_up, (state_h, state_c) = optimizee_V(V_tc_grad.type(torch.float32), state_h, state_c)
            elif network == 'FCN':
                V_tc_up = optimizee_V(V_tc_grad.type(torch.float32))

            V_tc_update = torch.reshape(V_tc_up, [K, Nt, Nr, 2])

            #Learning_strategy = "learn_one_step"  # V的更新方式  学习更新一步到位"learn_one_step", 学习辅助承接更新 "LAGD", 学习辅助与WMMSE交替 "LAGD_WMMSE", 学习辅助与GD交替 "LAGD_GD"

            # ############################################################ V generation through LAGD-Net #############################################################
            if Learning_strategy == "learn_one_step":
                V_tc_outer = V_init_tc + LAGD_lr * V_tc_update
                
            elif Learning_strategy == "LAGD":
                if LAGD_Strategy == "LAGD":
                    V_tc_outer = V_tc + LAGD_lr * V_tc_update

                elif LAGD_Strategy == "LAGD_momentum":
                    gradient_old = gradient_old.clone().detach()
                    V_tc_outer = V_tc + LAGD_lr * (beta * V_tc_update + (1 - beta) * gradient_old)
                    gradient_old = V_tc_update

                elif LAGD_Strategy == "LAGD_Adagard":
                    gradient = V_tc_update
                    Si = Si + gradient * gradient
                    Si = Si.clone().detach()
                    # gradient = gradient.clone().detach()
                    V_tc_outer = V_tc - LAGD_lr / torch.sqrt(Si + e * torch.ones_like(Si).to(device)) * gradient


                elif LAGD_Strategy == "LAGD_RMSProp":
                    gradient = V_tc_update
                    Si = Si + (1 - beta) * gradient * gradient
                    Si = Si.clone().detach()
                    V_tc_outer = V_tc - LAGD_lr / torch.sqrt(Si + e * torch.ones_like(Si).to(device)) * gradient

                elif LAGD_Strategy == "LAGD_Adam":
                    gradient = V_tc_update
                    Si = Si + (1 - beta) * gradient * gradient
                    Si = Si.clone().detach()
                    gradient_old = gradient_old.clone().detach()
                    V_tc_outer = V_tc - 1 / (torch.sqrt(Si) + e * torch.ones_like(Si).to(device)) * (
                                beta * gradient + (1 - beta) * gradient_old)
                    gradient_old = V_tc_update
            elif Learning_strategy == "LAGD_WMMSE":
                if (t) % (WMMSE_i + LAGD_i) < WMMSE_i:
                    V_tc.requires_grad = False
                    IF_WMMSE_solution = True
                    IF_LAGD_solution = False
                    V_tc_outer = tc_update_Vk(V_tc, W_tc, U_tc, H_tc, K, Nt, sigma)  # WMMSE solution
                else:
                    V_tc.requires_grad = True
                    IF_WMMSE_solution = False
                    IF_LAGD_solution = True
                    V_tc_outer = V_tc + LAGD_lr * V_tc_update  # LAGD solution
            elif Learning_strategy == "LAGD_GD":
                if (t) % (GD_j + LAGD_j) < GD_j:
                    V_tc.requires_grad = False
                    IF_GD_solution = True
                    IF_LAGD_solution = False
                    V_tc_outer = V_tc + LAGD_lr * V_tc.grad.clone().detach()  # traditional GD solution
                else:
                    V_tc.requires_grad = True
                    IF_GD_solution = False
                    IF_LAGD_solution = True
                    V_tc_outer = V_tc + LAGD_lr * V_tc_update  # LAGD solution


                # V_tc_outer = V_tc + LAGD_lr * V_tc_grad  # traditional GD solution

            # ############################################################ Random label losss #############################################################
            loss_random = 0
            filesource = os.listdir(os.path.abspath(random_label_load_path))
            random.shuffle(filesource)

            for r in range(random_inner_loop):
                V_label = torch.load(os.path.join(random_label_load_path, filesource[i]))
                loss_random += MSEloss(V_label, V_tc)

            loss_random = loss_random/random_inner_loop

            loss_random.backward(retain_graph=True)
            loss_random.detach()
            adam_optimizer_V.step()


        # ############################################################ V generation through LAGD-Net #############################################################

        # ################################ Compute WSR and gradient of V ##################################

        U_tc = tc_update_U(A_tc, U_tc, V_tc, H_tc, K, Nr)
        W_tc, W_old_tc = tc_update_Wk(U_tc, V_tc, W_old_tc, W_tc, K, d)
        # W_tc, W_old_tc = tc_update_Wk_Schulz(U_tc, V_tc, W_old_tc, W_tc, K, d)

        WSR_tc = torch.stack([torch.log(torch.sum(tc_det2(W_tc[k]))) for k in range(K)])
        WSR = torch.sum(WSR_tc)  # , dim = 0
        loss = -WSR
        loss.backward(retain_graph=True)

        V_tc_grad = torch.reshape(V_tc.grad.clone().detach(), [int((K * Nt * Nr * 2)/input_size_V), input_size_V]).to(device)


        ######################################### V update ###############################################

        if network == 'LSTM':
            state_h = None
            state_c = None
            V_tc_up, (state_h, state_c) = optimizee_V(V_tc_grad.type(torch.float32), state_h, state_c)
        elif network == 'FCN':
            V_tc_up = optimizee_V(V_tc_grad.type(torch.float32))

        V_tc_update = torch.reshape(V_tc_up, [K, Nt, Nr, 2])

        if Learning_strategy == "learn_one_step":
            V_tc_outer = V_init_tc + LAGD_lr * V_tc_update
            # V_tc_outer =  V_tc_update
        elif Learning_strategy == "LAGD":
            if LAGD_Strategy == "LAGD":
                V_tc_outer = V_tc + LAGD_lr * V_tc_update

            elif LAGD_Strategy == "LAGD_momentum":
                gradient_old = gradient_old.clone().detach()
                V_tc_outer = V_tc + LAGD_lr * (beta * V_tc_update + (1 - beta) * gradient_old)
                gradient_old = V_tc_update

            elif LAGD_Strategy == "LAGD_Adagard":
                gradient = V_tc_update
                Si = Si + gradient * gradient
                Si = Si.clone().detach()
                # gradient = gradient.clone().detach()
                V_tc_outer = V_tc - LAGD_lr / torch.sqrt(Si + e * torch.ones_like(Si).to(device)) * gradient


            elif LAGD_Strategy == "LAGD_RMSProp":
                gradient = V_tc_update
                Si = Si + (1 - beta) * gradient * gradient
                Si = Si.clone().detach()
                V_tc_outer = V_tc - LAGD_lr / torch.sqrt(Si + e * torch.ones_like(Si).to(device)) * gradient

            elif LAGD_Strategy == "LAGD_Adam":
                gradient = V_tc_update
                Si = Si + (1 - beta) * gradient * gradient
                Si = Si.clone().detach()
                gradient_old = gradient_old.clone().detach()
                V_tc_outer = V_tc - 1 / (torch.sqrt(Si) + e * torch.ones_like(Si).to(device)) * (
                            beta * gradient + (1 - beta) * gradient_old)
                gradient_old = V_tc_update

        # V_tc_outer = V_tc + LAGD_lr * V_tc_update



        elif Learning_strategy == "LAGD_WMMSE":
            if (t) % (WMMSE_i + LAGD_i) < WMMSE_i:
                V_tc.requires_grad = False
                IF_WMMSE_solution = True
                IF_LAGD_solution = False
                V_tc_outer = tc_update_Vk(V_tc, W_tc, U_tc, H_tc, K, Nt, sigma)  # WMMSE solution
            else:
                V_tc.requires_grad = True
                IF_WMMSE_solution = False
                IF_LAGD_solution = True
                V_tc_outer = V_tc + LAGD_lr * V_tc_update # LAGD solution
        elif Learning_strategy == "LAGD_GD":
            if (t) % (GD_j + LAGD_j) < GD_j:
                V_tc.requires_grad = False
                IF_GD_solution = True
                IF_LAGD_solution = False
                V_tc_outer = V_tc + LAGD_lr * V_tc.grad.clone().detach()  # traditional GD solution
            else:
                V_tc.requires_grad = True
                IF_GD_solution = False
                IF_LAGD_solution = True
                V_tc_outer = V_tc + LAGD_lr * V_tc_update # LAGD solution
            # V_tc_outer = V_tc + LAGD_lr * V_tc_grad  # traditional GD solution


        # ############################# Compute V Update loss for LAGD-Net##############################

        U_tc = tc_update_U(A_tc, U_tc, V_tc_outer, H_tc, K, Nr)
        W_tc, W_old_tc = tc_update_Wk(U_tc, V_tc_outer, W_old_tc, W_tc, K, d)
        # V_tc = tc_update_Vk(V_tc, W_tc, U_tc, H_tc, K, Nt, sigma)
        WSR_tc = torch.stack([torch.log(torch.sum(tc_det2(W_tc[k]))) for k in range(K)])
        WSR = torch.sum(WSR_tc)  #, dim = 0
        tc_trace1 = torch.sum(torch.stack([tc_trace(tc_mul(V_tc_outer[i], tc_CT(V_tc_outer[i]))) for i in range(K)]),
                              dim=0)
        power1 = tc_trace1[0]
        loss = -WSR #  - rho * (PT - power1)

        # ############################### Update parameters of LAGD-Net##################################

        # #### update V by each step

        # adam_optimizer_V.zero_grad()
        # loss.backward(retain_graph=True)
        # adam_optimizer_V.step()

        # #### update V by N(meta_loop) step
        if (t) % meta_loop == 0:
            Accumulated_loss = loss
        else:
            Accumulated_loss = Accumulated_loss + loss

        if (t + 1) % meta_loop == 0:
            # V_tc.requires_grad = False
            if IF_LAGD_solution == True:
                adam_optimizer_V.zero_grad()
                Average_loss = Accumulated_loss / meta_loop
                Average_loss.backward(retain_graph=True)
                adam_optimizer_V.step()
                Accumulated_loss = 0



        # ################################ Scale V tensor complexity ################################

        if (t + 1) % Scale_V_steps == 0:
            # tc_trace1 = torch.sum(torch.stack([tc_trace(tc_mul(V_tc_outer[i], tc_CT(V_tc_outer[i]))) for i in range(K)]), dim=0)
            # tc_trace_np = np.sum([np.trace(np.dot(V[i], V[i].conjugate().T)) for i in range(K)])
            # power1 = tc_trace1[0]
            # power_np = tc_trace_np.real
            alpha = torch.sqrt(PT / power1)
            # alpha_np = np.sqrt(PT / tc_trace_np)
            # alpha = torch.sqrt(PT / torch.sum(torch.stack([torch.trace(tc_mul(V_tc[i], tc_CT(V_tc[i]))) for i in range(K)])))

            # for k in range(K):  # Scale V_k
                # V[k] = alpha_np * V[k]
            V_tc = alpha * V_tc_outer.clone().detach()

            # tc_trace1 = torch.sum(torch.stack([tc_trace(tc_mul(V_tc[i], tc_CT(V_tc[i]))) for i in range(K)]), dim=0)
            # power1 = tc_trace1[0]

            U_tc = tc_update_U(A_tc, U_tc, V_tc, H_tc, K, Nr)
            W_tc, W_old_tc = tc_update_Wk(U_tc, V_tc, W_old_tc, W_tc, K, d)
            # V_tc = tc_update_Vk(V_tc, W_tc, U_tc, H_tc, K, Nt, sigma)

            WSR_tc = torch.stack([torch.log(torch.sum(tc_det2(W_tc[k]))) for k in range(K)])
            WSR = torch.sum(WSR_tc)  #, dim = 0

            WSR_new = tc_WSR_compute(A_tc, V_tc, H_tc, K, Nr)


        sum_rate.append(WSR.detach().cpu().numpy())

        # t += 1
        iteration.append(t + 1)
        # print(t)
        ratio = (t + 1) / max_iterations
        percent = str(t + 1)+'/'+str(max_iterations)

        time = timer() - start
        # progress(ratio, percent)

        if IF_LAGD_solution == True:
            solution_type = '<LAGD> '
        elif IF_WMMSE_solution == True:
            solution_type = '<WMMSE>'
        elif IF_GD_solution == True:
            solution_type = '<GD>   '

        if (t + 1)%print_step == 0:
            print('->  step :', t + 1, 'Solution:', solution_type , 'WSR(LAGD) =', '%.2f' % WSR, 'WSR(LAGD)_new =', '%.2f' % WSR_new, 'WSR(WMMSE) =', '%.2f' % WSR_np, 'Trace', '%.2f' % power1.item(), 'time=''%.2f' % time) # , 'Trace', '%.2f' % power1.item() 'Trace_WMMSE', '%.2f' % power_np, , end=''

        rate_iterations[t][num_test] = WSR

    rate_itera_WMMSE[num_test] = WSR_np

    rate.append(sum_rate[t - 1])
    print('\nLAGD WSR:', '%.4f' % sum_rate[t - 1])
    plt.plot(iteration, sum_rate, label='LAGD') #,marker = "o",markersize=5
    plt.plot(iteration, sum_rate_np, label='WMMSE') #,marker = "x",markersize=5
    plt.xlabel("iteration")
    plt.ylabel("rate")
    plt.legend()
    plt.grid()  # 添加网格
    plt.show()

    rate_np.append(sum_rate_np[t - 1])
    print('WMMSE WSR:', '%.4f' % sum_rate_np[t - 1])
    # plt.plot(iteration, sum_rate_np)
    # plt.xlabel("iteration")
    # plt.ylabel("rate_np")
    # plt.show()


    sheet.write(num_test + 1, col1, '%.2f' % sum_rate[t - 1], black_style)
    wb.save('LAGD_SNR_'+ str(SNR)+'dB'+'_K_'+str(K)+'_Nt_'+str(Nt)+'_Nr_'+str(Nr)+'.xls')  # 最后一定要保存，否则无效
    sheet.write(num_test + 1, col2, '%.2f' % sum_rate_np[t - 1], black_style)
    wb.save('LAGD_SNR_'+ str(SNR)+'dB'+'_K_'+str(K)+'_Nt_'+str(Nt)+'_Nr_'+str(Nr)+'.xls')  # 最后一定要保存，否则无效


rate_itera_LAGD = np.mean(rate_iterations, axis=1)  # axis=1，对每一个子数组，计算它的平均值
rate_itera_WMMSE = np.mean(rate_itera_WMMSE, axis=0) * np.ones_like(rate_itera_LAGD)

print("\nAverage LAGD WSR:", '%.4f' % np.array(rate).mean())
print("Average WMMSE WSR:", '%.4f' % np.array(rate_np).mean())

plt.plot(iteration, rate_itera_LAGD, label='LAGD', linewidth=1, marker = "o", markersize=2)  # ,marker = "o",markersize=5
plt.plot(iteration, rate_itera_WMMSE, label='WMMSE', linestyle='--', linewidth=3, color='k')  # ,marker = "x",markersize=5
plt.xlabel("iterations", fontproperties='Times New Roman', size=20, weight='bold')
plt.ylabel("WSR", fontproperties='Times New Roman', size=20, weight='bold')
plt.yticks(fontproperties='Times New Roman', size=18)#  ,weight='bold' 设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=18)
plt.tight_layout()
font = {'family': 'Times New Roman','size': 16}
plt.legend(prop=font)
plt.grid()  # 添加网格
plt.show()