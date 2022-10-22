import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import pickle

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print("Device_use:", device)

def progress(arg, percent):
	if arg > 1:
		arg = 1
	# 设计进度条和百分比

	image = int(50 * arg) * '#'
	# percent = str(int(arg * 100)) + '%'
	# 打印进度条
	print('\033[1;31m LAGD Process: \r[%-50s] %s \033[0m' % ( image, percent), end='')
	# 下载完成判断并提示
	if arg == 1:
		print(' ')



def tc_det2(A):
    if A.ndim == 3 and A.shape[0] == 2 and A.shape[1] == 2:
        A_real = A[:, :, 0]
        A_imag = A[:, :, 1]

        A_det_real = A_real[0][0] * A_real[1][1] - A_real[0][1] * A_real[1][0] + A_imag[0][1] * A_imag[1][0]
        A_det_imag = 0 - A_real[0][1] * A_imag[1][0] - A_real[1][0] * A_imag[0][1]

        A_det_tc = torch.cat((A_det_real.unsqueeze(-1), A_det_imag.unsqueeze(-1)), 0)

    return A_det_tc

def compare_tc_np(tc, np):
    np_tc = torch.from_numpy(np)
    np_tc_real = np_tc.real.unsqueeze(-1)
    np_tc_imag = np_tc.imag.unsqueeze(-1)
    np_tc = torch.cat((np_tc_real, np_tc_imag), -1)

    return torch.sum(tc - np_tc)

def save_variable(v,filename):
  f=open(filename,'wb')
  pickle.dump(v,f)
  f.close()
  return filename


def load_variavle(filename):
  f=open(filename,'rb')
  r=pickle.load(f)
  f.close()
  return r


def tc_det(A):

    if A.ndim == 3:

        A_real = A[:, :, 0]
        A_imag = A[:, :, 1]

        A_c = A_real + 1j * A_imag

        A_det = torch.linalg.det(A_c)

        A_det_real = A_det.real.unsqueeze(-1)
        A_det_imag = A_det.imag.unsqueeze(-1)
        A_det_tc = torch.cat((A_det_real, A_det_imag), 0)

        A_det_tc = A_det_tc.type(torch.float64)


    return A_det_tc

def tc_log(A):

    if A.ndim == 1:
        A_real = A[0]
        A_imag = A[1]

        A_c = A_real + 1j * A_imag

        A_log = torch.log(A_c)

        A_log_real = A_log.real.unsqueeze(-1)
        A_log_imag = A_log.imag.unsqueeze(-1)
        A_log_tc = torch.cat((A_log_real, A_log_imag), 0)

        A_log_tc = A_log_tc.type(torch.float64)


    if A.ndim == 3:
        A_real = A[:, :, 0]
        A_imag = A[:, :, 1]

        A_c = A_real + 1j * A_imag

        A_log = torch.log(A_c)

        A_log_real = A_log.real.unsqueeze(-1)
        A_log_imag = A_log.imag.unsqueeze(-1)
        A_log_tc = torch.cat((A_log_real, A_log_imag), 2)

        A_log_tc = A_log_tc.type(torch.float64)


    return A_log_tc


def tc_inv(A):

    if A.ndim == 3:

        A_real = A[:, :, 0]
        A_imag = A[:, :, 1]

        A_c = A_real + 1j * A_imag
        A_inv = torch.linalg.inv(A_c)

        A_inv_real = A_inv.real.unsqueeze(-1)
        A_inv_imag = A_inv.imag.unsqueeze(-1)
        A_inv_tc = torch.cat((A_inv_real, A_inv_imag), 2)

        A_inv_tc = A_inv_tc.type(torch.float64)

    return A_inv_tc

def tc_pinv(A):

    if A.ndim == 3:

        A_real = A[:, :, 0]
        A_imag = A[:, :, 1]

        A_c = A_real + 1j * A_imag
        A_pinv = torch.linalg.pinv(A_c)

        A_pinv_real = A_pinv.real.unsqueeze(-1)
        A_pinv_imag = A_pinv.imag.unsqueeze(-1)
        A_pinv_tc = torch.cat((A_pinv_real, A_pinv_imag), 2)

        A_pinv_tc = A_pinv_tc.type(torch.float64)

    return A_pinv_tc


def tc_mul(A, B):

    if A.dtype != B.dtype:

        A = A.type(torch.float64)
        B = B.type(torch.float64)

    if A.ndim == 3:

        A_real = A[:, :, 0]
        A_imag = A[:, :, 1]
        B_real = B[:, :, 0]
        B_imag = B[:, :, 1]

        C_real = torch.mm(A_real, B_real) - torch.mm(A_imag, B_imag)
        C_imag = torch.mm(A_real, B_imag) + torch.mm(A_imag, B_real)

        C = torch.cat((C_real.unsqueeze(-1), C_imag.unsqueeze(-1)), 2)

    if A.ndim == 2:

        C_real = torch.mm(A.real, B.real) - torch.mm(A.imag, B.imag)
        C_imag = torch.mm(A.real, B.imag) + torch.mm(A.imag, B.real)

        C = C_real + 1j * C_imag

    if A.ndim == 1:
        A_real = A[0]
        A_imag = A[1]
        B_real = B[0]
        B_imag = B[1]

        C_real = A_real * B_real - A_imag * B_imag
        C_imag = A_real * B_imag + A_imag * B_real

        C = torch.cat((C_real.unsqueeze(-1), C_imag.unsqueeze(-1)), 0)

    return C

def tc_T(A):
    if A.ndim == 3:

        # A_T = torch.zeros(A.shape[1], A.shape[0], A.shape[2])
        A_real = A[:, :, 0]
        A_imag = A[:, :, 1]
        A_T = torch.cat(((A_real.T).unsqueeze(-1), (A_imag.T).unsqueeze(-1)), 2)



    if A.ndim == 2:
        # A_T = torch.zeros(A.shape[1], A.shape[0], dtype=torch.complex128)
        A_T = A.T

    return A_T

def tc_CT(A):
    if A.ndim == 3:
        # A_CT = torch.zeros(A.shape[0], A.shape[2], A.shape[1], dtype=torch.complex128)
        A_T = tc_T(A)

        Imag = torch.zeros(A_T.shape[0], A_T.shape[1], 2).to(device)

        Imag[:, :, 1] = A_T[:, :, 1]

        A_CT = A_T - 2 * Imag

    if A.ndim == 2:
        # A_CT = torch.zeros(A.shape[1], A.shape[0], dtype=torch.complex128)
        A_T = tc_T(A)
        A_CT = A_T.real - 1j * A_T.imag

    if A.ndim == 1:
        A_real = A[0]
        A_imag = A[1]

        A_CT_real = A_real
        A_CT_imag = 0 - A_imag

        A_CT = torch.cat((A_CT_real.unsqueeze(-1), A_CT_imag.unsqueeze(-1)), 0)


    return A_CT

def tc_trace(A):

    if A.ndim == 3:

        A_real = A[:, :, 0]
        A_imag = A[:, :, 1]

        trace_real = torch.trace(A_real)
        trace_imag = torch.trace(A_imag)

        trace = torch.cat((trace_real.unsqueeze(-1), trace_imag.unsqueeze(-1)), 0)

    return trace



