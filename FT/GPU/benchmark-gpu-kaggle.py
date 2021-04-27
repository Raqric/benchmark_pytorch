#!/usr/bin/env python
# coding: utf-8

# 
# # Benchmarking GPUs in Kaggle
# 
# In this Kaggle notebook there is an adaptation from my Benchmark CPU to GPU using pytorch benchmark. The main method (benchmark) it change the input parameters, now it just needs the sizes to process. We are going to check the results and analyse them. Besides, there's my own method for timing, and it'is going to be used to analyse the timeit library from pytorch. 

# ## My CPU Benchmark adapted to GPU
# 
# Pytorch has hard coded a block size of 256 threads. So there's only one execution per matrix size.
# 
# One important advertisment! GPU accelerator has to be activated to use this notebook, if not, the notebook is not going to compile.

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

#Author: Raquel Ricoy

#Benchmark to study Kaggle's GPUs, CPUs and TPUs potential.
#It's going to use Pytorch and to stablish a script to calculate its performance and GFLOPS.

#Install pytorch
#!conda install -y pytorch torchvision -c pytorch

import torch

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
#print(os.listdir("../input"))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#Importing Libraries needed for use torch
import timeit
import torch.utils.benchmark as benchmark

#Functions obtained from Torch Webpages by PyTorch Benchmarks
def batched_dot_mul_sum(a, b):
    '''Computes batched dot by multiplying and summing'''
    return a.mul(b).sum(-1)

def batched_dot_bmm(a, b):
    '''Computes batched dot by reducing to bmm'''
    a = a.reshape(-1, 1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1], 1)
    return torch.bmm(a, b).flatten(-3)


# Method that do the benchmark and compare results with dot mul sum implementations and vectorSum
#Anotation: We cannot change the block threads in pytorch for GPU, it's always 256 threads per block! So the 1 threads that put in benchmark is erroneus and put it by default
def benchMark(sizes):
    results = []
    if(len(sizes) == 0):
        print("Parameter 'sizes' has to a have minumun of 1 parameters")
        return
    
    for n in sizes:
        # label and sub_label are the rows
        # description is the column
        label = 'Batched dot'
        sub_label = f'[{n}, {n}]'
        xCPU = torch.ones(n, n)
        xCUDA = xCPU.to(device="cuda:0")
        results.append(benchmark.Timer(
                stmt='batched_dot_mul_sum(x, x)',
                setup='from __main__ import batched_dot_mul_sum',
                globals={'x': xCUDA},
                label=label,
                sub_label=sub_label,
                description='mul/sum',
            ).blocked_autorange(min_run_time=1))
        results.append(benchmark.Timer(
                stmt='batched_dot_bmm(x, x)',
                setup='from __main__ import batched_dot_bmm',
                globals={'x': xCUDA},
                label=label,
                sub_label=sub_label,
                description='bmm',
            ).blocked_autorange(min_run_time=1))
    compare = benchmark.Compare(results)
    compare.print()
    return compare

#Evaluating with which GPU we are going to use
if torch.cuda.is_available(): 
    print("GPU is available")
    print("GPU device where we are gonna execute tests: ",torch.cuda.get_device_name())
else:
    print("GPU is NOT available")

#The limit dimension of the sizes is [65536,65536]. It is running out of memory with that sizes
sizes = [512,1024,2048,4096,8192,16384,32768]
compares = []

#The benchmark execute 5 times to gather data and afterwards 
for i in range(0,5):
    print("Benchmark execution: ",i+1, "\n")
    compares.insert(i,benchMark(sizes))


def ownBenchmark(sizes,writerCSV,operation):
    cuda0 = torch.device("cuda:0")
    for i in range(0,5):
        print("\nBenchmark execution for ",operation,": ",i+1, "\n")
        for n in sizes:
            timeInit = time.time()
            xCPU = torch.ones(n, n)
            xCUDA = xCPU.to(device=cuda0)
            if(operation == "mul_sum"):
                batched_dot_mul_sum(xCUDA,xCUDA)
            else:
                batched_dot_bmm(xCUDA,xCUDA)
            torch.cuda.synchronize()
            timeFinish = time.time()
            print(f"size matrix [{n}] -> {(timeFinish - timeInit):0.8f} s")
            writer.writerow([operation, n, i+1,(timeFinish - timeInit)])

#Now my own benchmark. With this i going to measure Speed ups and efficiencies. The pytorch benchmark give us too good results to be true...
# We are going to use the library time from python and do the syncronizations to the gpu device
import time #-> time.time() returns the time in seconds
import csv #We are going to generate an csv with the results to work with pandas

sizes = [512,1024,2048,4096,8192,16384,32768] # maximun size withou running out memory -> 65536

with open('results_gpu.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["operation", "sizeMatrix", "numberCase","timeElpased"])
    ownBenchmark(sizes,writer,"mul_sum")
    ownBenchmark(sizes,writer,"bmm")

#Generate the excel and giving a little of format
#TODO include the calculate of FLOPS in excel/dataFrame
import pandas as pd

df = pd.read_csv("results_gpu.csv")
df.info()

df_sorted = df.sort_values(by=["operation","numberCase"])

df_sorted.to_excel("results_gpu_excel.xlsx")

