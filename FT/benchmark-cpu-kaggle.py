#!/usr/bin/env python
# coding: utf-8

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

#Author: Raquel Ricoy

#Benchmark to study Kaggle's GPUs, CPUs and TPUs potential.
#It's going to use Pytorch and to stablish a script to calculate its performance and GFLOPS.

#Install pytorch
#!conda install -y pytorch torchvision -c pytorch

#Install openpyxl to import an excel with the operations in pandas
#get_ipython().system('pip install openpyxl')

import torch
import platform

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
#print(os.listdir("../input"))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#Importing Libraries needed for use torch
import timeit
import torch.utils.benchmark as benchmark


#Information about system
print('Platform processor:', platform.processor())
print('Platform architecture:', platform.architecture())

#Number of threads
num_cores = os.cpu_count()
print('Number of cores:',num_cores)
num_threads = num_cores


#Functions obtained from Torch Webpages por PyTorch Benchmarks
def batched_dot_mul_sum(a, b):
    '''Computes batched dot by multiplying and summing'''
    return a.mul(b).sum(-1)


def batched_dot_bmm(a, b):
    '''Computes batched dot by reducing to bmm'''
    a = a.reshape(-1, 1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1], 1)
    return torch.bmm(a, b).flatten(-3)

# Method that do the benchmark and compare results with dot mul sum implementations and vectorSum
def benchMark(sizes,nThreads):
    results = []
    if(len(sizes) == 0):
        print("Parameter 'sizes' has to a have minumun of 1 parameters")
        return
    if(len(nThreads)==0):
        print("Parameter 'nThreads' has to a have minumun of 1 parameters")
    
    for n in sizes:
        # label and sub_label are the rows
        # description is the column
        label = 'Batched dot'
        sub_label = f'[{n}, {n}]'
        x = torch.ones((n, n))
        for num_threads in nThreads:
            results.append(benchmark.Timer(
                stmt='batched_dot_mul_sum(x, x)',
                setup='from __main__ import batched_dot_mul_sum',
                globals={'x': x},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='mul/sum',
            ).blocked_autorange())
            results.append(benchmark.Timer(
                stmt='batched_dot_bmm(x, x)',
                setup='from __main__ import batched_dot_bmm',
                globals={'x': x},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='bmm',
            ).blocked_autorange())   
    compare = benchmark.Compare(results)
    compare.print()
    return compare




#The limit dimension of the sizes is with matrix of 16384x16384. It is running out of memory with that sizes
sizes = [512,1024,2048,4096,8192,16384,32768]
thread = torch.get_num_threads() #In FT we can just send one by one, we cannot set with pytorch numThreads

#The benchmark execute 5 times to gather data and afterwards 
#for i in range(0,5):
#    print("Benchmark execution: ",i+1, "\n")
#    benchMark(sizes,threads)

def ownBenchmark(sizes,writerCSV,operation,thread):        
    for i in range(0,5):
        print("\nBenchmark execution for ",operation,": ",i+1, "\n")
        print("\nNumber of threads: ",thread, "\n")
        for n in sizes:
          timeInit = time.time()

          xCPU = torch.ones(n, n)

          if(operation == "mul_sum"):
            batched_dot_mul_sum(xCPU,xCPU)
          else:
            batched_dot_bmm(xCPU,xCPU)

          timeFinish = time.time()
          print(f"size matrix [{n}] -> {(timeFinish - timeInit):0.8f} s")
          writer.writerow([operation, n, i+1,thread,(timeFinish - timeInit)])

#Now my own benchmark. With this i going to measure Speed ups and efficiencies. The pytorch benchmark give us too good results to be true...

import time #-> time.time() returns the time in seconds
import csv #We are going to generate an csv with the results to work with pandas

sizes = [512,1024,2048,4096,8192,16384,32768] # maximun size withou running out memory -> 65536
thread = torch.get_num_threads() #In FT we can just send one by one, we cannot set with pytorch numThreads

with open('results_cpu.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["operation", "sizeMatrix", "numberCase","nThreads","timeElpased"])
    ownBenchmark(sizes,writer,"mul_sum",thread)
    ownBenchmark(sizes,writer,"bmm",thread)


#Generate the excel and giving a little of format
#TODO include the calculate of FLOPS in excel/dataFrame
import pandas as pd

df = pd.read_csv("results_cpu.csv")
df.info()

df.sort_values(by=["operation","numberCase","nThreads"])

df[df.nThreads==thread].to_excel("results_cpu_excel_"+str(thread)+"Thread.xlsx")

