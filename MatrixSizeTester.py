import numpy as np
import os 
import math

#Theoretically a matrix should be unlimited in size and the only limit is the 
# local memory of the interpreter/processor

#To prove this, this program determines the matrix size limit that would be 
#constrained by the processor running this code and tests this by attempting to construct a matrix 
# of that size, below that size, and of infinite size. If there is no limit in theory the 
# error from going over this limit should be related to a system failure, rather than a syntax error

def getRAM(): 
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception: 
        try:
            with open("/proc/meminfo", "r") as f: 
                for line in f: 
                    if line.startsiwth("MemAvailable"):
                        kb = int(line.split()[1])
                        return kb * 1024
        except Exception:
            return 0

def estMatrixSize(available_bytes: int, dtype, safe_fract):
    #Returns n, element size, allowed bytes, where n is the largest n*n matrix fhat fits with a certain size
    #Run basic principles tests with 0 or single digits (0 and 1 take up the same memory?)
    if available_bytes <= 0:
        return 0,0,0
    elementSize = np.dtype(dtype).itemsize
    allowed = int(available_bytes * safe_fract) 
    maxElements = allowed//elementSize
    n = math.isqrt(maxElements)

    return n, elementSize, allowed

def normalizeDType(d):
    global goodInput
    if isinstance(d, (np.dtype,)): 
        return d
    if isinstance(d, str): 
        try: 
            goodInput = True
            return np.dtype(d)
        except Exception:
            try:
                return np.dtype(eval(d, {"np":np, "numpy":np}))
            except Exception as e: 
                raise ValueError(f"Can't interpret dtype from strin: {d}") from e
    try:
        return np.type(d)
    except Exception as e: 
        raise ValueError(f"Unsupported dtype: {d}") from e


def testMatrixSize(dtype, SF):
    avail = getRAM()
    n, elementSize, allowed = estMatrixSize(avail, dtype, SF)
    print("With datatype:", dtype, ". The available RAM of this device (allocated to running this program): ", avail/1024, "kB. The safety fraction", SF, ":")
    print("the max square matrix (n*n) size is n = ", n)

goodInput = False
while(not goodInput):
    userInput = input("What is the datatype and safety fraction (0<SF<1) you are using in this matrix? Answer separated with a comma (ex: np.float64,0.5): ")
    data = [p.strip() for p in userInput.split(",",1)]
    if len(data) != 2:
            raise ValueError("Enter dtype and safety fraction")

    dT = normalizeDType(data[0])
    try:
        SF = float(data[1])
        goodInput = True
    except ValueError: 
        raise ValueError("Safety fraction must be float")

testMatrixSize(dT, SF)




#Safe fraction is the most important term in answering this question
# the closer it is to 1, the larger the matrix can be, but this risks
# errors in memory allocation that can vary depending on
# the system running this code. The elements in the array are also a significant 
# factor, but you can use this function to test your system with the element 
# size and safety fraction that you need to figure out the maximum size of
# a matrix in a given context. 

