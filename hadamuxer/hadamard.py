import numpy as np

# This function taken from scripts given to me by DvS and GB (identical in both)
def isprime(n): # Check that a given n is prime                           
    for x in range(2, int(n**0.5)+1):
        if n % x == 0:
            return False
    return True

# This function adapted from a script given to me by DvS
def isvalid_Hadamard_sequence_lengths(n): # Check that a given n is a valid length for a Hadamard S matrix
    for m in range(0,n):
        if isprime(n) and n == 4*m +3:
            return True
    return False

# This function adapted from scripts given to me by DvS and GB
def make_S_mat(n): # create a Hadamard S matrix of a given size
    # check that the input size is indeed valid
    if isvalid_Hadamard_sequence_lengths(n) == True:
        
        # calculate first row of S matrix using quadratic residue method
        m = range(0,n)                   # define variable to hold array size
        Srow = [0 for i in m]            # define blank array that becomes first line of S matrix
    
        for i in range(0,(n-1)//2):      
            Srow[(i+1)*(i+1) % n] = 1    # DvS says "need only go to half range of n as indices are symmetric"
        Srow[0] = 1
    
        # calculate full S matrix 
        S = np.zeros((n,n), dtype = float)  # initialize a blank array
        for i in range(n):
            for j in range(n):
                S[i,j] = Srow[j]
            Srow = np.roll(Srow, -1)
            pass
        return S
    
    else:
        print('This is not a valid Hadamard matrix size')
        pass

