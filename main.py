"""
Read-me:
Call functions in this order:
    problem = gen_matrix(v,c)
    constrain(problem, string)
    obj(problem, string)
    maxz(problem)
gen_matrix() produces a matrix to be given constraints and an objective function to maximize or minimize.
    It takes var (variable number) and cons (constraint number) as parameters.
    gen_matrix(2,3) will create a 4x7 matrix by design.
constrain() constrains the problem. It takes the problem as the first argument and a string as the second. The string should be
    entered in the form of 1,2,G,10 meaning 1(x1) + 2(x2) >= 10.
    Use 'L' for <= instead of 'G'
Use obj() only after entering all constraints, in the form of 1,2,0 meaning 1(x1) +2(x2) +0
    The final term is always reserved for a constant and 0 cannot be omitted.
Use maxz() to solve a maximization LP problem. Use minz() to solve a minimization problem.
Disclosure -- pivot() function, subcomponent of maxz() and minz(), has a couple bugs. So far, these have only occurred when
    minz() has been called.
"""

import numpy as np

# generates an empty matrix with adequate size for variables and constraints.
def gen_matrix(var,cons):
    tab = np.zeros((cons+1, var+cons+2))
    return tab

# checks the furthest right column for negative values ABOVE the last row. If negative values exist, another pivot is required.
def next_round_r(table):
    m = min(table[:-1,-1])
    if m>= 0:
        return False
    else:
        return True

# checks that the bottom row, excluding the final column, for negative values. If negative values exist, another pivot is required.
def next_round(table):
    lr = len(table[:,0])
    m = min(table[lr-1,:-1])
    if m>=0:
        return False
    else:
        return True

# Similar to next_round_r function, but returns row index of negative element in furthest right column
def find_neg_r(table):
    # lc = number of columns, lr = number of rows
    lc = len(table[0,:])
    # search every row (excluding last row) in final column for min value
    m = min(table[:-1,lc-1])
    if m<=0:
        # n = row index of m location
        n = np.where(table[:-1,lc-1] == m)[0][0]
    else:
        n = None
    return n

#returns column index of negative element in bottom row
def find_neg(table):
    lr = len(table[:,0])
    m = min(table[lr-1,:-1])
    if m<=0:
        # n = row index for m
        n = np.where(table[lr-1,:-1] == m)[0][0]
    else:
        n = None
    return n

# locates pivot element in tableu to remove the negative element from the furthest right column.
def loc_piv_r(table):
        total = []
        # r = row index of negative entry
        r = find_neg_r(table)
        # finds all elements in row, r, excluding final column
        row = table[r,:-1]
        # finds minimum value in row (excluding the last column)
        m = min(row)
        # c = column index for minimum entry in row
        c = np.where(row == m)[0][0]
        # all elements in column
        col = table[:-1,c]
        # need to go through this column to find smallest positive ratio
        for i, b in zip(col,table[:-1,-1]):
            # i cannot equal 0 and b/i must be positive.
            if i**2>0 and b/i>0:
                total.append(b/i)
            else:
                # placeholder for elements that did not satisfy the above requirements. Otherwise, our index number would be faulty.
                total.append(0)
        element = max(total)
        for t in total:
            if t > 0 and t < element:
                element = t
            else:
                continue

        index = total.index(element)
        return [index,c]
# similar process, returns a specific array element to be pivoted on.
def loc_piv(table):
    if next_round(table):
        total = []
        n = find_neg(table)
        for i,b in zip(table[:-1,n],table[:-1,-1]):
            if i**2>0 and b/i>0:
                total.append(b/i)
            else:
                # placeholder for elements that did not satisfy the above requirements. Otherwise, our index number would be faulty.
                total.append(0)
        element = max(total)
        for t in total:
            if t > 0 and t < element:
                element = t
            else:
                continue

        index = total.index(element)
        return [index,n]

# Takes string input and returns a list of numbers to be arranged in tableu
def convert(eq):
    eq = eq.split(',')
    if 'G' in eq:
        g = eq.index('G')
        del eq[g]
        eq = [float(i)*-1 for i in eq]
        return eq
    if 'L' in eq:
        l = eq.index('L')
        del eq[l]
        eq = [float(i) for i in eq]
        return eq

# The final row of the tablue in a minimum problem is the opposite of a maximization problem so elements are multiplied by (-1)
def convert_min(table):
    table[-1,:-2] = [-1*i for i in table[-1,:-2]]
    table[-1,-1] = -1*table[-1,-1]
    return table

# generates x1,x2,...xn for the varying number of variables.
def gen_var(table):
    lc = len(table[0,:])
    lr = len(table[:,0])
    var = lc - lr -1
    v = []
    for i in range(var):
        v.append('x'+str(i+1))
    return v

# pivots the tableau such that negative elements are purged from the last row and last column
def pivot(row,col,table):
    # number of rows
    lr = len(table[:,0])
    # number of columns
    lc = len(table[0,:])
    t = np.zeros((lr,lc))
    pr = table[row,:]
    if table[row,col]**2>0: #new
        e = 1/table[row,col]
        r = pr*e
        for i in range(len(table[:,col])):
            k = table[i,:]
            c = table[i,col]
            if list(k) == list(pr):
                continue
            else:
                t[i,:] = list(k-r*c)
        t[row,:] = list(r)
        return t
    else:
        print('Cannot pivot on this element.')

# checks if there is room in the matrix to add another constraint
def add_cons(table):
    lr = len(table[:,0])
    # want to know IF at least 2 rows of all zero elements exist
    empty = []
    # iterate through each row
    for i in range(lr):
        total = 0
        for j in table[i,:]:
            # use squared value so (-x) and (+x) don't cancel each other out
            total += j**2
        if total == 0:
            # append zero to list ONLY if all elements in a row are zero
            empty.append(total)
    # There are at least 2 rows with all zero elements if the following is true
    if len(empty)>1:
        return True
    else:
        return False

# adds a constraint to the matrix
def constrain(table,eq):
    if add_cons(table) == True:
        lc = len(table[0,:])
        lr = len(table[:,0])
        var = lc - lr -1
        # set up counter to iterate through the total length of rows
        j = 0
        while j < lr:
            # Iterate by row
            row_check = table[j,:]
            # total will be sum of entries in row
            total = 0
            # Find first row with all 0 entries
            for i in row_check:
                total += float(i**2)
            if total == 0:
                # We've found the first row with all zero entries
                row = row_check
                break
            j +=1

        eq = convert(eq)
        i = 0
        # iterate through all terms in the constraint function, excluding the last
        while i<len(eq)-1:
            # assign row values according to the equation
            row[i] = eq[i]
            i +=1
        #row[len(eq)-1] = 1
        row[-1] = eq[-1]

        # add slack variable according to location in tableau.
        row[var+j] = 1
    else:
        print('Cannot add another constraint.')

# checks to determine if an objective function can be added to the matrix
def add_obj(table):
    lr = len(table[:,0])
    # want to know IF exactly one row of all zero elements exist
    empty = []
    # iterate through each row
    for i in range(lr):
        total = 0
        for j in table[i,:]:
            # use squared value so (-x) and (+x) don't cancel each other out
            total += j**2
        if total == 0:
            # append zero to list ONLY if all elements in a row are zero
            empty.append(total)
    # There is exactly one row with all zero elements if the following is true
    if len(empty)==1:
        return True
    else:
        return False

# adds the onjective functio nto the matrix.
def obj(table,eq):
    if add_obj(table)==True:
        eq = [float(i) for i in eq.split(',')]
        lr = len(table[:,0])
        row = table[lr-1,:]
        i = 0
    # iterate through all terms in the constraint function, excluding the last
        while i<len(eq)-1:
            # assign row values according to the equation
            row[i] = eq[i]*-1
            i +=1
        row[-2] = 1
        row[-1] = eq[-1]
    else:
        print('You must finish adding constraints before the objective function can be added.')

# solves maximization problem for optimal solution, returns dictionary w/ keys x1,x2...xn and max.
def maxz(table, output='summary'):
    while next_round_r(table)==True:
        table = pivot(loc_piv_r(table)[0],loc_piv_r(table)[1],table)
    while next_round(table)==True:
        table = pivot(loc_piv(table)[0],loc_piv(table)[1],table)

    lc = len(table[0,:])
    lr = len(table[:,0])
    var = lc - lr -1
    i = 0
    val = {}
    for i in range(var):
        col = table[:,i]
        s = sum(col)
        m = max(col)
        if float(s) == float(m):
            loc = np.where(col == m)[0][0]
            val[gen_var(table)[i]] = table[loc,-1]
        else:
            val[gen_var(table)[i]] = 0
    val['max'] = table[-1,-1]
    for k,v in val.items():
        val[k] = round(v,6)
    if output == 'table':
        return table
    else:
        return val

# solves minimization problems for optimal solution, returns dictionary w/ keys x1,x2...xn and min.
def minz(table, output='summary'):
    table = convert_min(table)

    while next_round_r(table)==True:
        table = pivot(loc_piv_r(table)[0],loc_piv_r(table)[1],table)
    while next_round(table)==True:
        table = pivot(loc_piv(table)[0],loc_piv(table)[1],table)

    lc = len(table[0,:])
    lr = len(table[:,0])
    var = lc - lr -1
    i = 0
    val = {}
    for i in range(var):
        col = table[:,i]
        s = sum(col)
        m = max(col)
        if float(s) == float(m):
            loc = np.where(col == m)[0][0]
            val[gen_var(table)[i]] = table[loc,-1]
        else:
            val[gen_var(table)[i]] = 0
    val['min'] = table[-1,-1]*-1
    for k,v in val.items():
        val[k] = round(v,6)
    if output == 'table':
        return table
    else:
        return val

print("start")
k = ['18.0,22.0,20.0,41.0,26.0,37.0,53.0,125.0,25.0,25.0,25.0,29.0,14.0,34.0,13.0,40.0,15.0,31.0,31.0,25.0,32.0,16.0,191.0,43.0,19.0,22.0,32.0,30.0,11.0,180.0,176.0,187.0,32.0,43.0,81.0,20.0,34.0,165.0,48.0,120.0,210.0,164.0,191.0,22.0,51.0,534.0,178.0,121.0,L,2000', '0.0,0.0,0.0,0.0,0.0,2.0,0.0,3.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,40.0,0.0,74.0,63.0,L,750', '0.0,0.0,0.0,0.0,0.0,10.0,0.0,17.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,11.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,13.0,0.0,90.0,0.0,L,2700', '42.0,67.0,18.0,835.0,157.0,102.0,1.0,18.0,5.0,1.0,119.0,1.0,198.0,51.0,223.0,0.0,370.0,35.0,56.0,0.0,50.0,0.0,13.0,2.0,306.0,316.0,346.0,218.0,160.0,0.0,0.0,0.0,251.0,38.0,38.0,38.0,31.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,13.0,0.0,90.0,0.0,G,850', '2573.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,20.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,L,30', '0.037,0.056,0.057,0.066,0.054,0.057,0.05,0.1,0.061,0.039,0.044,0.04,0.056,0.056,0.04,0.046,0.07,0.082,0.064,0.05,0.055,0.012,0.082,0.031,0.04,0.1,0.08,0.078,0.09,0.215,0.141,0.104,0.054,0.139,0.266,0.143,0.071,0.149,0.008,0.107,0.102,0.142,0.17,0.081,0.018,1.644,0.144,0.047,G,1.4', '0.019,0.048,0.028,0.058,0.085,0.06,0.089,0.021,0.04,0.037,0.086,0.02,0.053,0.078,0.07,0.027,0.08,0.104,0.069,0.06,0.08,0.039,0.048,0.04,0.09,0.22,0.26,0.115,0.12,0.052,0.051,0.041,0.13,0.09,0.132,0.141,0.117,0.064,0.024,0.11,0.056,0.049,0.055,0.402,0.09,0.161,0.114,0.076,G,1.6', '0.594,0.655,0.48,0.983,0.979,0.302,1.11,1.324,0.234,0.649,0.305,0.1,0.249,1.262,0.5,0.116,0.375,0.734,0.418,0.507,0.525,0.254,0.845,0.334,0.4,0.4,1.0,0.647,0.2,0.446,0.51,0.124,0.742,0.745,2.09,0.978,0.639,0.936,0.386,0.41,0.464,0.372,0.281,3.607,0.925,3.08,7.012,4.671,G,15', '0.594,0.655,0.48,0.983,0.979,0.302,1.11,1.324,0.234,0.649,0.305,0.1,0.249,1.262,0.5,0.116,0.375,0.734,0.418,0.507,0.525,0.254,0.845,0.334,0.4,0.4,1.0,0.647,0.2,0.446,0.51,0.124,0.742,0.745,2.09,0.978,0.639,0.936,0.386,0.41,0.464,0.372,0.281,3.607,0.925,3.08,7.012,4.671,L,300', '0.08,0.247,0.224,0.138,0.291,0.032,0.081,0.312,0.124,0.084,0.073,0.08,0.066,0.463,0.194,0.12,0.09,0.141,0.209,0.184,0.061,0.071,0.1,0.067,0.099,0.106,0.247,0.138,0.129,0.061,0.106,0.082,0.165,0.219,0.169,0.091,0.175,0.157,0.073,0.123,0.123,0.142,0.202,0.104,0.031,0.473,0.386,0.184,G,1.4', '0.08,0.247,0.224,0.138,0.291,0.032,0.081,0.312,0.124,0.084,0.073,0.08,0.066,0.463,0.194,0.12,0.09,0.141,0.209,0.184,0.061,0.071,0.1,0.067,0.099,0.106,0.247,0.138,0.129,0.061,0.106,0.082,0.165,0.219,0.169,0.091,0.175,0.157,0.073,0.123,0.123,0.142,0.202,0.104,0.031,0.473,0.386,0.184,L,55', '15.0,23.0,10.0,19.0,46.0,76.0,89.0,9.0,43.0,22.0,97.0,11.0,34.0,25.0,66.0,19.0,38.0,33.0,18.0,57.0,64.0,25.0,24.0,109.0,14.0,15.0,80.0,105.0,9.0,132.0,115.0,72.0,129.0,61.0,65.0,52.0,63.0,160.0,57.0,42.0,152.0,73.0,152.0,17.0,7.0,87.0,13.0,27.0,G,240', '15.0,23.0,10.0,19.0,46.0,76.0,89.0,9.0,43.0,22.0,97.0,11.0,34.0,25.0,66.0,19.0,38.0,33.0,18.0,57.0,64.0,25.0,24.0,109.0,14.0,15.0,80.0,105.0,9.0,132.0,115.0,72.0,129.0,61.0,65.0,52.0,63.0,160.0,57.0,42.0,152.0,73.0,152.0,17.0,7.0,87.0,13.0,27.0,L,1300', '0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.04,8.75,0.0,4.6,1.79,G,2.4', '13.7,97.0,80.4,5.9,127.7,3.2,7.4,12.1,36.6,2.2,15.0,53.0,6.0,131.2,45.0,7.4,9.2,12.2,57.0,48.2,18.8,14.8,18.2,4.9,30.0,30.0,69.0,58.1,43.0,0.0,1.1,0.0,35.3,85.0,40.0,5.6,89.2,1.3,24.9,0.0,1.1,0.0,0.7,2.1,0.0,0.6,3.4,0.5,G,200', '0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.2,0.0,0.0,20.0,3.9,L,50', '0.54,0.79,0.37,0.66,1.58,0.68,0.19,0.52,0.15,0.3,0.43,0.15,0.2,2.13,0.09,0.02,0.22,0.41,0.11,0.08,0.55,0.0,0.52,0.04,1.89,1.5,0.7,0.21,1.0,1.57,0.79,1.64,2.26,0.88,0.13,1.13,0.78,0.86,0.01,0.63,1.09,0.93,1.64,0.01,0.85,0.31,2.95,0.51,G,7', '0.54,0.79,0.37,0.66,1.58,0.68,0.19,0.52,0.15,0.3,0.43,0.15,0.2,2.13,0.09,0.02,0.22,0.41,0.11,0.08,0.55,0.0,0.52,0.04,1.89,1.5,0.7,0.21,1.0,1.57,0.79,1.64,2.26,0.88,0.13,1.13,0.78,0.86,0.01,0.63,1.09,0.93,1.64,0.01,0.85,0.31,2.95,0.51,L,800', '7.9,6.5,7.4,13.2,4.9,207.5,14.8,4.1,76.0,3.5,108.6,0.0,75.2,16.2,45.5,0.4,126.3,43.0,38.2,15.5,207.0,1.3,4.5,0.2,830.0,400.0,541.9,212.7,250.0,10.7,15.5,10.9,437.1,177.0,24.8,41.6,101.6,9.2,1.3,0.0,11.4,9.5,10.9,0.0,1.0,4.3,0.1,1.8,G,75', '10.0,9.0,10.0,33.0,7.0,45.0,21.0,5.0,40.0,9.0,160.0,26.0,27.0,13.0,105.0,23.0,36.0,37.0,45.0,22.0,72.0,25.0,17.0,16.0,51.0,117.0,81.0,92.0,120.0,25.0,26.0,84.0,232.0,42.0,25.0,24.0,47.0,18.0,56.0,17.0,46.0,16.0,43.0,3.0,59.0,255.0,32.0,13.0,G,800', '10.0,9.0,10.0,33.0,7.0,45.0,21.0,5.0,40.0,9.0,160.0,26.0,27.0,13.0,105.0,23.0,36.0,37.0,45.0,22.0,72.0,25.0,17.0,16.0,51.0,117.0,81.0,92.0,120.0,25.0,26.0,84.0,232.0,42.0,25.0,24.0,47.0,18.0,56.0,17.0,46.0,16.0,43.0,3.0,59.0,255.0,32.0,13.0,L,2300', '24.0,22.0,20.0,35.0,26.0,22.0,73.0,48.0,26.0,24.0,52.0,16.0,24.0,34.0,37.0,29.0,29.0,38.0,30.0,44.0,37.0,20.0,28.0,40.0,46.0,41.0,76.0,58.0,60.0,130.0,132.0,105.0,25.0,69.0,108.0,52.0,66.0,167.0,31.0,151.0,156.0,103.0,137.0,86.0,97.0,642.0,285.0,214.0,G,1000', '24.0,22.0,20.0,35.0,26.0,22.0,73.0,48.0,26.0,24.0,52.0,16.0,24.0,34.0,37.0,29.0,29.0,38.0,30.0,44.0,37.0,20.0,28.0,40.0,46.0,41.0,76.0,58.0,60.0,130.0,132.0,105.0,25.0,69.0,108.0,52.0,66.0,167.0,31.0,151.0,156.0,103.0,137.0,86.0,97.0,642.0,285.0,214.0,L,3000', '11.0,11.0,10.0,12.0,12.0,13.0,42.0,24.0,12.0,14.0,47.0,8.0,10.0,19.0,19.0,10.0,13.0,25.0,16.0,15.0,20.0,10.0,22.0,23.0,81.0,70.0,38.0,42.0,21.0,65.0,42.0,59.0,27.0,23.0,33.0,14.0,21.0,33.0,27.0,64.0,45.0,40.0,46.0,9.0,18.0,392.0,32.0,34.0,G,340', '0.27,0.37,0.34,0.3,0.43,0.7,0.61,0.34,0.47,0.23,1.46,0.6,0.64,0.64,0.8,0.21,0.86,1.03,0.8,0.42,1.48,0.34,0.28,0.8,1.8,2.57,1.3,1.6,0.2,1.96,2.74,3.45,0.47,1.4,1.47,2.14,0.73,3.11,0.42,1.48,2.7,2.23,1.95,0.5,4.61,5.73,0.39,0.71,G,7', '0.27,0.37,0.34,0.3,0.43,0.7,0.61,0.34,0.47,0.23,1.46,0.6,0.64,0.64,0.8,0.21,0.86,1.03,0.8,0.42,1.48,0.34,0.28,0.8,1.8,2.57,1.3,1.6,0.2,1.96,2.74,3.45,0.47,1.4,1.47,2.14,0.73,3.11,0.42,1.48,2.7,2.23,1.95,0.5,4.61,5.73,0.39,0.71,L,50', '0.17,0.17,0.13,0.24,0.25,0.67,0.4,0.28,0.18,0.16,0.47,0.06,0.16,0.2,0.19,0.17,0.18,0.24,0.22,0.27,0.39,0.28,0.36,0.35,0.36,0.38,0.23,0.56,0.11,1.04,1.0,1.28,0.21,0.42,1.24,0.54,0.41,1.18,0.83,1.09,1.42,0.88,0.91,0.52,39.3,4.34,0.57,0.42,G,12', '0.17,0.17,0.13,0.24,0.25,0.67,0.4,0.28,0.18,0.16,0.47,0.06,0.16,0.2,0.19,0.17,0.18,0.24,0.22,0.27,0.39,0.28,0.36,0.35,0.36,0.38,0.23,0.56,0.11,1.04,1.0,1.28,0.21,0.42,1.24,0.54,0.41,1.18,0.83,1.09,1.42,0.88,0.91,0.52,39.3,4.34,0.57,0.42,L,40', '0.059,0.049,0.066,0.045,0.017,0.084,0.127,0.204,0.019,0.081,0.076,0.037,0.027,0.088,0.021,0.039,0.029,0.069,0.017,0.039,0.083,0.05,0.104,0.075,0.179,0.191,0.17,0.157,0.077,0.194,0.225,0.267,0.046,0.07,0.176,0.189,0.049,0.233,0.058,0.191,0.327,0.218,0.204,0.318,2.858,1.22,0.058,0.095,G,0.9', '0.059,0.049,0.066,0.045,0.017,0.084,0.127,0.204,0.019,0.081,0.076,0.037,0.027,0.088,0.021,0.039,0.029,0.069,0.017,0.039,0.083,0.05,0.104,0.075,0.179,0.191,0.17,0.157,0.077,0.194,0.225,0.267,0.046,0.07,0.176,0.189,0.049,0.233,0.058,0.191,0.327,0.218,0.204,0.318,2.858,1.22,0.058,0.095,L,10', '0.0,0.0,0.0,0.1,0.1,0.2,0.2,0.3,0.3,0.3,0.3,0.4,0.4,0.4,0.5,0.5,0.6,0.6,0.6,0.6,0.6,0.6,0.7,0.7,0.9,0.9,0.9,0.9,0.9,1.1,1.1,1.2,1.3,1.6,1.8,2.3,2.5,2.6,2.8,2.8,3.4,4.2,5.8,9.3,19.7,25.4,29.7,52.6,G,30', '0.0,0.0,0.0,0.1,0.1,0.2,0.2,0.3,0.3,0.3,0.3,0.4,0.4,0.4,0.5,0.5,0.6,0.6,0.6,0.6,0.6,0.6,0.7,0.7,0.9,0.9,0.9,0.9,0.9,1.1,1.1,1.2,1.3,1.6,1.8,2.3,2.5,2.6,2.8,2.8,3.4,4.2,5.8,9.3,19.7,25.4,29.7,52.6,L,280', '237.0,188.0,175.0,320.0,211.0,238.0,286.0,372.0,170.0,229.0,369.0,138.0,168.0,285.0,252.0,146.0,194.0,211.0,243.0,299.0,276.0,233.0,282.0,325.0,379.0,762.0,606.0,296.0,330.0,330.0,375.0,521.0,213.0,389.0,244.0,202.0,316.0,343.0,246.0,171.0,270.0,472.0,405.0,318.0,156.0,813.0,476.0,382.0,G,2500', '237.0,188.0,175.0,320.0,211.0,238.0,286.0,372.0,170.0,229.0,369.0,138.0,168.0,285.0,252.0,146.0,194.0,211.0,243.0,299.0,276.0,233.0,282.0,325.0,379.0,762.0,606.0,296.0,330.0,330.0,375.0,521.0,213.0,389.0,244.0,202.0,316.0,343.0,246.0,171.0,270.0,472.0,405.0,318.0,156.0,813.0,476.0,382.0,L,2800', '5.0,3.0,3.0,69.0,4.0,140.0,60.0,167.0,18.0,2.0,27.0,2.0,19.0,6.0,65.0,4.0,28.0,6.0,27.0,30.0,16.0,39.0,146.0,78.0,213.0,226.0,14.0,3.0,41.0,217.0,218.0,221.0,17.0,25.0,5.0,2.0,33.0,218.0,420.0,163.0,222.0,218.0,217.0,5.0,85.0,30.0,387.0,388.0,G,600', '0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.015,0.0,0.0,0.0,0.0,0.323,17.8,1.038,0.168,G,2.1', '0.08,0.07,0.054,0.1,0.1,0.751,0.105,1.148,0.017,0.063,0.132,0.063,0.022,0.145,0.042,0.013,0.024,0.044,0.035,0.016,0.07,0.017,0.769,0.055,0.063,0.041,0.152,0.252,0.012,2.467,2.449,2.427,0.084,0.046,0.152,0.04,0.049,2.477,0.285,0.973,3.384,2.459,2.441,0.16,0.063,5.903,0.674,0.228,L,10', '-0.08,-0.07,-0.054,-0.1,-0.1,-0.751,-0.105,-1.148,-0.017,-0.063,-0.132,-0.063,-0.022,-0.145,-0.042,-0.013,-0.024,-0.044,-0.035,-0.016,-0.07,-0.017,-0.769,-0.055,-0.063,-0.041,-0.152,-0.252,-0.012,-2.467,-2.449,-2.427,-0.084,-0.046,-0.152,-0.04,-0.049,-2.477,-0.285,-0.958,-3.384,-2.459,-2.441,-0.16,0.26,11.897,0.364,-0.06,G,0']
m = gen_matrix(48,len(k))
for i in k:
	constrain(m, i)
print("hey")
obj(m,'3.89,5.13,4.64,9.58,6.03,2.98,11.95,20.45,5.8,5.88,3.65,9.32,2.92,7.66,2.18,9.34,2.87,6.97,7.37,4.97,7.34,3.4,39.62,9.56,3.74,4.33,5.5,4.35,1.29,22.04,21.19,23.32,5.42,8.95,14.45,3.88,6.64,18.71,11.29,21.21,25.48,19.4,24.37,3.26,2.72,28.88,0.1,0.1,0.0')
print(minz(m))

