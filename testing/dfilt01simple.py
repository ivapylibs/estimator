#!/usr/bin/python
#============================ dfilt01simple ============================
'''!
@brief  Test of discrete time linear filter for simple scalar signal
'''
#============================ dfilt01simple ============================
#
# @author   Patricio A. Vela,   pvela@gatech.edu
# @date     2023/08/10          [created]
#
#============================ dfilt01simple ============================

#--[0] Setup workspace.
import numpy as np
import estimator.dtfilters as df


#--[1] Define the system and mismatched initial conditions.
A = np.array( [[1, 0.5], [0, 1]])
C = np.array( [[1, 0]] )
L = df.calcGainByDARE(A, C, 0.5*np.identity(2), 0.15)

x  = np.array([[1] ,[-0.25]])
x0 = np.array([[0.9], [0.2]])


#--[2] Instantiate estimator and run.
pEst = df.Linear(A, C, L, x0)

for i in range(10):
  x = np.matmul(A, x)

  pEst.process( np.matmul(C,x) )

  print((x, pEst.x_hat))


#--[3] Print final error.
print(np.linalg.norm(pEst.x_hat - x))

#
#============================ dfilt01simple ============================
