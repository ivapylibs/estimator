#=============================== dtfilters ===============================
'''!
@brief  Implementation of basic discrete time filter schemes.

Here we differentiate a filter as applying to a signal absent external
signaling and an observer as applying to a signal with a known external
(control) signal that gets factored in.  Due to the mixing of language
(e.g., Kalman filter), the differences get muddled at higher levels of
implementation (deeper sub-classes).  Don't fret it, just look at the 
class description.  
'''
#=============================== dtfilters ===============================
#
# @author   Patricio A. Vela,       pvela@gatech.edu
# @date     2023/08/10              [created]
#
# NOTE:
#   Using 80 columns, 2 space indent, tab expands to 4 spaces.
#
#=============================== dtfilters ===============================

import numpy as np
import scipy.linalg as la

#
#-------------------------------------------------------------------------
#================================ NoFilter ===============================
#-------------------------------------------------------------------------
#

class NoFilter(object):
  '''!
  @brief    Defines the base filtering class, which is no filter.

  The NoFilter base implementation makes sense from a certain perspective.
  It is sometimes desired to be able to switch filtering off to compare
  outcomes.  In some cases, instantiating as a NoFilter instance would
  implement such functionality.
  '''
  
  #-------------------------------- init --------------------------------
  #
  '''!
  @brief    Constructor for nofilter instance.
  '''

  def __init__(self, x0 = None):

    self.x_hat = x0
    self.isInit = x0 is not None


  #------------------------------ predict ------------------------------
  #
  '''!
  @brief    Prediction step; static prediction for base class.
  '''
  def predict(self):

    if self.isInit:
      return self.x_hat
    else:
      return None


  #------------------------------ correct ------------------------------
  #
  '''!
  @brief    Obtain measurement and use it to perform correction of the
            predicted state.

  @param[in]    z   The measurement (at the current time)

  @param[out]   x   The estimated state. 
  @param[out]   y   The estiamted output (same as state for NoFilter).

  The no filter instance simply accepts the measurement as the corrected
  value.  Actually filtering sub-classes would perform filtering.
  '''
  def correct(self, z):

    self.x_hat = z

    if (not self.isInit):
      self.isInit = True

    return z, z


  #------------------------------ process ------------------------------
  '''!
  @brief    Apply the filter update process given a new measurement.

  @param[in]    z   The measurement (set to None if not available).
  '''
  def process(self, z):

    self.predict()
    if (z is not None):
      self.correct(z)


  #------------------------------ getState -----------------------------
  #
  '''!
  @brief    Get the internal state of the filter.
  '''
  def getState(self):
    return self.x_hat

  #------------------------------ setState -----------------------------
  #
  '''!
  @brief    Set the internal state of the filter.
  '''
  def getState(self, x):
    self.x_hat = x
    self.isInit = True


  #------------------------------- reset -------------------------------
  #
  '''!
  @brief     Reset the filter state (to not initialized).
  '''
  def reset(self):

    self.x_hat = None
    self.isInit = False

  #---------------------------- measurement ----------------------------
  #
  '''!
  @brief     Get the estimated measurement.

  For NoFilter just return the current state.
  '''
  def measurement(self):
    return self.x_hat

  #---------------------------- displayState ---------------------------
  #
  '''!
  @brief    Display/print the current state of the system.
  '''
  def displayState(self):

    print(self.x_hat)



#
#-------------------------------------------------------------------------
#================================= Linear ================================
#-------------------------------------------------------------------------
#


class Linear(NoFilter):
  '''!
  @brief    Defines simple linear filtering class.

  Here, a filter operates as a predictor-corrector system:
  \begin{equation}
  \begin{split}
    x_{k+1} & = A x_k \\
    x_k & = x_k + L (z_k - C x_k) \\
    y_k = C x_k
  \end{equation}
  which doesn't squarely agree with the controls version (where
  prediction and correction are done simultaneously).  Here the
  separation of prediction and correction means that the error to
  correct is based on the predicted state (not previous state).
  '''
  
  #-------------------------------- init --------------------------------
  #
  '''!
  @brief    Constructor for linear filter instance.
  '''

  def __init__(self, A, C, L, x0 = None):

    self.A = A
    self.C = C
    self.L = L

    self.x_hat = x0
    self.isInit = x0 is not None


  #------------------------------ predict ------------------------------
  #
  '''!
  @brief    Prediction step; static prediction for base class.
  '''
  def predict(self):

    if self.isInit:
      self.x_hat = np.matmul(self.A , self.x_hat)
      return self.x_hat
    else:
      return None


  #------------------------------ correct ------------------------------
  #
  '''!
  @brief    Obtain measurement and use it to perform correction of the
            predicted state.

  @param[in]    y       The measurement (at the current time)

  @param[out]   x_hat   The estimated state. 
  @param[out]   y_hat   The estimated output.

  The no filter instance simply accepts the measurement as the corrected
  value.  Actually filtering sub-classes would perform filtering.
  '''
  def correct(self, y):

    if (self.isInit):
      self.x_hat = self.x_hat + np.matmul(self.L ,  \
                                          y - np.matmul(self.C,self.x_hat) )

      return self.x_hat, np.matmul(self.C, self.x_hat)
    else:
      return None, None

  #------------------------------ getState -----------------------------
  #
  '''!
  @brief    Get the internal state of the filter.
  '''
  def getState(self):
    return self.x_hat

  #------------------------------ setState -----------------------------
  #
  '''!
  @brief    Set the internal state of the filter.
  '''
  def setState(self, x):
    self.x_hat = x
    self.isInit = True


  #------------------------------- reset -------------------------------
  #
  '''!
  @brief     Reset the filter state (to not initialized). Keep (A, C, K).
  '''
  def reset(self):

    self.x_hat = None
    self.isInit = False

  #---------------------------- measurement ----------------------------
  #
  '''!
  @brief     Get the estimated measurement.

  For NoFilter just return the current state.
  '''
  def measurement(self):
    return np.matmul(self.C,self.x_hat)

  #---------------------------- displayState ---------------------------
  #
  '''!
  @brief    Display/print the current state of the system.
  '''
  def displayState(self):

    print(self.x_hat)




#--------------------------- calcGainByDARE --------------------------
#
def calcGainByDARE(A,C,Q,R):

  X = la.solve_discrete_are(np.transpose(A), np.transpose(C), Q, R)
  
  # L = A * X * (C') * inv( R + C*X*(C') ) 
  L = np.matmul(A , np.matmul(X ,  \
                    np.matmul( np.transpose(C) ,  \
              la.inv(R + np.matmul(C , np.matmul(X , np.transpose(C)))) ) ) )
  return L 


#
#=============================== dtfilters ===============================
