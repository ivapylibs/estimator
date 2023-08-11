#================================ dtbackpass ===============================
'''!
@brief  Implementation of basic discrete time backwards pass schemes

'''
#================================ dtbackpass ===============================
#
# @author   Patricio A. Vela,       pvela@gatech.edu
# @date     2023/08/11              [created from ivaMatlibs version]
#
# NOTE:
#   Using 80 columns, 2 space indent, tab expands to 4 spaces.
#
#================================ dtbackpass ===============================

#
#---------------------------------------------------------------------------
#================================= Kalman ==================================
#---------------------------------------------------------------------------
#
#(

class bpKalman(object):
  '''!
  @brief    Base class for backwards pass of Kalman smoother.

  Backward pass of Kalman smoother.  Since there are many implementations,
  it is done as a class.  This is the base class that implements the
  simplest version, which is the Rauch, Tung, Striebel version.
 
 
  References:
    http://arl.cs.utah.edu/resources/Kalman%20Smoothing.pdf
 
    HE Rauch, F Tung, and CT Striebel  
    Maximum likelihood estimates of linear dynamic systems.
    J. Amer. Inst. Aeronautics and Astronautics, 3(8):1445–1450, 1965.
 
    John Crassidis and John Junkins.
    Optimal Estimation of Dynamic Systems.
    CRC Press, 2004.
 
  @todo WARNING: Not yet coded up for non-zero B.
  ''' 
  #------------------------------ bpKalman -----------------------------
  '''!
  @brief  Constructor.  Requires the A matrix.  The initial backwards
          recursion state can be set later.
  
  @param[in]  A   State transition matrix.
  @param[in]  XT  The terminal (in time) state.
  @param[in]  PT  The terminal (in time) covariance.
 
  @todo   Make sure that interpretation of XT, PT correct.
  '''
  def __init__(self, A, XT, PT)
    #properties
    #  A;        %< The state transition matrix.
    #
    #  K;        %< The backwards Kalman gain.
    #
    #  x_rev;    %< State estimate at the end of the window.
    #  P_rev;    %< Covariance estimate at the end of the window.
    #
    #  K_rev;    %< Kalman gain applied at the end of the window.
    #end

    this.A = A;
  
    if (nargin >= 2)
      this.x_rev = XT;
  
    if (nargin == 3)
      this.P_rev = PT;
  
  #--------------------------- setFilterState --------------------------
  '''!
  @brief  Set current estimate of the state vector and state covariance.
  
  @param[in]  XT  The state vector.
  @param[in]  PT  The state covariance.
  '''
  def setFilterState(this, xT, PT):

    this.x_rev = xT
    this.P_rev = PT

  #--------------------------- getFilterState --------------------------
  '''!
  @brief  Get current estimate of the state vector and state covariance.
  
  @param[out] xT  The state vector.
  @param[out] PT  The state covariance.
  '''
  def [xT, PT] = getFilterState(this)

    xT = this.x_rev;
    PT = this.P_rev;

  #------------------------------ process ------------------------------
  '''!
  @brief  Given the current Kalman filter correction and next prediction,
          perform the backward recursion.
  
  @param[in]  corr    Current corrected state.
  @param[in]  pred    Current predicted state.
  @param[in]  u       The external/control input.
  
  @param[out] xS      The smoothed state.
  @param[out] PS      The smoothed covariance.
  '''
  def [xS, PS] = process(this, corr, pred, u):

    K_rev = corr.P * transpose(this.A) * inv(pred.P);

    this.P_rev = corr.P + K_rev * (this.P_rev - pred.P) * transpose(K_rev);
    this.x_rev = corr.x + K_rev * (this.x_rev - pred.x);

    this.K_rev = K_rev;

    if (nargout >= 1)
      xS = this.x_rev;
    if (nargout == 2)
      PS = this.P_rev;

#
#)

#
#---------------------------------------------------------------------------
#================================ FixedLag =================================
#---------------------------------------------------------------------------
#
#(

class FixedLag(Kalman):
  '''!
  @brief    The backwards process for an extended RTS fixed-lag smoother.

  Reference:
    https://cse.sc.edu/~terejanu/files/tutorialKS.pdf
    G. Terejanu (Dept of CSE, Univ. Buffalo)
    Crib Sheet: Linear Kalman Smoothing.
  '''

  #------------------------------ FixedLag ------------------------------
  '''!
  @brief  Constructor for fixed lag backward pass of extended RTS smoother.
  
  @param[in]  A   State transition matrix.
  @param[in]  B   Input matrix (optional, set to empty).
  @param[in]  C   Measurement matrix.
  @param[in]  Q   Process noise matrix.
  @param[in]  R   Measurement noise matrix.
  @param[in]  XT  The terminal (in time) state.
  @param[in]  PT  The terminal (in time) covariance.
  
  @todo   Make sure that interpretation of XT, PT correct.
  '''
  def __init__(self, nLag, A, B, C, Q, MT, XT = None, PT = None)

    #nLag;          %< Number of discrete time steps of the lag.
    #corrH;         %< Correction history.
    #predH;         %< Prediction history.
    #smoothH        %< Smoother history (gain, smoother covariance).
    #B;             %< Input matrix.
    #C;             %< Measurement matrix.
    #Q;             %< Process noise.
    #M;             %< Smoothing gain.
    #isInit;        %< Has smoothing been initialized?

    if (nargin < 7):
      XT = [];
      PT = [];

    if (nargin < 6):
      MT = 1;

    super(FixedLag,self).__init__(A, XT, PT);

    this.B = B;
    this.C = C;

    this.Q = Q;

    this.nLag = nLag;
    this.isInit = false;

    this.M = MT;

  #------------------------------- reset -------------------------------
  '''!
  @brief  Reset the backwards process.
 
  @param[in]  MT  Set initial smoothing gain (optional).
  '''
  def reset(this, MT):

    this.isInit = false;
    this.corrH = [];
    this.predH = [];
  
    if ( (nargin == 2) && (~isempty(MT)) ):
      this.M = MT;
    else:
      this.M = 1;
  

  #----------------------------- initialize ----------------------------
  '''!
  @brief  Pass all of the smoothed information needed to initialize the
          fixed lag backwards process.
  
  @param[in]  XT      State.
  @param[in]  PT      Covariance.
  @param[in]  corrH   Correction history.
  @param[in]  predH   Prediction history.
  '''
  def initialize (this, XT, PT, corrH, predH):

    if ( (length(predH) >= this.nLag) && (length(corrH) >= this.nLag) ):
      this.isInit = true;

    this.setFilterState(XT, PT);

    this.corrH = corrH(end-this.nLag+1:end);
    this.predH = predH(end-this.nLag+1:end);

    this.smoothH = struct('K',[],'P',[]);
    this.smoothH(this.nLag).x = this.x_rev;
    this.smoothH(this.nLag).P = this.P_rev;

    for ii = (this.nLag-1):-1:1
      this.process( this.corrH(ii), this.predH(ii+1) );

      this.smoothH(ii).x = this.x_rev;
      this.smoothH(ii).P = this.P_rev;
      this.smoothH(ii).K = this.K_rev;

  #------------------------------ process ------------------------------
  '''!
  @brief  Incorporate the new information into the fixed lag backwards
          process.  Argument is previous correction and current prediction.
  
  @param[in]  corr    Previous correction.
  @param[in]  pred    Current prediction.
  @param[in]  u       External/control input.
  @param[in]  dx      Kalman update.
  @param[in]  Kf      Kalman gain. (Double check)
  
  @todo   Confirm dx and Kf meaning.
  '''
  def [xS, PS] = process(this, corr, pred, u, dx, Kf):

    #fprintf('I am initialized? %d.\n', this.isInit);
  
    if (nargin < 4)
      [xS, PS] = this.process@bpKalman( corr, pred );
      return;
  
    # @todo go read up on things. Everything below is not implemented. Why?
    # @todo isInit is never triggered to be true.
  
    if (isempty(this.corrH))                  %! If empty, store data and return.
      this.corrH = corr;
      this.predH = pred;
  
      xS = [];                                %! No estimates available.
      PS = [];
      return;
    elseif (length(this.corrH) < this.nLag)   %! If history not full, keep storing.
      corrH(end+1) = corr;
      predH(end+1) = pred;
  
      xS = [];                                %! No estimates available.
      PS = [];
      return;
    end
  
    if (~this.isInit)                         %! Assumes M has been initialized already.
      this.x_rev = corr.x;
      this.P_rev = corr.P;
  
      %@todo Still need to snag corr and pred properly.
  
      this.smoothH = struct('x',[],'P',[],'K',[]);
      this.smoothH(this.nLag) = this.x_rev;
      this.smoothH(this.nLag) = this.P_rev;
  
      for ii = (this.nLag-1):-1:1
        this.process@bpKalman( this.corrH(ii), this.predH(ii+1) );
  
        this.smoothH(ii).x = this.x_rev;
        this.smoothH(ii).P = this.P_rev;
        this.smoothH(ii).K = this.K_rev;
      end
  
    else                          %! Ready to start fixed-lag backwards process.
      disp('I am here!');
      %-- Compute the smoother Kalman gain for reverse direction.
      K_rev = corr.P * transpose(this.A) * inv(pred.P);
  
      %-- Update the correction gain factor.
      this.M = this.M * K_rev;
  
      %-- Update the smoother's lagging state estimate (forward one timestep)
      Lres = this.Q * inv(transpose(this.A)) * inv(this.corrH(1).P);
      xres = this.x_rev - this.corrH(1).x;
      %TODO: inv(transpose(A)) is effectively a constant for linear systems.
      %TODO: Later on modify to allow for changes.
  
      this.x_rev = this.A * this.x_rev + Lres * xres + this.M * dx;
      if (~isempty(this.B))
        this.x_rev = this.x_rev + this.B * u;
      end
  
      %-- Update the smoother's lagging covariance estimate.
      invK_rev = inv(this.smoothH(1).K);
      dP = invK_rev * (this.corrH(1).P - this.P_rev) * transpose(invK_rev);
      dM = this.M * Kf * this.C * pred.P * transpose(this.M);
  
      this.P_rev = this.predH(2).P - dP - dM;
  
      
      %-- Toss out earliest part of history, and tack on newest.
      this.smoothH(this.nLag).K = K_rev;
  
      this.corrH(1) = [];
      this.predH(1) = [];
      this.smoothH(1) = [];
  
      this.corrH(this.nLag) = corr;
      this.predH(this.nLag) = pred;
  
      %-- Process return values.
      xS = this.x_rev;
      PS = this.P_rev;
    end

#
#)

#
#================================ dtbackpass ===============================

