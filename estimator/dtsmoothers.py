
%============================= smootherKalman ============================
%
% @class    smootherKalman
%
% @brief    A linear Kalman fixed-lag smoother.
%
% Perform Kalman smoothing on a set of discrete data under the assumption
% a discrete linear system.
%
% smoother = smoothKalman(A, B, C, Q, R, XO, P0)
%
% Create a Kalman smoother for the discrete linear system:
%
% \f{equation}
% \begin{split}
%   \dot x & = A x + B u \\
%        y & = C x
% \end{split}\f}
%
% \f$Q\f$ describes system noise, \f$R\f$ describes measurement noise,
% and the initial state and covariance are \f$X0\f$ and \f$P0\f$, respectively.
% Both \f$X0\f$ and \f$P0\f$ are optional, so long as they get specified
% prior to actually smoothing.  The smoother needs to be initialized.
%
%  NOTE:
%    - Can initialize with y0 and will reconstruct x0 using pinv(C).
%
%  Reference for implementation:
%
%   http://fedc.wiwi.hu-berlin.de/xplore/tutorials/xlghtmlnode57.html
%   (stale)
%   http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/tutorials/xlghtmlnode57.html
%   http://arl.cs.utah.edu/resources/Kalman%20Smoothing.pdf
%
%============================= smootherKalman ============================

%
% @author   Patricio A. Vela, pvela@gatech.edu
% @date     2016/06/03
%
% @note
%    indent is 2 spaces.
%    tabstop is 4 spaces (with conversion to spaces).
%
%============================= smootherKalman ============================
classdef smootherKalman < dfKalman

%============================ Member Variables ===========================
%
%(

properties
  P0;           %< The initial covariance.

  dback;        %< The backwards process to apply.

  pred;         %< The stored prediction information.
  corr;         %< The stored correction information.
  smoothed;     %< The stored smoothed data.

  tau;          %< How many pieces of information will be smoothed.
end


%)
%
%============================ Member Functions ===========================
%
%(
methods

  %=========================== smootherKalman ==========================
  %
  % @brief  Constructor. Requires the A, B, Q, and R matrices.
  %
  % The initial window for the smoother is set to infinity
  % meaning that the smoother keeps all data.
  %
  % @param[in]  A   State transition matrix.
  % @param[in]  B   Input matrix (optional, set to empty).
  % @param[in]  C   Measurement matrix.
  % @param[in]  Q   Process noise matrix.
  % @param[in]  R   Measurement noise matrix.
  % @param[in]  X0  The initial state (optional).
  % @param[in]  P0  The initial covariance (optional).
  %
  function this = smootherKalman(A, B, C, Q, R, X0, P0)

  if (nargin < 6)
    X0 = [];
  end

  if (nargin < 7)
    P0 = [];
  end

  this@dfKalman(A, B, C, Q, R, X0, P0);

  if ((nargin == 7) && (~isempty(P0)))
    this.P0 = P0;
  else
    this.P0 = [];
  end

  this.dback = bpKalman(A);
  this.tau = inf;
  this.tauReady = 1;

  end


  %=========================== setFilterState ==========================
  %
  % @brief  Set the current/initial estimate of the state vector and state
  %         covariance.
  %
  % @param[in]  X0  The state vector.
  % @param[in]  P0  The state covariance.
  %
  function setFilterState(this, X0, P0)

  this.P0 = P0;
  this.setFilterState@dfKalman(X0, P0);

  end

  %============================= emptyState ============================
  %
  % @brief  Returns an empty structure containing the necessary state
  %         information.
  %
  function eState = emptyState(this)

  eState = struct('x',{},'P',{});

  end

  %=============================== reset ===============================
  %
  % @brief  Reset smoother. Clear any past values in the history stacks.
  %
  function reset(this)

  this.pred = [];
  this.corr = [];

  this.reset@dfKalman();

  end

  %============================ limitHistory ===========================
  %
  % @brief  Apply a limit to how far back the history can go.
  %
  function limitHistory(this, tau)

  this.tau = tau;

  end

  %=============================== delay ===============================
  %
  % @brief  Return the delay in receiving the smoothed output.
  %
  function dTau = delay(this)

  dTau = this.tau;

  end

  %============================== predict ==============================
  %
  % @brief  If the data is coming in sequentially, then process that way
  %         and store data until no longer the case.
  %
  % @param[out] pX  Predicted state (prior).
  % @param[out] pP  Predicted covariance (prior).
  %
  function [pX, pP] = predict(this)

  if (~this.isInit)
    pX = [];
    pP = [];
    return;
  end

  this.predict@dfKalman();

  if (isempty(this.pred))
    this.pred.x = this.x_hat;
    this.pred.P = this.P_hat;
  elseif (length(this.pred) < this.tau)
    this.pred(end+1).x = this.x_hat;    %! Add new field to the end.
    this.pred(end).P = this.P_hat;      %! Add into the new field at the end.
  else
    this.pred(1) = [];
    this.pred(this.tau).x = this.x_hat;
    this.pred(this.tau).P = this.P_hat;
  end

  end

  %============================== correct ==============================
  %
  % @brief  Runs the correction routine for current measurement.
  %
  % Runs the correction step, unless the filter has not been initialized.
  % In which case, the corrected (posterior) outputs are empty / not defined.
  % Otherwise, corrected (posterior) outputs are available.
  %
  % @param[in]  y       Current measurement.
  % @param[out] cX      Corrected state (posterior).
  % @param[out] cP      Corrected covariance (posterior).
  % @param[out] cY      Corrected measurement (posterior).
  %
  function [cX, cP, cY] = correct(this, y)

  if (~this.isInit)
    cX = [];
    cP = [];
    cY = [];
    return;
  end

  [cX, cP, cY] = this.correct@dfKalman(y);

  if (isempty(this.corr))
    this.corr.x = this.x_hat;
    this.corr.P = this.P_hat;
  elseif (length(this.corr) < this.tau)
    this.corr(end+1).x = this.x_hat;    %! Add new field to the end.
    this.corr(end).P = this.P_hat;      %! Add into the new field at the end.
  else
    this.corr(1) = [];
    this.corr(this.tau).x = this.x_hat;
    this.corr(this.tau).P = this.P_hat;
  end

  end

  %============================== complete =============================
  %
  % @brief  All data available has been passed to the smoother.
  %         Time to complete the process by performing the backwards pass.
  %
  % @param[out] sData   The output smoothed data.
  %
  function [sData] = complete(this)

  this.dback.setFilterState(this.corr(end).x, this.corr(end).P);
  this.smoothed = this.emptyState();

  %==[2] Perform the smoothing operation.
  nSamp = length(this.pred);
  this.smoothed(nSamp) = this.corr(end);
  for ii = (length(this.pred)-1): -1: 1
    [sx, sP] = this.dback.process( this.corr(ii), this.pred(ii+1) );
    this.smoothed(ii).x  = sx;
    this.smoothed(ii).sP = sP;
  end

  if (nargout > 0)
    sData.data = this.smoothed;
    sData.x = this.smoothed(1).x;
    sData.P = this.smoothed(1).P;
  end

  end


  %=============================== apply ===============================
  %
  % @brief  Apply Kalman smoothing to a collection of data.
  %
  % Given a collection of data, apply Kalman smoothing to it.  Smoothing
  % consists of a forward filter pass, followed by a backwards "smoother"
  % pass.  Also returns the final covariance (of single lagged state).
  %
  % @param[in]  yData   The measurements.
  % @param[in]  uData   The inputs.
  % @param[in]  X0      The initial state.
  % @param[in]  P0      The initial covariance.
  %
  % @param[out] sx      The smoothed data.
  % @param[out] sP      The final smoothed covariance.
  %
  function [sx, sP, slag] = apply(this, yData, uData, X0, P0)

  if (nargin >= 4)
    this.setState(X0);
  end

  if (nargin == 5)
    this.setCovariance(P0);
  end

  sx   = zeros(size(X0,1), size(yData,2));
  slag = zeros(size(X0,1), size(yData,2) - this.tau + 1);
  sP   = zeros(size(X0,1),size(X0,1), size(yData,2));
  % @todo Need to populate sP.

  isReady = false;

  %!==[1] Perform the filtering operation.
  %!
  this.clearHistory();
  for ii = 1:size(yData,2)
    [sF, sL] = this.process(yData(:,ii));

    sx(:,ii) = sF;
    if (isReady)
      slag(:,ii-this.tau+1) = sL;
    elseif (~isempty(sL))
      isReady = true;
    end

    %sP = this.dback.P_rev;
  end

  end


%@todo  Review code written by Marion and determine if should be
%       incorporated into this class.
%! %============================ getEstimate ============================
%! %
%! % Accesses a delayed state in the estimator. tauDelay is an index,
%! % implementation-wise, so there is no need to specify a sample period.
%! %
%! %
%! %
%! % Per Dr. Vela's suggestion, the maximum lookback is the most recent half
%! % of the available data.
%! %
%! function [x, isAvailable, tauRet] = getEstimate(this, tauDelay)
%!
%! %==[1] Return if going beyond time horizon
%! if (this.tau > length(this.pred))
%!     tauMax = length(this.pred);
%! else
%!     tauMax = this.tau;
%! end
%! maxHorizon = ceil(tauMax / 2);  % ceil to avoid rounding to 0 available samples
%! if (tauDelay > maxHorizon)
%!     x = this.pred(:, end).x;
%!     isAvailable = false;
%!     tauRet = maxHorizon;
%!     return
%! end
%!
%! %==[2] If valid tauDelay, return data
%! x = this.pred(:, end-tauDelay).x;
%! isAvailable = true;
%! tauRet = tauDelay;
%!
%! end



end
%)

end
%
%============================= smootherKalman ============================




============================== dFixedLagKS ==============================
%
% @class    dFixedLagKS
%
% @brief    Implementation of a discrete time fixed lag Kalman smoother.
%
%
%============================== dFixedLagKS ==============================

%
% @author   Patricio A. Vela, pvela@gatech.edu
% @date     2016/06/10
%
% @note
%    indent is 2 spaces.
%    tabstop is 4 spaces (with conversion to spaces).
%
%============================== dFixedLagKS ==============================
classdef dFixedLagKS < smootherKalman

%============================ Member Variables ===========================
%
%(
properties
  t;            %< Current time.
  isReady;      %< Is the system warmed up and ready (lag buffer is full).
end

%)
%
%============================ Member Functions ===========================
%
%(
methods

  %============================ dFixedLagKS ============================
  %
  % @brief  Constructor for the fixed lag filter.
  %
  % @param[in]  tau     The length / time window of the lag.
  % @param[in]  A   State transition matrix.
  % @param[in]  B   Input matrix (optional, set to empty).
  % @param[in]  C   Measurement matrix.
  % @param[in]  Q   Process noise matrix.
  % @param[in]  R   Measurement noise matrix.
  % @param[in]  X0  The initial state (optional).
  % @param[in]  P0  The initial covariance (optional).
  %
  function this = dFixedLagKS(tau, A, B, C, Q, R, X0, P0)

  if (nargin < 7)
    X0 = [];
  end

  if (nargin < 8)
    P0 = [];
  end

  this@smootherKalman(A, B, C, Q, R, X0, P0);

  this.tau = tau;
  this.t   = 0;

  this.dback = bpFixedLag(tau, A, B, C, Q);

  this.isReady = false;

  end

  %============================== predict ==============================
  %
  % @brief  The prediction step.
  %
  % @param[out] xF  What?
  % @param[out] xL  What?
  %
  % @todo   Need to figure out what out params are and how to set.
  %
  function [xF, xL] = predict(this)

  if (~this.isInit)
    xF = [];
    xL = [];
    return;
  end

  if (this.t < this.tau)
    this.predict@smootherKalman();
    this.t = this.t + 1;
  else
    this.predict@dfKalman();
    if (this.t == this.tau)
      this.t = this.t + 1;
    end

    this.pred.x = this.x_hat;
    this.pred.P = this.P_hat;
  end


  end

  %============================== complete =============================
  %
  % @brief  Enough data has been collected. Use it to perform the backwards
  %  smoothing process so that we have a complete state estimate (which
  %  is basically the current state and the lagged state).
  %
  function complete(this)

  if (this.isReady)
    this.dback.initialize(this.x_hat, this.P_hat, this.corr, this.pred);
  end

  end

  %============================== correct ==============================
  %
  % @brief  The correction step.
  %
  % @param[out] xF  What?
  % @param[out] xL  What?
  %
  % @todo   Need to set output.
  %
  function [xF, xL] = correct(this, y)

  if (~this.isInit)
    xF = [];
    xL = [];
    return;
  end

  if (this.t < this.tau)
    this.correct@smootherKalman(y);
  elseif (this.t == this.tau)
    this.correct@smootherKalman(y);
    this.complete();
    this.clearHistory();

    this.corr.x = this.x_hat;
    this.corr.P = this.P_hat;
  else
    this.correct@dfKalman(y);       % Does not store corr/pred history.

    this.dback.process(this.corr, this.pred, [], this.dx, this.K);

    this.corr.x = this.x_hat;       % Now store it for next time.
    this.corr.P = this.P_hat;
  end

  end

  %
  %============================== process ==============================
  %
  % @brief  Perform the steps all together.
  %
  % @param[out] xCurr   Current estimate of state.
  % @param[out] xLag    Current lagged estimate of state.
  %
  function [xCurr, xLag] = process(this, y)

  this.predict();
  this.correct(y);

  xCurr = this.x_hat;
  xLag  = this.dback.x_rev;

  end

end
%
%)

end
%
%============================== dFixedLagKS ==============================

## dFixedLagRTS NOT SURE ABOUT SOURCE MATLAB.
