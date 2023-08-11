
# MOVE TO dtobservers

%================================ dfKalman ===============================
%
% @class    dfKalman
%
% @brief    Classic Kalman filter for discrete linear systems.
%
% > dkfilt = dfKalman(A, B, C, Q, R, XO, P0)
%
% Create a Kalman filter for the linear system:
%
% \f{equation} \nonumber
% \begin{split}
%    dx/dt & = A x + B u \\
%        y & = C x
% \end{split} \f}
%
% Q describes system noise, R describes measurement noise, and
% the initial state and covariance are X0 and P0, respectively.
%
% Example: (1D point mass, unit time, no forcing)
%
%  >> dkfilt = dfKalman([1 1;0 1], [], [1 0], ...
%                       zeros(2), .1, [0 1]', [0.5 0;0 0.5]);
%
%  ... process loop ...
%
%  >> xpred = dkfilt.predict();
%
%  ...
%
%  >> [y, x] = dkfilter.correct(ymeasured);
%
%  ...
%
%
%  NOTE:
%    - Can initialize with y0 and will reconstruct x0 using pinv(C).
%
%
%  Copyright 2007.
%
%================================ dfKalman ===============================

%
% @file     dfKalman.m
%
% @author   Jimi Malcolm,		(Created)
% @author   Patricio A. Vela.	(Modified)      pvela@gatech.edu
%
% @date     2016/05/23 (conversion to class)
% @date     2006/XX/XX (created)
%
%================================ dfKalman ===============================
classdef dfKalman < dfilter

%============================ Member Variables ===========================
%
%(

properties
  A;        %< State transition matrix.
  B;        %< Input matrix.

  Q;        %< Process noise covariance matrix.
  R;        %< Measurement noise covariance matrix.

  P_hat;    %< The estimated covariance matrix.

  K;        %< Last used Kalman gain
  dx;       %< Last used correction.
end


%)
%
%============================ Member Functions ===========================
%
%(
methods

  %============================== dfKalman =============================
  %
  % @brief  Constructor for dfKalman class.
  %
  % @param[in]  A   The state transition matrix.
  % @param[in]  B   The input matrix.
  % @param[in]  C   The measurement matrix.
  % @param[in]  Q   The process noise.
  % @param[in]  R   The measurement noise.
  % @param[in]  x   The initial state (optional).
  % @param[in]  P   The initial covariance matrix (optional).
  % @param[in]  fparms  Additional options to fix.
  %
  function this = dfKalman(A, B, C, Q, R, x, P, fparms)

  %!== Initialize object and state + covariance.
  if (nargin < 5)
    x = [];
  end

  this@dfilter(C, x);

  if ( (nargin < 7) || isempty(P) )
    this.P_hat = [];
    this.isInit = false;
  else
    this.P_hat = P;
  end

  %!== Set system dynamics
  this.A = A;
  this.B = B;

  %!== Set system uncertainty.
  this.Q = Q;
  this.R = R;

  %!== Modify display function and any other parameters.
  if ( (nargin == 8) && ~isempty(fparms) && isfield(fparms, 'displayFunc') )
    this.displayFunc = fparms.displayFunc;
  end

  end

  %------------------------------- set ------------------------------
  %
  % @brief  Set one of the parameters of the filter.  Allows
  %         post-construction modification of Kalman filter parameters.
  %
  % @param[in]  fname   Field name.
  % @param[in]  fval    Field value.
  %
  function set(this,fname, fval)

  switch (fname)
    case 'displayFunc',
      this.displayFunc = fval;
    case 'A'
      this.A = fval;
    case 'B'
      this.B = fval;
    case 'C'
      this.C = fval;
    case 'Q'
      this.Q = fval;
    case 'R'
      this.R = fval;
    case 'P'
      this.P = fval;
  end

  end

  %
  %------------------------------- get ------------------------------
  %
  % @brief  Get one of the parameters / fields of the filter.
  %
  % @param[in]  fname   Field name.
  % @param[out]  fval    Field value.
  %
  function fval = get(this, fname)

  switch (fname)
    case 'A',
      fval = this.A;
    case 'B',
      fval = this.B;
    case 'Q',
      fval = this.Q;
    case 'R',
      fval = this.R;
    otherwise
      fval = this.get@dfilter(fname);
  end

  end

  %
  %------------------------------- predict -------------------------------
  %
  % @brief  Propogate forward in time the filter state.
  %
  % @param[in]  u       External/control input.
  % @param[out] xpred   Predicted state (prior).
  % @param[out] Ppred   Predicted covariance (prior).
  %
  function [xpred, Ppred] = predict(this, u)

  if (~this.isInit)
    return;
  end

  if (nargin == 1)
    this.x_hat = this.A * this.x_hat;
  else
    x_hat = this.A * this.x_hat + B * this.u;
  end
  this.P_hat = this.A * this.P_hat * transpose(this.A) + this.Q;

  if (nargout == 2)
    xpred = this.x_hat;
    Ppred = this.P_hat;
  elseif (nargout == 1)
    xpred = this.x_hat;
  end

  end

  %
  %------------------------------ correct ------------------------------
  %
  % @brief  Given a measurement, apply correction to the predicted state.
  %
  % Runs the correction step, unless the filter has not been initialized.
  % In which case, the corrected (posterior) outputs are empty / not defined.
  % Otherwise, corrected (posterior) outputs are available.
  %
  % @param[in]  y       Current measurement.
  % @param[out] x       Corrected state (posterior).
  % @param[out] Ppred   Corrected covariance (posterior).
  % @param[out] y       Corrected measurement (posterior).
  %
  function [x, P, y] = correct(this, y)

  if (~this.isInit)
    x = [];
    P = [];
    y = [];
    return;
  end

  %-- Compute the Kalman gain.
  K = (this.P_hat * transpose(this.C)) ...
        * inv( this.C*this.P_hat*transpose(this.C) + this.R );
  this.K = K;

  %-- Apply the correction step.
  this.dx    = K* (y - this.C * this.x_hat);
  this.x_hat = this.x_hat + this.dx;
  this.P_hat = (eye(size(this.P_hat)) - K * this.C) * this.P_hat;

  %-- Generate measurement if needed.
  y = this.C * this.x_hat;

  if (nargout >= 1)
    x = this.x_hat;
  end

  if (nargout >= 2)
    P = this.P_hat;
  end

  end

  %
  %---------------------------- getCovariance ----------------------------
  %
  % @brief  Get the current covariance estimate.
  %
  % @param[out] P   Current estimated covariance.
  %
  function P = getCovariance(this)
  P = this.P_hat;
  end

  %
  %---------------------------- setCovariance ----------------------------
  %
  % @brief  Set the current covariance estimate.
  %
  % @param[out] P   Covariance matrix.
  %
  function setCovariance(this, P)
  this.P_hat = P;
  end

  %---------------------------- setFilterState ---------------------------
  %
  % @brief  Set the current estimate of the state vector and state
  %         covariance.
  %
  % @param[in]  X0  State vector.
  % @param[in]  P0  Covariance matrix.
  %
  function setFilterState(this, X0, P0)

  this.setState(X0);
  this.setCovariance(P0);
  this.isInit = true;

  end

  %================================ reset ==============================
  %
  % @brief  Reset the current filter state (not initialized).
  %
  function reset(this)

  this.P_hat = [];
  this.reset@dfilter();

  end

  %=========================== getFilterState ==========================
  %
  % @brief  Get the current estimate of the state vector and state
  %         covariance.
  %
  % @param[out] X   State vector.
  % @param[out] P   Covariance matrix.
  %
  function [X, P] = getFilterState(this)

  X = this.getState();
  P = this.getCovariance();

  end

end
%
%)

end
%
%================================ dfKalman ===============================


