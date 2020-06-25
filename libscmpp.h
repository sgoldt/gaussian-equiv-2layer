/**
 * File:    libscmpp.cpp
 *
 * Utility functions for two-layer neural networks.
 */

#ifndef LIBSCMPP
#define LIBSCMPP

#include <cmath>
#include <getopt.h>
#include <stdexcept>
#include <string.h>
#include <unistd.h>

// #define ARMA_NO_DEBUG
#include <armadillo>
#include <chrono>
using namespace arma;

// MATHEMATICAL CONSTANTS
const double ONE_OVER_SQRT2 = 1.0/datum::sqrt2;
const double SQRT_2_OVER_PI = datum::sqrt2/sqrt(datum::pi);

// codes for activation functions
const int LINEAR = 0;
const int ERF = 1;
const int RELU = 2;
const int SIGN = 3;
const int QUAD = 4;
const int ABS = 5;

// weight initialisations
const int INIT_LARGE = 1;
const int INIT_SMALL = 2;
const int INIT_INFORMED = 3;
const int INIT_DENOISE = 4;
const int INIT_MIXED = 5;
const int INIT_MIXED_NORMALISE = 6;
const int INIT_NATI = 7;
const int INIT_NATI_MF = 8;

vec v_empty = vec();

/**
 * A wrapper around Armadillos trapz function that performs numerical
 * integration for all the vectors in the given cube w.r.t. the given x.
 *
 * Parameters:
 * -----------
 * x : vec
 *     the spacing of x with respect to which the integral is computed
 * f : cube(r, c, s)
 *     cube where the integral is done over the last dimension
 *
 * Returns:
 * --------
 * result : mat (r, c)
 *     result of the trapezoidal integrals along z
 */
mat trapz(vec& x, cube& cu) {
  mat result = zeros(cu.n_rows, cu.n_cols);

  for (int i = 0; i < cu.n_rows; i++) {
    for (int k = 0; k < cu.n_cols; k++) {
      result(i, k) = as_scalar(trapz(x, (vec) cu.tube(i, k)));
    }
  }

  return result;
}

mat trapz(vec& x, cube& cu, vec& weight) {
  mat result = zeros(cu.n_rows, cu.n_cols);

  for (int i = 0; i < cu.n_rows; i++) {
    for (int k = 0; k < cu.n_cols; k++) {
      vec y = weight % ((vec) cu.tube(i, k));
      result(i, k) = as_scalar(trapz(x, y));
    }
  }

  return result;
}

/**
 * Returns the moving average of the given vector x with window-size w.
 */
vec moving_average(vec& x, int w) {
  vec window = vec(w, fill::ones);
  vec avg = 1. / w * conv(x, window, "same");
  return avg.head(x.n_elem - 1);
}


/**
 * Returns a random rotation matrix, drawn from the Haar distribution
 * (the only uniform distribution on SO(n)).
 *
 * The algorithm is described in the paper Stewart, G.W., "The efficient
 * generation of random orthogonal matrices with an application to condition
 * estimators", SIAM Journal on Numerical Analysis, 17(3), pp. 403-409, 1980.
 * For more information see
 * https://en.wikipedia.org/wiki/Orthogonal_matrix#Randomization
 *
 * This implementation is a translation from the SciPy code, to be found at
 * https://github.com/scipy/scipy/blob/v1.2.1/scipy/stats/_multivariate.py,
 * which in turn is a wrapping of wrapping the random_rot code from the MDP
 * Toolkit, https://github.com/mdp-toolkit/mdp-toolkit
 */
mat special_ortho_group(int N) {
  mat H = eye<mat>(N, N);
  vec D = ones<vec>(N, 1);  // vec = colvec, i.e. dense matrix with one column
  for (int n = 1; n < N; n++) {
    vec x = randn<vec>(N-n+1);
    D(n-1) = x(0) >= 0 ? 1 : -1;
    x(0) -= D(n-1) * as_scalar(sqrt(sum(pow(x, 2))));
    // Householder transformation
    mat Hx = eye<mat>(N-n+1, N-n+1) - 2.* (x * x.t()) / as_scalar(sum(pow(x, 2)));
    mat asdf = eye<mat>(N, N);
    asdf.submat(n-1, n-1, N-1, N-1) = Hx;
    H = H * asdf;
  }
  // fix the last sign s.t. the determinant is 1
  D(D.n_elem - 1) = pow(-1, 1 - (N % 2)) * prod(D);
  H = diagmat(D) * H;

  return H;
}

/**
 * Prints a status update with the generalisation error and elements of Q and
 * R.
 *
 * Parameters:
 * -----------
 * t :
 *     time
 * eg : scalar
 *     generalisation error
 * et : scalar
 *     training error
 * eg_frac : scalar
 *     fractional generalisation error
 * et_frac : scalar
 *     fractional testing error
 * diff : scalar
 *     mean absolute change in weights
 * Q, R, T:
 *     order parameters
 * quiet : bool
 *     if True, output reduced information
 */
std::string status(double t, double eg, double et, double eg_frac,
                   double et_frac, double diff,
                   mat& Q, mat& R, mat&T, vec& A, vec& v, bool quiet=false) {
  std::ostringstream msg;

  msg << t << ", " << eg << ", " << et << ", " << eg_frac << ", "
      << et_frac << ", " << diff << ", ";

  if (!quiet) {
    int M = R.n_cols;
    int K = Q.n_rows;

    // print elements of Q
    for (int k = 0; k < K; k++) {
      for (int l = k; l < K; l++) {
        msg << Q(k, l) << ", ";
      }
    }
    if (!R.is_empty()) {
      // print elements of R
      for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
          msg << R(k, m) << ", ";
        }
      }
    }

    if(!A.is_empty()) {
      //print elements of A
      for (int n = 0; n < M; n++) {
        msg << A(n) << ", ";
      }
    }
    for (int i = 0; i < K; i++) {
      msg << v(i) << ", ";
    }
  }

  std::string msg_str = msg.str();
  return msg_str.substr(0, msg_str.length() - 2);
}


const char* activation_name(int g) {
  switch (g) {
    case LINEAR:
      return "lin";
    case ERF:
      return "erf";
    case RELU:
      return "relu";
    case SIGN:
      return "sgn";
    case QUAD:
      return "quad";
    case ABS:
      return "abs";
    default:
      return "";
  }
}


/**
 * Returns the fraction of entries where these two arrays do not have the same
 * entry.
 *
 * Parameters:
 * -----------
 * mat a, b : 
 *     the two matrices, which each must be of the size (N, 1)
 */
double frac_error(mat& a, mat& b) {
  if (a.n_elem != b.n_elem) {
    throw std::invalid_argument("vectors a and b must have the same length.");
  }
     
  bool polar = (a.min() < 0) || (b.min() < 0);

  a /= a.max();
  b /= b.max();

  if (polar) {  // ie encoded with -1, 1
    return 1 - 0.5 * mean(mean((a % b) + 1));  // % is the element-wise product
  } else {  // ie encoded with 0, const
    return mean(mean(abs(a - b)));
  }
}


/**
 * Element-wise application of g(x) to the matrix x.
 *
 * @param  x  some matrix
 */
mat g_lin(mat& x) {
  return x;
}


/**
 * Element-wise application of the derivative of Erf(x/sqrt(2)) to the matrix x
 *
 * @param  x  some matrix
 */
mat dgdx_lin(mat& x) {
  return ones(size(x));
}

/**
 * Element-wise application of g(x)=x^2 to the matrix x.
 *
 * @param  x  some matrix
 */
mat g_quad(mat& x) {
  return pow(x, 2);
}

/**
 * Element-wise application of g(x)=x^2 to the matrix x.
 *
 * @param  x  some matrix
 */
mat dgdx_quad(mat& x) {
  return 2 * x;
}

/**
 * Element-wise application of g(x)=x^2 to the matrix x.
 *
 * @param  x  some matrix
 */
mat g_abs(mat& x) {
  return abs(x);
}

/**
 * Element-wise application of erf(x/sqrt(2)) to the matrix x.
 *
 * @param  x  some matrix
 */
mat g_erf(mat& x) {
  return erf(ONE_OVER_SQRT2 * x);
}

/**
 * Element-wise application of the derivative of Erf(x/sqrt(2)) to the matrix x
 *
 * @param  x  some matrix
 */
mat dgdx_erf(mat& x) {
  return SQRT_2_OVER_PI*exp(-0.5*pow(x, 2));
}

/**
 * Rectified linear unit activation function.
 *
 * @param  x  some matrix
 */
mat g_relu(mat& x) {
  double max = x.max();
  return max > 0 ? clamp(x, 0, max) : x.fill(0);
}

/**
 * Derivative of the rectified linear unit.
 *
 * @param  x  some matrix
 */
mat dgdx_relu(mat& x) {
  mat dgdx = zeros<mat>(size(x));
  dgdx.elem(find(x > 0)).ones();
  return dgdx;
}

/**
 * Rectified linear unit activation function.
 *
 * @param  x  some matrix
 */
mat g_sign(mat& x) {
  return sign(x);
}


/**
 * Computes the generalisation error for the given overlap and self-overlap
 * matrices.
 * 
 * Parameters
 * ----------
 * Q, T: mat (K, K), mat (M, M)
 *     student's and teacher's self-overlap, resp.
 * R: mat (K, M)
 *     student-teacher overlap matrix
 * A, v: second-layer weights of the teacher and student, resp.
 * g1, g2:
 *     teacher's and student's activation function, resp.
 */
double eg_analytical(mat& Q, mat& R, mat& T, vec& A, vec& v,
                     mat (*g1)(mat&), mat (*g2)(mat&)) {
  double epsilon = 0;

  if (g1 == g2 and g1 == g_erf) {
    // student-student overlaps
    vec sqrtQ = sqrt(1 + Q.diag());
    mat normalisation = sqrtQ * sqrtQ.t();
    epsilon += 1. / M_PI * as_scalar(accu(v * v.t() % asin(Q / normalisation)));
  
    // teacher-teacher overlaps
    vec sqrtT = sqrt(1 + T.diag());
    normalisation = sqrtT * sqrtT.t();
    epsilon += 1. / M_PI * as_scalar(accu(A * A.t() % asin(T / normalisation)));

    // student-teacher overlaps
    normalisation = sqrtQ * sqrtT.t();
    epsilon -= 2. / M_PI * as_scalar(accu(v * A.t() % asin(R / normalisation)));
  } else if (g1 == g2 && g1 == g_lin) {
    epsilon = 0.5 * (accu(v * v.t() % Q) + accu(A * A.t() % T)
                     - 2. * accu(v * A.t() % R));
  }

  return epsilon;
}


/**
 * Computes the generalisation error for the given teacher and student network.
 * 
 * Parameters
 * ----------
 * B: mat (M, N)
 *     teacher's weight matrix
 * A : vec (M)
 *     hidden unit-to-output weights of the teacher
 * w : (K, N)
 *     input-to-hidden unit weights of the student
 * v : (K)
 *     hidden unit-to-output weights of the student
 * g1, g2:
 *     teacher's and student's activation function, resp.
 */
double eg_analytical(mat& B, vec& A, mat& w, vec& v,
                     mat (*g1)(mat&), mat (*g2)(mat&)) {
  const int N = B.n_cols;

  mat Q = w * w.t() / N;
  mat R = w * B.t() / N;
  mat T = B * B.t() / N;

  return eg_analytical(Q, R, T, A, v, g1, g2);
}

/**
 * Computes the output of the neural network with the given weights.
 *
 * Parameters:
 * ----------
 * w :
 *     input-to-hidden unit weights of the scm
 * v :
 *     hidden unit-to-output weights of the scm
 * xs : (bs, N)
 *     SCM inputs
 * g :
 *     activation function
 */
mat phi(mat& w, vec& v, mat& xs, mat (*g)(mat&)) {
  mat act = xs * w.t() / sqrt(w.n_cols);  // activation of the hidden units
  mat hidden = (*g)(act);  // apply the non-linearity point-wise
  return hidden * v;  // and sum up!    
}

/**
 * Computes the output of the neural network with the given weights.
 *
 * Parameters:
 * ----------
 * w :
 *     input-to-hidden unit weights of the scm
 * xs : (bs, N)
 *     SCM inputs
 * g :
 *     activation function
 */
mat phi(mat& w, mat& xs, mat (*g)(mat&)) {
  mat act = xs * w.t() / sqrt(w.n_cols);  // activation of the hidden units
  mat hidden = (*g)(act);  // apply the non-linearity point-wise
  return sum(hidden, 1); // and sum up !
}

/**
 * Numerically computes the mse of an SCM with the given weights as half the
 * squared difference between the SCM's output and the given ys.
 *
 * w : (K, N)
 *     input-to-hidden unit weights of the scm
 * v : (K)
 *     hidden unit-to-output weights of the scm
 * xs : (bs, N)
 *     input matrix of size (bs, N) on which the mse is evaluated.
 * ys : (bs, 1)
 *     the "true" outputs for the given xs.
 * g :
 *     activation function of the SCM to use.
 */
double mse_numerical(mat& w, vec& v, mat& xs, mat& ys, mat (*g)(mat&)) {
  return 0.5 * as_scalar(mean(pow(ys - phi(w, v, xs, g), 2)));
}


/**
 * Numerically computes the mse between the scalars y1 and y2.
 */
double mse_numerical(mat& y1, mat& y2) {
  if (!(size(y1) == size(y2))) {
    throw std::invalid_argument("matrices y1 and y2 must have the same size.");
  }
  return 0.5 * as_scalar(mean(pow(y1 - y2, 2)));
}


/**
 * Classifies the given outputs of an SCM.
 *
 * Parameters:
 * ----------
 * ys : (bs, 1)
 *     SCM outputs
 *
 * returns:
 * --------
 * classes (bs, 1)
 *     class labels in polar encoding \pm 1
 */
mat classify(mat& ys) {
  mat classes = sign(ys);
  classes.replace(0, 1);
  return classes;
}


/**
 * Classifies the output of the neural network with the given weights.
 *
 * Parameters:
 * ----------
 * w : (K, N)
 *     input-to-hidden layer weights of the scm
 * v : (K)
 *     hidden-to-output layer weights 
 * xs : (bs, N)
 *     SCM inputs
 * g :
 *     activation function
 *
 * Returns:
 * --------
 * classes (mat)
 *     class labels in polar encoding \pm 1
 */
mat classify(mat& w, vec& v, mat& xs, mat (*g)(mat&)) {
  mat phis = phi(w, v, xs, g);
  return classify(phis);
}

/**
 * Classifies the output of the neural network with the given weights.
 *
 * Parameters:
 * ----------
 * w : (K, N)
 *     input-to-hidden layer weights of the scm
 * xs : (bs, N)
 *     SCM inputs
 * g :
 *     activation function
 * boundary :
 *     decision boundary that separates the two classes if the output function
 *     is ReLU.
 *
 * Returns:
 * --------
 * classes (mat)
 *     binary (0, 1) if the given activation is ReLU, else \pm 1
 */
mat classify(mat& w, mat& xs, mat (*g)(mat&)) {
  mat phis = phi(w, xs, g);
  return classify(phis);
}

/**
 * Computes the weight increment of the input-to-hidden unit weights for
 * gradient descent on the mean squared error.
 *
 * Parameters:
 * -----------
 * w : (K, N)
 *     input-to-hidden unit weights of the scm
 * v : (K)
 *     hidden unit-to-output weights of the scm
 * scale : double
 *     scalar that rescales the output(s) of the network; useful when using
 *     dropout, for example.
 */
void update_gradients(mat& gradw, vec& gradv, mat& w, vec& v, mat& xs, mat& ys,
                      mat(*g)(mat&), mat(*dgdx)(mat&),
                      bool both, double scale=1) {
  const int bs = xs.n_rows;  // mini-batch size

  // forward pass
  mat act = w * xs.t() / sqrt(w.n_cols);  // (K, bs) activation of the hidden units
  mat hidden = (*g)(act);  // (K, bs) apply the non-linearity point-wise
  mat ys_pred = v.t() * hidden;  // (1, bs) and sum up!

  // backward pass
  vec error = ys - scale * ys_pred.t();  // (bs, 1)
  mat deriv = dgdx(act);  // (K, bs)
  deriv.each_col() %= v;

  if (both) {
    gradv = 1. / bs * g(act) * error;
  }

  gradw = 1. / bs * deriv * diagmat(error) * xs;
}

/**
 * Computes the weight increment of the input-to-hidden unit weights for
 * gradient descent on the mean squared error.
 *
 * Parameters:
 * -----------
 * w : (K, N)
 *     input-to-hidden unit weights of the scm
 * scale : double
 *     scalar that rescales the output(s) of the network; useful when using
 *     dropout, for example.
 */
void update_gradient(mat& gradw, mat& w, mat& xs, mat& ys,
                     mat(*g)(mat&), mat(*dgdx)(mat&), double scale=1) {
  const int N = w.n_cols;
  const int bs = xs.n_rows;  // mini-batch size

  mat act = w * xs.t() / sqrt(N);
  vec error = ys - scale * phi(w, xs, g);
  gradw = 1. / bs * dgdx(act) * diagmat(error) * xs;
}


/**
 * Sets the teacher weights
 *
 * Returns:
 * --------
 * true if initialisation was successful, false in case of an error.
 */
bool init_teacher_randomly(mat& B0, vec& A0, int N, int M, double uniform,
                           bool both, bool normalise=false, bool meanfield=0,
                           int mix = 0, double sparse=0) {
  B0 = randn<mat>(M, N);   // teacher input-to-hidden weights
  A0 = vec(M, fill::ones);  // teacher hidden-to-output weights
  if (abs(uniform) > 0) {
    A0 *= uniform;
  } else if (both) {
    A0 = vec(M, fill::randn);
  }
  if (normalise) {
    A0 /= M;
  } else if (meanfield) {
    A0 /= sqrt(M);
  }
  if (sparse > 0) {
    // hide a fraction sparse of first-layer teacher weights
    if (sparse > 1) {
      cerr << "Cannot have sparse > 1. Will exit now " << endl;
      return false;
    }
    mat mask = randu<mat>(size(B0));
    mask.elem(find(mask > sparse)).ones();
    mask.elem(find(mask < sparse)).zeros();
    B0 %= mask;
  }

  if (mix) {
    // flip the sign of half of the teacher's second-layer weights
    vec mask = ones<vec>(M);
    mask.head(round(M/2.)) *= -1;
    A0 %= mask;
  }

  return true;
}

/**
 * Randomly initialises the student weights.
 */
void init_student_randomly(mat& w, vec& v, int N, int K, int init,
                           double uniform, bool both, bool normalise,
                           bool meanfield, bool mix) {
  // INIT_LARGE:
  double prefactor_w = 1;
  double prefactor_v = 1;
  if (init == INIT_SMALL) {
    prefactor_w = 1e-3;
    prefactor_v = 1e-3;
  } else if (init == INIT_MIXED) {
    prefactor_w = 1. / sqrt(N);
    prefactor_v = 1. / sqrt(K);
  } else if (init == INIT_MIXED_NORMALISE) {
    prefactor_w = 1. / N;
    prefactor_v = 1. / K;
  }
  
  w = prefactor_w * randn<mat>(K, N);
  if (both) {
    v = prefactor_v * randn<vec>(K);
  } else {
    v = vec(K, fill::ones);
    if (normalise) {
      v.fill(1. / K);
    } else if (meanfield) {
      if (abs(uniform) > 0) {
        v.fill(uniform / sqrt(K));
      } else {
        v.fill(1. / sqrt(K));
      }
    } else if (abs(uniform) > 0) {
      v.fill(uniform);
      
      if (mix) {
        // flip the sign of half of the student's second-layer weights, too
        vec mask = ones<vec>(K);
        mask.head(round(K/2.)) *= -1;
        v %= mask;
      }
    }
  }
}


#endif
