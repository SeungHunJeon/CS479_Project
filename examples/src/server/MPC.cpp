//
// Created by user on 22. 10. 31..
//
#include "MPC.hpp"
#include "unsupported/Eigen/MatrixFunctions"

/// tuning variable
size_t N = 8;
double dt = 0.01;

size_t state_dim = 7;
size_t con_dim = 2;

QP *myQP;

void Sparse_Matrix::get_Sparse_P(Eigen::MatrixXd P, Eigen::MatrixXd R) {
  std::vector<int> outer;
  std::vector<int> inner;
  std::vector<double> value;

  outer.push_back(0);

  Eigen::MatrixXd tmp;
  tmp.setZero(state_dim+con_dim, state_dim+con_dim);
  tmp.block(0,0, con_dim, con_dim) << R;
  tmp.block(con_dim, con_dim, state_dim, state_dim) << P;

  Eigen::SparseMatrix<double> tmp_sparse = tmp.sparseView();
  tmp_sparse.makeCompressed();

  std::vector<int> tmp_outer, tmp_outer_;
  std::vector<int> tmp_inner, tmp_inner_;
  std::vector<double> tmp_value;

  /// Distills the elements of each sparse matrices
  for (int i=1; i<tmp_sparse.outerSize()+1; i++)
    tmp_outer.push_back(tmp_sparse.outerIndexPtr()[i]);

  for (int i=0; i<tmp_sparse.size() - tmp_sparse.outerSize() + 1; i++) {
    tmp_value.push_back(tmp_sparse.valuePtr()[i]);
    tmp_inner.push_back(tmp_sparse.innerIndexPtr()[i]);
  }

  tmp_outer_ = tmp_outer;
  tmp_inner_ = tmp_inner;

  /// Concatenate the sub-sparse matrices into Uni-Big sparse matrix
  // For Outer Index
  for (int j=0; j<N; j++) {
    for (int i=0; i<tmp_outer.size(); i++)
      tmp_outer[i] = tmp_outer_[i] + outer.back();
    outer.insert(end(outer), begin(tmp_outer), end(tmp_outer));
  }

  // For Inner Index
  for (int j=0; j<N; j++) {
    for (int i=0; i<tmp_inner.size(); i++)
      tmp_inner[i] = tmp_inner_[i] + j*(state_dim + con_dim);
    inner.insert(end(inner), begin(tmp_inner), end(tmp_inner));
  }

  // For Value
  for (int j=0; j<N; j++) {
    value.insert(end(value), begin(tmp_value), end(tmp_value));
  }

  Pjc = outer;
  Pir = inner;
  Ppr = value;

  /// TODO
  /// How to return the outer, inner, value vectors? separate each functions & return each value?
  /// such as class manner
}

void Sparse_Matrix::get_Sparse_A (Eigen::MatrixXd model_A, Eigen::MatrixXd model_B) {
  std::vector<int> outer;
  std::vector<int> inner;
  std::vector<double> value;

  outer.push_back(0);

  Eigen::MatrixXd tmp, Tmp;
  tmp.setZero(state_dim*2, state_dim+con_dim);
  tmp.block(0, 0, state_dim, con_dim) << model_B;
  tmp.block(state_dim, con_dim, state_dim, state_dim) << model_A;
  tmp.block(0, con_dim, state_dim, state_dim) << -Eigen::MatrixXd::Identity(state_dim, state_dim);

  Tmp.setZero(state_dim, state_dim+con_dim);
  Tmp = tmp.block(0, 0, state_dim, state_dim + con_dim);

  Eigen::SparseMatrix<double> tmp_sparse = tmp.sparseView();
  Eigen::SparseMatrix<double> Tmp_sparse = Tmp.sparseView();
  tmp_sparse.makeCompressed();
  Tmp_sparse.makeCompressed();
  std::vector<int> tmp_outer, tmp_outer_, Tmp_outer, Tmp_outer_;
  std::vector<int> tmp_inner, tmp_inner_, Tmp_inner, Tmp_inner_;
  std::vector<double> tmp_value, Tmp_value;

  /// Distills the elements of each sparse matrices
  for (int i=1; i<tmp_sparse.outerSize()+1; i++)
    tmp_outer.push_back(tmp_sparse.outerIndexPtr()[i]);

  for (int i=0; i<tmp_sparse.size() - tmp_sparse.outerSize() + 1; i++) {
    tmp_value.push_back(tmp_sparse.valuePtr()[i]);
    tmp_inner.push_back(tmp_sparse.innerIndexPtr()[i]);
  }

  for (int i=1; i<Tmp_sparse.outerSize()+1; i++)
    Tmp_outer.push_back(Tmp_sparse.outerIndexPtr()[i]);

  for (int i=0; i<Tmp_sparse.size() - Tmp_sparse.outerSize() + 1; i++) {
    Tmp_value.push_back(Tmp_sparse.valuePtr()[i]);
    Tmp_inner.push_back(Tmp_sparse.innerIndexPtr()[i]);
  }

  tmp_outer_ = tmp_outer;
  Tmp_outer_ = Tmp_outer;
  tmp_inner_ = tmp_inner;
  Tmp_inner_ = Tmp_inner;

  /// Concatenate the sub-sparse matrices into Uni-Big sparse matrix
  // For outer
  for (int j=0; j<N; j++) {
    if (j == N-1) {
      for (int i=0; i<Tmp_outer.size(); i++)
        Tmp_outer[i] = Tmp_outer_[i] + outer.back();
      outer.insert(end(outer), begin(Tmp_outer), end(Tmp_outer));
      break;
    }

    for (int i=0; i<tmp_outer.size(); i++)
      tmp_outer[i] = tmp_outer_[i] + outer.back();
    outer.insert(end(outer), begin(tmp_outer), end(tmp_outer));
  }

  // For inner
  for (int j=0; j<N; j++) {
    if (j == N-1) {
      for (int i=0; i<Tmp_inner.size(); i++)
        Tmp_inner[i] = Tmp_inner_[i] + j*state_dim;
      inner.insert(end(inner), begin(Tmp_inner), end(Tmp_inner));
      break;
    }

    for (int i=0; i<tmp_inner.size(); i++)
      tmp_inner[i] = tmp_inner_[i] + j*state_dim;
    inner.insert(end(inner), begin(tmp_inner), end(tmp_inner));
  }

  // For Value
  for (int j=0; j<N; j++) {
    if(j == N-1) {
      value.insert(end(value), begin(Tmp_value), end(Tmp_value));
      break;
    }
    value.insert(end(value), begin(tmp_value), end(tmp_value));
  }

  Ajc = outer;
  Air = inner;
  Apr = value;

}

void Sparse_Matrix::get_Sparse_G () {
  std::vector<int> outer;
  std::vector<int> inner;
  std::vector<double> value;

  outer.push_back(0);

  Eigen::MatrixXd tmp;
  tmp.setZero(con_dim*2, con_dim+state_dim);
  tmp.block(0, 0, con_dim, con_dim) << Eigen::MatrixXd::Identity(con_dim, con_dim);
  tmp.block(con_dim, 0, con_dim, con_dim) << -Eigen::MatrixXd::Identity(con_dim, con_dim);

  Eigen::SparseMatrix<double> tmp_sparse = tmp.sparseView();
  tmp_sparse.makeCompressed();

  std::vector<int> tmp_outer, tmp_outer_;
  std::vector<int> tmp_inner, tmp_inner_;
  std::vector<double> tmp_value;

  /// Distills the elements of each sparse matrices
  for (int i=1; i<tmp_sparse.outerSize()+1; i++)
    tmp_outer.push_back(tmp_sparse.outerIndexPtr()[i]);

  for (int i=0; i<tmp_sparse.size() - tmp_sparse.outerSize() + 1; i++) {
    tmp_value.push_back(tmp_sparse.valuePtr()[i]);
    tmp_inner.push_back(tmp_sparse.innerIndexPtr()[i]);
  }

  tmp_outer_ = tmp_outer;
  tmp_inner_ = tmp_inner;

  // For outer
  for (int j=0; j<N; j++) {
    for (int i=0; i<tmp_outer.size(); i++)
      tmp_outer[i] = tmp_outer_[i] + outer.back();
    outer.insert(end(outer), begin(tmp_outer), end(tmp_outer));
  }

  // For inner
  for (int j=0; j<N; j++) {
    for (int i=0; i<tmp_inner.size(); i++)
      tmp_inner[i] = tmp_inner_[i] + j*state_dim;
    inner.insert(end(inner), begin(tmp_inner), end(tmp_inner));
  }

  // For Value
  for (int j=0; j<N; j++) {
    value.insert(end(value), begin(tmp_value), end(tmp_value));
  }

  Gjc = outer;
  Gir = inner;
  Gpr = value;

}

void MPC::init() {
  /// Discretize the model dynamics using ZOH manner
  Eigen::MatrixXd A_tmp = model_A;
  model_A = (model_A*dt).exp();
  model_B = A_tmp.inverse()*(model_A - Eigen::MatrixXd::Identity(A.rows(), A.cols()))*model_B;

  /// Initialize the weighting Matrix
  P_con.setZero(state_dim*(N+1), state_dim*(N+1));
  R_con.setZero(con_dim*N, con_dim*N);
  Q_con.setZero(P_con.rows()+R_con.rows(), P_con.rows()+R_con.rows());

  for (int i = 0; i< N+1; i++) {
    P_con.block(state_dim*i, state_dim*i, state_dim, state_dim) << P;
    if (i == N)
    {
      continue;
    }
    R_con.block(con_dim*i, con_dim*i, con_dim, con_dim) << R;
  }

  Q_con.block(0, 0, state_dim*(N+1), state_dim*(N+1)) << P_con;
  Q_con.block(state_dim*(N+1), state_dim*(N+1), con_dim*N, con_dim*N) << R_con;

  var.setZero(state_dim*(N+1) + con_dim *N);
  result.setZero(state_dim + con_dim);

}



void MPC::update() {

  /// Using result

  /// B update

  /// var update

}

Eigen::VectorXd MPC::Solve(Eigen::VectorXd Var) {















  return result;
}