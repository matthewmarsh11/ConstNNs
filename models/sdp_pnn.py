import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Union
from base import *
import cvxpy as cp
from scipy.stats import chi2
import numpy as np
# from cvxpylayers.torch import CvxpyLayer
import scipy

class SDP_PNN(BaseModel):
    """
    Probabilistic Neural Network with integrated SDP solver
    to project and enforce linear equality constraints

    Args:
        config: int, Configuration object of network specs
        input_dim: int, Dimension of input tensor
        output_dim: int, Dimension of output tensor
        A: torch.tensor: Constraint matrix: of shape (m, * input_dim) - m is the number of constraints
        B: Constraint matrix: of shape (m, * output_dim) - m is the number of constraints
        b: Constraint vector: of shape (m, 1) - m is the number of constraints
    """
    
    def __init__(self, config: MLPConfig, input_dim: int, output_dim: int,
                    A: torch.Tensor, B: torch.Tensor, b: torch.Tensor,
                    epsilon: Union[torch.Tensor, float],
                    probability_level: float = 0.95,
                    reg_term: float = 1e-5):
        super().__init__(config)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.probability_level = probability_level
        self.reg_term = reg_term  # Added regularization term
        
        self.chi2_thresh = chi2.ppf(self.probability_level, df=self.output_dim)
        
        self.register_buffer("A", A.unsqueeze(0) if A.shape[0] == input_dim else A)
        self.register_buffer("B", B.unsqueeze(0) if B.shape[0] == output_dim else B)
        self.register_buffer("b", b)
        
        self.num_constraints = self.A.shape[0]
        
        # Check if epsilon is a scalar
        if isinstance(self.epsilon, (int, float)):
            self.epsilon = torch.eye(self.num_constraints, device=config.device) * self.epsilon
        elif not torch.all(self.epsilon == torch.diag(torch.diagonal(self.epsilon))):
            raise ValueError("Epsilon must be either a scalar or a diagonal matrix")
        
        self.activation = getattr(nn, config.activation)()
        
        # Build layers
        layers = []
        current_dim = self.input_dim
        
        # Input layer
        for _ in range(config.num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, config.hidden_dim),
                self.activation
            ])
            
            layers.append(nn.Dropout(p=config.dropout))
        
            current_dim = config.hidden_dim
        
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(current_dim, self.output_dim)
        
        self.cholesky_size = self.output_dim * (self.output_dim + 1) // 2
        self.fc_cholesky = nn.Linear(current_dim, self.cholesky_size)
        
        self.build_cvxpy_layer()
        
    def build_cvxpy_layer(self):
        d = self.output_dim
        m = self.num_constraints

        # Declare parameters that WILL be used in the objective
        Sigma_P_inv = cp.Parameter((d, d))
        SigmaPinv_muP = cp.Parameter(d)
        muP_SigmaPinv_muP = cp.Parameter(nonneg=True)
        logdet_Sigma_P = cp.Parameter(nonneg=True)
        x = cp.Parameter(self.input_dim)

        # Decision variables
        mu_Q = cp.Variable(d)
        Sigma_Q = cp.Variable((d, d), PSD=True)

        # KL terms (fully expanded for DPP compliance)
        term1 = cp.trace(Sigma_P_inv @ Sigma_Q)
        term2 = cp.sum_squares(Sigma_P_inv @ mu_Q) - 2 * cp.sum(cp.multiply(SigmaPinv_muP, mu_Q)) + muP_SigmaPinv_muP
        term3 = -cp.log_det(Sigma_Q)

        kl_div = 0.5 * (term1 + term2 - d + logdet_Sigma_P + term3)
        objective = cp.Minimize(kl_div)

        # Constraints — fixed from buffers
        A_param = cp.Constant(self.A.cpu().numpy())
        B_param = cp.Constant(self.B.cpu().numpy())
        b_param = cp.Constant(self.b.squeeze(-1).cpu().numpy())
        eps_param = cp.Constant(self.epsilon.cpu().numpy())
        chi2_thresh = cp.Constant(float(self.chi2_thresh))

        linear_constraint = A_param @ x + B_param @ mu_Q == b_param
        psd_constraint = eps_param - (1.0 / chi2_thresh) * (B_param @ Sigma_Q @ B_param.T) >> 0

        problem = cp.Problem(objective, [linear_constraint, psd_constraint])
        parameters=[
            Sigma_P_inv,
            SigmaPinv_muP,
            muP_SigmaPinv_muP,
            logdet_Sigma_P,
            x
        ]
        # Only include used parameters in the exact order
        self.cvxpy_layer = CvxpyLayer(
            problem,
            parameters=parameters,
            variables=[mu_Q, Sigma_Q],
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        z = self.layers(x)
        mu = self.fc(z)

        # Cholesky output size = number of lower-triangular elements
        tril_indices = torch.tril_indices(row=self.output_dim, col=self.output_dim, offset=0, device=x.device)
        cholesky_raw_elements = self.fc_cholesky(z)  # shape: (batch_size, num_tril_elements)

        L_batch = torch.zeros(batch_size, self.output_dim, self.output_dim, device=x.device)

        for i in range(batch_size):
            # Fill lower triangle
            L = torch.zeros(self.output_dim, self.output_dim, device=x.device)
            L[tril_indices[0], tril_indices[1]] = cholesky_raw_elements[i]

            # Apply softplus + epsilon to diagonal to ensure positive entries
            diag_idx = torch.arange(self.output_dim, device=x.device)
            L[diag_idx, diag_idx] = torch.nn.functional.softplus(L[diag_idx, diag_idx]) + 1e-6
            L_batch[i] = L

        mu_Q, Sigma_Q = self.project_constraints(mu, L_batch, x)

        # Ensure Sigma_Q is positive definite before Cholesky
        Sigma_Q = Sigma_Q + torch.eye(self.output_dim, device=x.device).unsqueeze(0) * self.reg_term
        
        # Use try-except to handle potential numerical issues with Cholesky
        try:
            L_out = torch.linalg.cholesky(Sigma_Q)
        except Exception:
            # Fall back solution if Cholesky fails
            print("Warning: Cholesky decomposition failed, using regularized eigendecomposition")
            L_out = self.stable_cholesky(Sigma_Q)

        return mu_Q, L_out
        
    def stable_cholesky(self, matrix_batch):
        """A more stable Cholesky implementation using eigendecomposition"""
        batch_size = matrix_batch.shape[0]
        dim = matrix_batch.shape[1]
        L_batch = torch.zeros_like(matrix_batch)
        
        for i in range(batch_size):
            matrix = matrix_batch[i]
            # Get eigenvalues and eigenvectors
            eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
            # Ensure eigenvalues are positive
            eigenvalues = torch.clamp(eigenvalues, min=self.reg_term)
            # Reconstruct the matrix as V * sqrt(D) where D is diagonal matrix of eigenvalues
            sqrt_eigenvalues = torch.sqrt(eigenvalues)
            L = eigenvectors @ torch.diag(sqrt_eigenvalues)
            L_batch[i] = L
            
        return L_batch

    def project_constraints(self, mean: torch.Tensor, L: torch.Tensor, input_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = mean.shape[0]
        d = self.output_dim
        device = mean.device

        mu_Q_list = []
        Sigma_Q_list = []

        for i in range(batch_size):
            mu_P_i = mean[i]
            L_i = L[i]
            
            # Compute covariance and add regularization
            Sigma_P_i = torch.matmul(L_i, L_i.t())
            Sigma_P_i = Sigma_P_i + torch.eye(d, device=device) * self.reg_term
            
            x_i = input_x[i]

            # Use SVD for more stable inverse computation
            try:
                Sigma_P_inv_i = self.stable_inverse(Sigma_P_i)
                
                # Compute intermediate values
                Sigma_Pinv_mu_P_i = torch.matmul(Sigma_P_inv_i, mu_P_i)
                muP_SigmaPinv_mu_P = torch.matmul(mu_P_i, Sigma_Pinv_mu_P_i)
                
                # Use stable logdet computation
                logdet_Sigma_P_i = self.stable_logdet(Sigma_P_i)
                
                # Call the solver with robust parameters
                mu_q, Sigma_q = self.cvxpy_layer(
                    Sigma_P_inv_i,
                    Sigma_Pinv_mu_P_i,
                    muP_SigmaPinv_mu_P,
                    logdet_Sigma_P_i,
                    x_i,
                    solver_args={"solve_method": "SCS", "eps": 1e-5, "max_iters": 5000}
                )
                
                # Ensure output covariance is PSD
                Sigma_q = self.ensure_psd(Sigma_q)
                
            except Exception as e:
                print(f"Error in optimization: {e}")
                # Fallback strategy: use direct projection for mean and a scaled identity for covariance
                # Project mean directly using the precomputed matrices
                mu_q = torch.matmul(self.Astar, x_i) + self.bstar
                # Create a scaled identity covariance as fallback
                Sigma_q = torch.eye(d, device=device) * 0.1
            
            mu_Q_list.append(mu_q)
            Sigma_Q_list.append(Sigma_q)

        mu_Q = torch.stack(mu_Q_list, dim=0)
        Sigma_Q = torch.stack(Sigma_Q_list, dim=0)

        return mu_Q, Sigma_Q
        
    def stable_inverse(self, matrix):
        """Use SVD for more stable matrix inversion"""
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        # Truncate small singular values
        S_inv = torch.where(S > self.reg_term, 1.0 / S, torch.zeros_like(S))
        return torch.matmul(Vh.t(), torch.matmul(torch.diag(S_inv), U.t()))
        
    def stable_logdet(self, matrix):
        """Compute log determinant in a numerically stable way"""
        # Use eigenvalues for stability
        eigenvalues = torch.linalg.eigvalsh(matrix)
        # Filter out very small eigenvalues
        valid_eigenvalues = eigenvalues[eigenvalues > self.reg_term]
        if len(valid_eigenvalues) == 0:
            # Return a small value if all eigenvalues are too small
            return torch.tensor(-30.0, device=matrix.device)
        return torch.sum(torch.log(valid_eigenvalues))
        
    def ensure_psd(self, matrix):
        """Ensure a matrix is positive semi-definite"""
        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        # Replace negative eigenvalues with small positive values
        eigenvalues = torch.clamp(eigenvalues, min=self.reg_term)
        # Reconstruct the matrix
        return eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.t()