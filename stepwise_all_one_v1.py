import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.datasets import load_breast_cancer

def vectorized_score_selection(df, target_col):
    y = df[target_col].values
    included = []
    remaining = [col for col in df.columns if col != target_col]
    

    # 1. Fit the current (null) model to get residuals and weights
    if not included:
        X_inc = np.ones((len(df), 1))
    else:
        X_inc = sm.add_constant(df[included]).values
        
    # Use GLM to get the current model's state (assuming Logistic/Binomial)
    # For Linear Regression, use sm.families.Gaussian()
    model = sm.GLM(y, X_inc, family=sm.families.Binomial()).fit()
    
    mu = model.mu              # Predicted probabilities
    w = mu * (1 - mu)          # Weights for logistic (W = p*(1-p))
    w_sqrt = np.sqrt(w)
    grad = y - mu              # Working residuals (the score component)

    # 2. Prepare the Candidate Matrix
    X_rem = df[remaining].values
    
    # 3. Vectorized Math: Score Statistics for ALL candidates at once
    # U is the score vector for all candidates: X_rem.T @ (y - mu)
    U = X_rem.T @ grad 
    
    # Calculate the adjusted variance (denominator of the score test)
    # This accounts for the variables already in the model
    X_inc_w = X_inc * w_sqrt[:, None]
    X_rem_w = X_rem * w_sqrt[:, None]
    
    # Use QR decomposition for a fast, stable projection
    Q, _ = np.linalg.qr(X_inc_w)
    # Project candidates onto the orthogonal space of included variables
    X_rem_w_ortho = X_rem_w - Q @ (Q.T @ X_rem_w)
    
    # V is the variance of each candidate's score
    V = np.sum(X_rem_w_ortho**2, axis=0)
    
    # Handle potential division by zero for perfectly collinear features
    V = np.where(V < 1e-10, np.nan, V)
    
    # 4. Compute Chi-Square Stats and P-Values
    chi2_stats = (U**2) / V
    p_values = stats.chi2.sf(chi2_stats, df=1)
    
    score_df = {
        'Variable': remaining,
        'Score Chi': chi2_stats,
        'p-value':p_values
        }
    
    #score_df = pd.DataFrame(score_df)
            
    return score_df

# Example Run
data = load_breast_cancer()
df_example = pd.DataFrame(data.data, columns=data.feature_names)
df_example['target'] = data.target

selected = vectorized_score_selection(df_example, 'target')

#%%

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.datasets import load_breast_cancer
from tqdm import tqdm  # Ensure you have this installed: pip install tqdm

def fast_score_selection_float32(df, target_col):
    # 1. Memory Efficiency: Convert to float32
    # This reduces RAM usage by 50%
    y = df[target_col].values.astype(np.float32)
    remaining_names = [col for col in df.columns if col != target_col]
    X_all = df[remaining_names].values.astype(np.float32)
    
    included_indices = []
    remaining_indices = list(range(len(remaining_names)))
    
    print(f"Starting Selection: {len(remaining_indices)} candidates, {len(y)} observations.")
    
    # Initialize the Progress Bar
    # We use a manually updated pbar because the loop stops when alpha is hit
    pbar = tqdm(total=len(remaining_indices), desc="Selecting Features", unit="var")


    # 2. Fit current model to get residuals (mu) and weights (w)
    if not included_indices:
        X_inc = np.ones((len(y), 1), dtype=np.float32)
    else:
        # Statsmodels internally converts to float64, but we keep 
        # our source matrix small as long as possible.
        X_inc = sm.add_constant(X_all[:, included_indices])
        
    # Fit Null Model (Logistic Regression)
    # GLM is fast for 60k rows
    model = sm.GLM(y, X_inc, family=sm.families.Binomial()).fit()
    mu = model.mu.astype(np.float32)
    w = (mu * (1 - mu)).astype(np.float32)
    w_sqrt = np.sqrt(w)[:, np.newaxis]
    grad = (y - mu).astype(np.float32)

    # 3. Vectorized Math on ALL remaining candidates
    X_rem = X_all[:, remaining_indices]
    
    # Calculate Score Vector U (Gradient)
    U = X_rem.T @ grad
    
    # 4. Fast Orthogonalization using QR
    # This handles the covariance with variables already in the model
    X_inc_w = X_inc * w_sqrt
    X_rem_w = X_rem * w_sqrt
    
    Q, _ = np.linalg.qr(X_inc_w)
    
    # Associative projection: Q @ (Q.T @ X_rem_w) is much faster 
    # than (Q @ Q.T) @ X_rem_w for large N
    projection = Q @ (Q.T @ X_rem_w)
    V = np.sum((X_rem_w - projection)**2, axis=0)
    
    # 5. Compute Chi-Square and P-values
    V[V < 1e-10] = np.nan 
    chi2_stats = (U**2) / V
    p_values = stats.chi2.sf(chi2_stats, df=1)


    
    score_df = {
        'Variable': remaining_names,
        'Score Chi': chi2_stats,
        'p-value':p_values
        }
    
    score_df = pd.DataFrame(score_df)
    
    pbar.close()
    return score_df


# Example Run
data = load_breast_cancer()
df_example = pd.DataFrame(data.data, columns=data.feature_names)
df_example['target'] = data.target

selected_features1 = fast_score_selection_float32(df_example, 'target')


#%%

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.datasets import load_breast_cancer

def vectorized_score_selection(df, target_col, alpha=0.05):
    y = df[target_col].values
    included = []
    remaining = [col for col in df.columns if col != target_col]
    
    print(f"{'Step':<5} | {'Added Variable':<25} | {'Score P-value':<15}")
    print("-" * 60)

    step = 1
    while len(remaining) > 0:
        # 1. Fit the current (null) model to get residuals and weights
        if not included:
            X_inc = np.ones((len(df), 1))
        else:
            X_inc = sm.add_constant(df[included]).values
            
        # Use GLM to get the current model's state (assuming Logistic/Binomial)
        # For Linear Regression, use sm.families.Gaussian()
        model = sm.GLM(y, X_inc, family=sm.families.Binomial()).fit()
        
        mu = model.mu              # Predicted probabilities
        w = mu * (1 - mu)          # Weights for logistic (W = p*(1-p))
        w_sqrt = np.sqrt(w)
        grad = y - mu              # Working residuals (the score component)

        # 2. Prepare the Candidate Matrix
        X_rem = df[remaining].values
        
        # 3. Vectorized Math: Score Statistics for ALL candidates at once
        # U is the score vector for all candidates: X_rem.T @ (y - mu)
        U = X_rem.T @ grad 
        
        # Calculate the adjusted variance (denominator of the score test)
        # This accounts for the variables already in the model
        X_inc_w = X_inc * w_sqrt[:, None]
        X_rem_w = X_rem * w_sqrt[:, None]
        
        # Use QR decomposition for a fast, stable projection
        Q, _ = np.linalg.qr(X_inc_w)
        # Project candidates onto the orthogonal space of included variables
        X_rem_w_ortho = X_rem_w - Q @ (Q.T @ X_rem_w)
        
        # V is the variance of each candidate's score
        V = np.sum(X_rem_w_ortho**2, axis=0)
        
        # Handle potential division by zero for perfectly collinear features
        V = np.where(V < 1e-10, np.nan, V)
        
        # 4. Compute Chi-Square Stats and P-Values
        chi2_stats = (U**2) / V
        p_values = stats.chi2.sf(chi2_stats, df=1)

        # 5. Select the best variable
        best_idx = np.nanargmin(p_values)
        best_p = p_values[best_idx]
        best_var = remaining[best_idx]

        if best_p < alpha:
            included.append(best_var)
            remaining.remove(best_var)
            print(f"{step:<5} | {best_var:<25} | {best_p:.6e}")
            step += 1
        else:
            break
            
    return included

# Example Run
data = load_breast_cancer()
df_example = pd.DataFrame(data.data, columns=data.feature_names)
df_example['target'] = data.target

selected = vectorized_score_selection(df_example, 'target')


#%%


import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.datasets import load_breast_cancer

def vectorized_score_selection(df, target_col, alpha=0.05):
    y = df[target_col].values
    included = []
    remaining = [col for col in df.columns if col != target_col]
    
    print(f"{'Step':<5} | {'Added Variable':<25} | {'Score P-value':<15}")
    print("-" * 60)

    step = 1
    while len(remaining) > 0:
        # 1. Fit the current (null) model to get residuals and weights
        if not included:
            X_inc = np.ones((len(df), 1))
        else:
            X_inc = sm.add_constant(df[included]).values
            
        # Use GLM to get the current model's state (assuming Logistic/Binomial)
        # For Linear Regression, use sm.families.Gaussian()
        model = sm.GLM(y, X_inc, family=sm.families.Binomial()).fit()
        
        mu = model.mu              # Predicted probabilities
        w = mu * (1 - mu)          # Weights for logistic (W = p*(1-p))
        w_sqrt = np.sqrt(w)
        grad = y - mu              # Working residuals (the score component)

        # 2. Prepare the Candidate Matrix
        X_rem = df[remaining].values
        
        # 3. Vectorized Math: Score Statistics for ALL candidates at once
        # U is the score vector for all candidates: X_rem.T @ (y - mu)
        U = X_rem.T @ grad 
        
        # Calculate the adjusted variance (denominator of the score test)
        # This accounts for the variables already in the model
        X_inc_w = X_inc * w_sqrt[:, None]
        X_rem_w = X_rem * w_sqrt[:, None]
        
        # Use QR decomposition for a fast, stable projection
        Q, _ = np.linalg.qr(X_inc_w)
        # Project candidates onto the orthogonal space of included variables
        X_rem_w_ortho = X_rem_w - Q @ (Q.T @ X_rem_w)
        
        # V is the variance of each candidate's score
        V = np.sum(X_rem_w_ortho**2, axis=0)
        
        # Handle potential division by zero for perfectly collinear features
        V = np.where(V < 1e-10, np.nan, V)
        
        # 4. Compute Chi-Square Stats and P-Values
        chi2_stats = (U**2) / V
        p_values = stats.chi2.sf(chi2_stats, df=1)

        # 5. Select the best variable
        best_idx = np.nanargmin(p_values)
        best_p = p_values[best_idx]
        best_var = remaining[best_idx]

        if best_p < alpha:
            included.append(best_var)
            remaining.remove(best_var)
            print(f"{step:<5} | {best_var:<25} | {best_p:.6e}")
            step += 1
        else:
            break
            
    return included

# Example Run
data = load_breast_cancer()
df_example = pd.DataFrame(data.data, columns=data.feature_names)
df_example['target'] = data.target

selected = vectorized_score_selection(df_example, 'target')

#%%