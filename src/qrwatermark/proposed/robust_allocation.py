from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog

@dataclass(frozen=True)
class MinimaxAllocation:
    weights: np.ndarray
    guaranteed_margin: float
    scenario_margins: np.ndarray
    success: bool
    message: str

def solve_minimax_allocation(score_matrix,*,max_weight=None,costs=None,budget=None)->MinimaxAllocation:
    """Solve max_p min_l (S p)_l over the probability simplex.

    Rows of S are extreme convolution scenarios and columns are candidate
    carriers/frequency channels. Optional max_weight prevents a degenerate
    single-carrier solution; optional costs/budget impose a perceptual budget.
    This is the finite robust counterpart of the unknown-filter allocation
    problem once the convolution uncertainty set is represented by extremes.
    """
    S=np.asarray(score_matrix,dtype=np.float64)
    if S.ndim!=2 or S.shape[0]<1 or S.shape[1]<1: raise ValueError("score_matrix must be non-empty 2-D")
    L,J=S.shape
    # Variables [p_1,...,p_J,t], minimize -t.
    c=np.zeros(J+1); c[-1]=-1.0
    A=[]; b=[]
    for ell in range(L):
        row=np.zeros(J+1); row[:J]=-S[ell]; row[-1]=1.0; A.append(row); b.append(0.0)
    if costs is not None:
        if budget is None: raise ValueError("budget is required when costs are provided")
        cc=np.asarray(costs,dtype=np.float64).ravel()
        if cc.size!=J: raise ValueError("costs must match carrier count")
        row=np.zeros(J+1); row[:J]=cc; A.append(row); b.append(float(budget))
    Aeq=np.zeros((1,J+1)); Aeq[0,:J]=1.0
    bounds=[(0.0,1.0 if max_weight is None else float(max_weight)) for _ in range(J)]+[(None,None)]
    res=linprog(c,A_ub=np.asarray(A),b_ub=np.asarray(b),A_eq=Aeq,b_eq=np.array([1.0]),bounds=bounds,method='highs')
    if not res.success:
        return MinimaxAllocation(np.full(J,np.nan),float('-inf'),np.full(L,np.nan),False,str(res.message))
    p=np.asarray(res.x[:J]); margins=S@p
    return MinimaxAllocation(p,float(np.min(margins)),margins,True,str(res.message))
