import sys
import scipy as sp
import scipy.linalg as splin
from numpy import linalg as LA

def sunsal(M,y,AL_iters=1000,lambda_0=0.,positivity=False,addone=False,tol=1e-4,x0 = None,verbose=False):

    [LM,p] = M.shape # mixing matrixsize
    [L,N] = y.shape # data set size
    if LM != L:
        sys.exit('mixing matrix M and data set y are inconsistent')

    AL_iters = int(AL_iters)
    if (AL_iters < 0 ):
        sys.exit('AL_iters must a positive integer')

    # If lambda is scalar convert it into vector
    lambda_0 = ( lambda_0 * sp.ones((N,p)) ).T
    if (lambda_0<0).any():
        sys.exit('lambda_0 must be positive')

    # compute mean norm
    norm_m = splin.norm(M)*(25+p)/float(p)
    # rescale M and Y and lambda
    M = M/norm_m
    y = y/norm_m
    lambda_0 = lambda_0/norm_m**2

    if x0 is not None:
        if (x0.shape[0]==p) or (x0.shape[0]==N):
            sys.exit('initial X is not inconsistent with M or Y')


    #---------------------------------------------
    # just least squares
    #---------------------------------------------
    if (lambda_0.sum() == 0) and (not positivity) and (not addone):
        z = sp.dot(splin.pinv(M),y)
        # primal and dual residues
        res_p = 0.
        res_d = 0.
        return z,res_p,res_d,None

    #---------------------------------------------
    # least squares constrained (sum(x) = 1)
    #---------------------------------------------
    SMALL = 1e-12;
    if (lambda_0.sum() == 0) and (addone) and (not positivity):
        F = sp.dot(M.T,M)
        # test if F is invertible
        if LA.cond(F) > SMALL:
            # compute the solution explicitly
            IF = splin.inv(F);
            z = sp.dot(sp.dot(IF,M.T),y) - (1./IF.sum())*sp.dot(sp.sum(IF,axis=1,keepdims=True) , ( sp.dot(sp.dot(sp.sum(IF,axis=0,keepdims=True),M.T),y) - 1.))
            # primal and dual residues
            res_p = 0
            res_d = 0

            return z,res_p,res_d,None
        else:
            sys.exit('Bad conditioning of M.T*M')


    #---------------------------------------------
    #  Constants and initializations
    #---------------------------------------------
    mu_AL = 0.01
    mu = 10*lambda_0.mean() + mu_AL

    [UF,SF] = splin.svd(sp.dot(M.T,M))[:2]
    IF = sp.dot( sp.dot(UF,sp.diag(1./(SF+mu))) , UF.T )
    Aux = (1./IF.sum()) * sp.sum(IF,axis=1,keepdims=True)
    x_aux = sp.sum(Aux,axis=1,keepdims=True)
    IF1 = IF - sp.dot(Aux,sp.sum(IF,axis=0,keepdims=True))


    yy = sp.dot(M.T,y)

    #---------------------------------------------
    #  Initializations
    #---------------------------------------------

    # no intial solution supplied
    if x0 is None:
       x = sp.dot( sp.dot(IF,M.T) , y)
    else:
        x = x0

    z = x
    # scaled Lagrange Multipliers
    d  = 0*z

    #---------------------------------------------
    #  AL iterations - main body
    #---------------------------------------------
    tol1 = sp.sqrt(N*p)*tol
    tol2 = sp.sqrt(N*p)*tol
    i=1
    res_p = sp.inf
    res_d = sp.inf
    maskz = sp.ones(z.shape)
    mu_changed = 0

    #--------------------------------------------------------------------------
    # constrained  leat squares (CLS) X >= 0
    #--------------------------------------------------------------------------
    if (lambda_0.sum() ==  0)  and (not addone):
        while (i <= AL_iters) and ((abs(res_p) > tol1) or (abs(res_d) > tol2)):
            # save z to be used later
            if (i%10) == 1:
                z0 = z
            # minimize with respect to z
            z = sp.maximum(x-d,0)
            # minimize with respect to x
            x = sp.dot(IF,yy + mu*(z+d))
            # Lagrange multipliers update
            d -= (x-z)

            # update mu so to keep primal and dual residuals whithin a factor of 10
            if (i%10) == 1:
                # primal residue
                res_p = splin.norm(x-z)
                # dual residue
                res_d = mu*splin.norm(z-z0)
                if verbose:
                    print("i = {:d}, res_p = {:f}, res_d = {:f}\n").format(i,res_p,res_d)
                # update mu
                if res_p > 10*res_d:
                    mu = mu*2
                    d = d/2
                    mu_changed = True
                elif res_d > 10*res_p:
                    mu = mu/2
                    d = d*2
                    mu_changed = True

                if  mu_changed:
                    # update IF and IF1
                    IF = sp.dot( sp.dot(UF,sp.diag(1./(SF+mu))) , UF.T )
                    # Aux = (1./IF.sum()) * sp.sum(IF,axis=1,keepdims=True)
                    # x_aux = sp.sum(Aux,axis=1,keepdims=True)
                    # IF1 = IF - sp.dot(Aux,sp.sum(IF,axis=0,keepdims=True))
                    mu_changed = False

            i+=1

    #--------------------------------------------------------------------------
    # Fully constrained  leat squares (FCLS) X >= 0
    #--------------------------------------------------------------------------
    elif (lambda_0.sum() ==  0)  and addone:
        while (i <= AL_iters) and ((abs(res_p) > tol1) or (abs(res_d) > tol2)):
            # save z to be used later
            if (i%10) == 1:
                z0 = z
            # minimize with respect to z
            z = sp.maximum(x-d,0)
            # minimize with respect to x
            x = sp.dot(IF1,yy + mu*(z+d)) + x_aux
            # Lagrange multipliers update
            d -= (x-z)

            # update mu so to keep primal and dual residuals whithin a factor of 10
            if (i%10) == 1:
                # primal residue
                res_p = splin.norm(x-z)
                # dual residue
                res_d = mu*splin.norm(z-z0)
                if verbose:
                    print("i = {:d}, res_p = {:f}, res_d = {:f}\n").format(i,res_p,res_d)
                # update mu
                if res_p > 10*res_d:
                    mu = mu*2
                    d = d/2
                    mu_changed = True
                elif res_d > 10*res_p:
                    mu = mu/2
                    d = d*2
                    mu_changed = True

                if  mu_changed:
                    # update IF and IF1
                    IF = sp.dot( sp.dot(UF,sp.diag(1./(SF+mu))) , UF.T )
                    Aux = (1./IF.sum()) * sp.sum(IF,axis=1,keepdims=True)
                    x_aux = sp.sum(Aux,axis=1,keepdims=True)
                    IF1 = IF - sp.dot(Aux,sp.sum(IF,axis=0,keepdims=True))
                    mu_changed = False

            i+=1

        #--------------------------------------------------------------------------
        # generic SUNSAL: lambda > 0
        #--------------------------------------------------------------------------
    else:
        # implement soft_th
        while (i <= AL_iters) and ((abs(res_p) > tol1) or (abs(res_d) > tol2)):
            # save z to be used later
            if (i%10) == 1:
                z0 = z
            # minimize with respect to z
            nu = x-d
            z = sp.sign(nu) * sp.maximum(sp.absolute(nu) - lambda_0/mu,0)
            # teste for positivity
            if positivity:
                z = sp.maximum(z,0)
            # teste for sum-to-one
            if addone:
                x = sp.dot(IF1,yy+mu*(z+d)) + x_aux
            else:
                x = sp.dot(IF,yy+mu*(z+d))
            # Lagrange multipliers update
            d -= (x-z)

            # update mu so to keep primal and dual residuals whithin a factor of 10
            if (i%10) == 1:
                # primal residue
                res_p = splin.norm(x-z)
                # dual residue
                res_d = mu*splin.norm(z-z0)
                if verbose:
                    print("i = {:d}, res_p = {:f}, res_d = {:f}\n").format(i,res_p,res_d)
                # update mu
                if res_p > 10*res_d:
                    mu = mu*2
                    d = d/2
                    mu_changed = True
                elif res_d > 10*res_p:
                    mu = mu/2
                    d = d*2
                    mu_changed = True

                if  mu_changed:
                    # update IF and IF1
                    IF = sp.dot( sp.dot(UF,sp.diag(1./(SF+mu))) , UF.T )
                    Aux = (1./IF.sum()) * sp.sum(IF,axis=1,keepdims=True)
                    x_aux = sp.sum(Aux,axis=1,keepdims=True)
                    IF1 = IF - sp.dot(Aux,sp.sum(IF,axis=0,keepdims=True))
                    mu_changed = False

            i+=1

    return x,res_p,res_d,i

