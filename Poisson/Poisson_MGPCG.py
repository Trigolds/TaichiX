import numpy as np
import time
import taichi as ti

lp = ti.f32
ti.init(default_fp=lp, arch=ti.x64, kernel_profiler=False)

#define grid
level_max = 2
nb_ele_1D = 2**level_max
nb_node_1D = nb_ele_1D + 1
h = 1.0


#MGPCG V Cycle
@ti.data_oriented
class MGPCG_Jacobi_V:
    def __init__(self):
        # grid parameters
        self.show_matrix = True
        self.use_jacobi_smoother = False 
        self.w = 2.0/3.0
        self.err = 0.0

        self.n_mg_levels = level_max
        self.pre_and_post_smoothing = 20
        self.dim = 2
        self.iter = 0
        self.max_iteration = 500

        self.N_ext = 1  # number of ext cells set so that that total grid size is still power of 2
        self.N_tot = nb_node_1D
        self.b = 10

        self.e = ti.var(dt=lp)
        self.x = ti.var(dt=lp)  
        self.x_ref = ti.var(dt=lp) 
        self.res = ti.var(dt=lp)  
        self.b = ti.var(dt=lp)  
        self.p = ti.var(dt=lp)  # conjugate gradient
        self.Ap = ti.var(dt=lp)  # matrix-vector product
        self.alpha = ti.var(dt=lp)  # step size
        self.beta = ti.var(dt=lp)  # step size
        self.sum = ti.var(dt=lp)  # storage for reductions

        indicies = ti.ij
        self.grid = ti.root.dense(indicies, self.N_tot).place(self.x, self.x_ref, self.b, self.p, self.Ap, self.res, self.e)
        ti.root.place(self.alpha, self.beta, self.sum)

    def foo(self, i:int, j:int):
        return 2*(i*h)**2 + 3*(j*h)**2

    def init(self):
        self.Ap.fill(0.0)
        self.p.fill(0.0)
        self.x.fill(0.0)
        for i in range(self.N_tot):
            for j in range(self.N_tot):
                self.x_ref[i,j] = self.foo(i,j)
                if (i==0 or j==0 or i==self.N_tot-1 or j==self.N_tot-1):
                    self.x[i,j] = self.foo(i,j)
                    self.b[i,j] = self.x[i,j]
                    self.res[i,j] = 0.0
        for i in range(self.N_ext, self.N_tot - self.N_ext):
            for j in range(self.N_ext, self.N_tot - self.N_ext):
                self.b[i,j] = 10
                self.x[i,j] = 0.0

    @ti.kernel
    def compute_r0(self):
        for i in range(self.N_ext, self.N_tot - self.N_ext):
            for j in range(self.N_ext, self.N_tot - self.N_ext):
                self.res[i,j] = self.b[i,j] - (self.x[i+1,j] + self.x[i-1,j] + self.x[i,j+1] + self.x[i,j-1] - 4.0*self.x[i,j])/h**2
        
    @ti.kernel
    def compute_Ap(self):
        for i in range(self.N_ext, self.N_tot - self.N_ext):
            for j in range(self.N_ext, self.N_tot - self.N_ext):
                self.Ap[i,j] = (self.p[i+1,j] + self.p[i-1,j] + self.p[i,j+1] + self.p[i,j-1] - 4.0*self.p[i,j])/h**2

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):
        self.sum[None] = 0
        for I in ti.grouped(p):
            self.sum[None] += p[I] * q[I]

    @ti.kernel
    def update_x(self):
        for I in ti.grouped(self.p):
            self.x[I] += self.alpha[None] * self.p[I]

    @ti.kernel
    def update_r(self):
        for I in ti.grouped(self.p):
            self.res[I] -= self.alpha[None] * self.Ap[I]

    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            self.p[I] = self.e[I] + self.beta[None] * self.p[I]

    def restrit(self, r):
        tmp = np.zeros((r.shape[0]//2+1,r.shape[1]//2+1))
        for i in range(self.N_ext, tmp.shape[0] - self.N_ext):
            for j in range(self.N_ext, tmp.shape[1] - self.N_ext):
                tmp[i, j] = 1.0/16.0*( r[2*i-1,2*j-1] + r[2*i+1,2*j-1] + r[2*i-1,2*j+1] + r[2*i+1,2*j+1] +
                                    2.0*( r[2*i,2*j-1] + r[2*i,2*j+1] + r[2*i-1,2*j] + r[2*i+1,2*j] ) +
                                    4.0*r[2*i,2*j])
        return tmp

    def prolongate(self, u):
        tmp = np.zeros((u.shape[0]*2-1, u.shape[1]*2-1))
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                tmp[2*i,2*j] = u[i,j]
                if(i!=u.shape[0]-1 and j!=u.shape[1]-1):
                    tmp[2*i+1,2*j] = 0.5*(u[i,j] + u[i+1,j])
                    tmp[2*i,2*j+1] = 0.5*(u[i,j] + u[i,j+1])
                    tmp[2*i+1,2*j+1] = 0.25*(u[i,j] + u[i+1,j] + u[i,j+1] + u[i+1,j+1])
        return tmp
    
    def Jacobi(self, u, b):
        tmp = np.zeros(u.shape)
        i = self.N_ext
        j = self.N_ext
        tmp[i,j] = 0.25*(u[i+1,j] + u[i-1,j] + u[i,j-1] + u[i,j+1] - h**2*b[i,j])  
        return tmp

    def smoother_Jacobi(self, u, b):
        for i in range(self.N_ext, u.shape[0] - self.N_ext):
            for j in range(self.N_ext, u.shape[1] - self.N_ext):
                tmp = 0.25*(u[i+1,j] + u[i-1,j] + u[i,j-1] + u[i,j+1] - h**2*b[i,j])  
                u[i,j] = self.w*tmp + (1-self.w)*u[i,j]

    def GS(self, u, b, phase):
        for i in range(self.N_ext, u.shape[0] - self.N_ext):
            for j in range(self.N_ext, u.shape[1] - self.N_ext):
                if (i + j) & 1 == phase:
                    u[i,j] = 0.25*(u[i+1,j] + u[i-1,j] + u[i,j-1] + u[i,j+1] - h**2*b[i,j])  
    
    def smoother_GS(self, u, b):
        self.GS(u,b,0)
        self.GS(u,b,1)
    
    def residual(self, u, b):
        tmp = np.zeros(u.shape)
        for i in range(self.N_ext, u.shape[0] - self.N_ext):
            for j in range(self.N_ext, u.shape[1] - self.N_ext):
                tmp[i, j] = b[i, j] - (u[i+1,j] + u[i-1,j] + u[i,j-1] + u[i,j+1] - 4.0*u[i,j])/h**2
        return tmp

    def MG(self, u, b):
        for _ in range(self.pre_and_post_smoothing):
            if(self.use_jacobi_smoother):
                self.smoother_Jacobi(u,b)
            else:
                self.smoother_GS(u,b)

        if(u.shape[0]>3):
            r = self.residual(u, b)
            rc = self.restrit(r)
            ec = self.MG(np.zeros((u.shape[0]//2+1,u.shape[1]//2+1)), rc)
            e = self.prolongate(ec)
            u = u + e

        for _ in range(self.pre_and_post_smoothing):
            if(self.use_jacobi_smoother):
                self.smoother_Jacobi(u,b)
            else:
                self.smoother_GS(u,b)

        return u

    def e2e0(self, e):
        for i in range(e.shape[0]):
            for j in range(e.shape[1]):
                self.e[i,j] = e[i,j]
   

    def apply_dirichlet(self):
        for i in range(self.N_tot):
            for j in range(self.N_tot):
                if (i==0 or j==0 or i==self.N_tot-1 or j==self.N_tot-1):
                    self.res[i,j] = 0
    
    def final_error(self):
        self.reduce(self.res, self.res)
        print("Residual: ", self.sum[None])

    def run(self):
        self.init()
        self.compute_r0()

        e = self.MG(np.zeros((self.N_tot, self.N_tot)), self.res)
        self.e2e0(e)
        
        self.update_p()
        self.reduce(self.p, self.res)
        old_rTr = self.sum[None]

        # CG
        while self.iter < self.max_iteration:
            # self.alpha = rTr / pTAp
            self.compute_Ap()
            self.reduce(self.p, self.Ap)
            pAp = self.sum[None]
            self.alpha[None] = old_rTr / pAp

            # self.x = self.x + self.alpha self.p
            self.update_x()
            
            # self.r = self.r - self.alpha self.Ap
            self.update_r()
            self.apply_dirichlet()

            # check for convergence
            self.reduce(self.res, self.res)
            rTr = self.sum[None]
            if rTr < 1.0e-9:
                break

            # self.e = M^-1 self.r
            e = self.MG(np.zeros((self.N_tot, self.N_tot)), self.res)
            self.e2e0(e)

            # self.beta = new_rTr / old_rTr
            self.reduce(self.e, self.res)
            new_rTr = self.sum[None]
            self.beta[None] = new_rTr / old_rTr

            # self.p = self.e + self.beta self.p
            self.update_p()
            
            old_rTr = new_rTr

            print(f'iter {self.iter}, residual={rTr}')
            self.iter += 1

        if(self.show_matrix):
            print("Solution:",self.x.to_numpy())
            print("Ref Solution:",self.x_ref.to_numpy())
        self.final_error()


def main():
    solver = MGPCG_Jacobi_V()
    t = time.time()
    solver.run()
    print(f'Solver time: {time.time() - t:.3f} s')

if __name__ == '__main__':
    main()