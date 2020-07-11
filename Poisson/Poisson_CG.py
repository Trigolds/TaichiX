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


#Conjugate Gradient
@ti.data_oriented
class CG:
    def __init__(self):
        # grid parameters
        self.show_matrix = True
        self.iter = 0
        self.max_iteration = 500

        self.N_ext = 1  # number of ext cells set so that that total grid size is still power of 2
        self.N_tot = nb_node_1D

        # setup sparse simulation data arrays
        self.r = ti.var(dt=lp)  # residual
        self.x = ti.var(dt=lp)  # solution
        self.x_ref = ti.var(dt=lp)  # solution
        self.b = ti.var(dt=lp)  # b
        self.p = ti.var(dt=lp)  # conjugate gradient
        self.Ap = ti.var(dt=lp)  # matrix-vector product
        self.alpha = ti.var(dt=lp)  # step size
        self.beta = ti.var(dt=lp)  # step size
        self.sum = ti.var(dt=lp)  # storage for reductions

        indicies = ti.ij
        self.grid = ti.root.dense(indicies, self.N_tot).place(self.x, self.x_ref, self.b, self.p, self.Ap, self.r)
        ti.root.place(self.alpha, self.beta, self.sum)

    def foo(self, i:int, j:int):
        return 2*(i*h)**2 + 3*(j*h)**2

    #@ti.kernel
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
                    self.r[i,j] = 0.0
        for i in range(self.N_ext, self.N_tot - self.N_ext):
            for j in range(self.N_ext, self.N_tot - self.N_ext):
                self.b[i,j] = 10
                self.x[i,j] = 0.0

    @ti.kernel
    def compute_r0(self):
        for i in range(self.N_ext, self.N_tot - self.N_ext):
            for j in range(self.N_ext, self.N_tot - self.N_ext):
                self.r[i,j] = self.b[i,j] - (self.x[i+1,j] + self.x[i-1,j] + self.x[i,j+1] + self.x[i,j-1] - 4.0*self.x[i,j])/h**2

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
            self.r[I] -= self.alpha[None] * self.Ap[I]

    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            self.p[I] = self.r[I] + self.beta[None] * self.p[I]


    def apply_dirichlet(self):
        for i in range(self.N_tot):
            for j in range(self.N_tot):
                if (i==0 or j==0 or i==self.N_tot-1 or j==self.N_tot-1):
                    self.r[i,j] = 0
    
    def final_error(self):    
        # for i in range(self.N_tot):
        #     for j in range(self.N_tot):
        #         self.err += (self.x[i,j]-self.x_ref[i,j])**2
        print("Residual: ", self.sum[None])

    def run(self):
        # gui = ti.GUI("Multigrid Preconditioned Conjugate Gradients",
        #              res=(self.N_gui, self.N_gui))
        self.init()
        self.compute_r0()
        #print("r0:",self.r.to_numpy())
        self.update_p()
        #print("p:",self.p.to_numpy())
        self.reduce(self.p, self.r)
        old_rTr = self.sum[None]

        # CG
        while self.iter < self.max_iteration:
            # self.alpha = rTr / pTAp
            self.compute_Ap()
            #print("Ap:",self.Ap.to_numpy())
            self.reduce(self.p, self.Ap)
            pAp = self.sum[None]
            self.alpha[None] = old_rTr / pAp
            #print("alpha:",self.alpha[None])

            # self.x = self.x + self.alpha self.p
            self.update_x()
            
            # self.r = self.r - self.alpha self.Ap
            self.update_r()
            self.apply_dirichlet()

            # check for convergence
            self.reduce(self.r, self.r)
            rTr = self.sum[None]
            if rTr < 1.0e-9:
                break

            # self.beta = new_rTr / old_rTr
            self.reduce(self.r, self.r)
            new_rTr = self.sum[None]
            self.beta[None] = new_rTr / old_rTr

            # self.p = self.e + self.beta self.p
            self.update_p()
            
            old_rTr = new_rTr

            print(f'iter {self.iter}, residual={rTr}')
            self.iter += 1

        # ti.kernel_profiler_print()
        if(self.show_matrix):
            print("Solution:",self.x.to_numpy())
            print("Ref Solution:",self.x_ref.to_numpy())
        self.final_error()


def main():
    solver = CG()
    t = time.time()
    solver.run()
    print(f'Solver time: {time.time() - t:.3f} s')

if __name__ == '__main__':
    main()