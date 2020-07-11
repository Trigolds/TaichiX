import numpy as np
import time
import taichi as ti

lp = ti.f32
ti.init(default_fp=lp, arch=ti.x64, kernel_profiler=False)
def vec(*xs):
  return ti.Vector(list(xs))

#define grid
level_max = 8
nb_ele_1D = 2**level_max
nb_node_1D = nb_ele_1D + 1
h = 1.0
dt = 5.0
K = 5.0
pixels = ti.Vector(3, dt=ti.f32, shape=(nb_node_1D, nb_node_1D))


#Conjugate Gradient
@ti.data_oriented
class CG:
    def __init__(self):
        # grid parameters
        self.iter = 0
        self.max_iteration = 500000

        self.N_ext = 1  # number of ext cells set so that that total grid size is still power of 2
        self.N_tot = nb_node_1D
        self.init_T = 293.15 # Inital temperature
        self.bc_T = self.init_T + 300.0 # Boundary temperature

        # setup sparse simulation data arrays
        self.r = ti.var(dt=lp)  # residual
        self.x = ti.var(dt=lp)  # solution
        self.b = ti.var(dt=lp)  # b
        self.p = ti.var(dt=lp)  # conjugate gradient
        self.Ap = ti.var(dt=lp)  # matrix-vector product
        self.alpha = ti.var(dt=lp)  # step size
        self.beta = ti.var(dt=lp)  # step size
        self.sum = ti.var(dt=lp)  # storage for reductions

        indicies = ti.ij
        self.grid = ti.root.dense(indicies, self.N_tot).place(self.x, self.b, self.p, self.Ap, self.r)
        ti.root.place(self.alpha, self.beta, self.sum)

    def init(self, step):
        self.Ap.fill(0.0)
        self.p.fill(0.0)
        for i in range(self.N_tot):
            for j in range(self.N_tot):
                if (i==0 or j==0 or i==self.N_tot-1 or j==self.N_tot-1):
                    self.x[i,j] = self.bc_T
                    self.b[i,j] = self.x[i,j]
                    self.r[i,j] = 0.0
        for i in range(self.N_ext, self.N_tot - self.N_ext):
            for j in range(self.N_ext, self.N_tot - self.N_ext):
                if(step == 0):
                    self.b[i,j] = self.init_T
                else:
                    self.b[i,j] = self.x[i,j]
                self.x[i,j] = 0.0

    @ti.kernel
    def compute_r0(self):
        for i in range(self.N_ext, self.N_tot - self.N_ext):
            for j in range(self.N_ext, self.N_tot - self.N_ext):
                self.r[i,j] = self.b[i,j] - ((1.0 + 4.0*K*dt/h**2)*self.x[i,j] - K*dt/h**2*(self.x[i+1,j] + self.x[i-1,j] + self.x[i,j+1] + self.x[i,j-1]))

    @ti.kernel
    def compute_Ap(self):
        for i in range(self.N_ext, self.N_tot - self.N_ext):
            for j in range(self.N_ext, self.N_tot - self.N_ext):
                self.Ap[i,j] = (1.0 + 4.0*K*dt/h**2)*self.p[i,j] - K*dt/h**2*(self.p[i+1,j] + self.p[i-1,j] + self.p[i,j+1] + self.p[i,j-1])

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
        print(f'iter {self.iter}, residual={self.sum[None]}')
    
    @ti.kernel
    def render(self):
        for i, j in pixels:
            val = (self.x[i,j] - self.init_T) / (self.bc_T - self.init_T)
            col = vec(val, 0.0, 1.0-val)
            pixels[i, j] = vec(col[0], col[1], col[2])

    def run(self, step):
        self.init(step)
        self.compute_r0()
        self.update_p()
        self.reduce(self.p, self.r)
        old_rTr = self.sum[None]

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
            self.reduce(self.r, self.r)
            rTr = self.sum[None]
            if rTr < 1.0e-12:
                break
            # self.beta = new_rTr / old_rTr
            self.reduce(self.r, self.r)
            new_rTr = self.sum[None]
            self.beta[None] = new_rTr / old_rTr
            # self.p = self.e + self.beta self.p
            self.update_p()   
            old_rTr = new_rTr
            self.iter += 1

        self.final_error()
        self.render()


def main(output_img=True):
    solver = CG()
    t = time.time()
    gui = ti.GUI("Thermal2D", res=(nb_node_1D, nb_node_1D))
    for step in range(200):
        solver.run(step)
        print(f'Step: {step}; Solver time: {time.time() - t:.3f} s')
        gui.set_image(pixels.to_numpy())
        if gui.get_event(ti.GUI.ESCAPE):
            exit()
        if output_img:       
            gui.show(f'{step:04d}.png')
        else:
            gui.show()

if __name__ == '__main__':
    main()