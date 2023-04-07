import numpy as np
import pandas as pd
from gurobipy import Model, GRB, Column, quicksum, LinExpr
from data_preprocessor import PreprocessData


class TwoD_BPP:
    def __init__(self, b_path='B.pickle', r_path='R.pickle'):
        self.B_set = PreprocessData(b_path).process()
        self.I_set = PreprocessData(r_path).process()

        self.B = self.B_set.index.to_list()
        self.I = self.I_set.index.to_list()
        self.L = np.arange(1, 3)
        self.a = [1, 3]
        self.b = [1, 3]

        # Parameters of j in B
        self.C_j = self.B_set['C'].to_dict()
        self.L_j = self.B_set['L'].to_dict()
        self.H_j = self.B_set['H'].to_dict()
        self.m = len(self.B)

        # Parameters of i in I
        self.l_i = self.I_set['l'].to_dict()
        self.h_i = self.I_set['h'].to_dict()
        self.r_plus_i = self.I_set['r+'].to_dict()
        self.f_i = self.I_set['f'].to_dict()
        self.rho_i = self.I_set['rho'].to_dict()
        self.phi_i = self.I_set['phi'].to_dict()
        self.n = len(self.I)

        self.M = 1e9

        # Variables of j in B
        self.u_j = {}
        self.Rho_j = {}
        self.Phi_j = {}

        # Variables of i in I
        self.x_i = {}
        self.xp_i = {}
        self.z_i = {}
        self.zp_i = {}
        self.r_iab = {}
        self.x_ikp = {}
        self.z_ikp = {}
        self.g_i = {}
        self.h_ik = {}
        self.s_ik = {}
        self.eta1_ik = {}
        self.eta3_ik = {}
        self.beta_lik = {}
        self.nu_ik = {}
        self.m_ik = {}

        # Variables of i in I and j in B
        self.p_ij = {}

    def bpp_model(self):
        model = Model("2D_BPP")

        # Define Decision Variables
        for j in self.B:
            self.u_j[j] = model.addVar(vtype=GRB.BINARY, name=f'u_{j}')
            self.Rho_j[j] = model.addVar(vtype=GRB.BINARY, name=f'Rho_{j}')
            self.Phi_j[j] = model.addVar(vtype=GRB.BINARY, name=f'Phi_{j}')

        for i in self.I:
            self.x_i[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f'x_{i}')
            self.xp_i[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f'x\'_{i}')
            self.z_i[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f'z_{i}')
            self.zp_i[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f'z\'_{i}')
            self.g_i[i] = model.addVar(vtype=GRB.BINARY, name=f'g_{i}')
            for j in self.B:
                self.p_ij[i, j] = model.addVar(vtype=GRB.BINARY, name=f'p_{i, j}')
            for a in self.a:
                for b in self.b:
                    self.r_iab[i, a, b] = model.addVar(vtype=GRB.BINARY, name=f'r_{i, a, b}')
            for k in self.I:
                self.x_ikp[i, k] = model.addVar(vtype=GRB.BINARY, name=f'x^p_{i, k}')
                self.z_ikp[i, k] = model.addVar(vtype=GRB.BINARY, name=f'z^p_{i, k}')
                self.h_ik[i, k] = model.addVar(vtype=GRB.BINARY, name=f'h_{i, k}')
                self.s_ik[i, k] = model.addVar(vtype=GRB.BINARY, name=f's_{i, k}')
                self.eta1_ik[i, k] = model.addVar(vtype=GRB.BINARY, name=f'eta^1_{i, k}')
                self.eta3_ik[i, k] = model.addVar(vtype=GRB.BINARY, name=f'eta^3_{i, k}')
                self.nu_ik[i, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'nu_{i, k}')
                self.m_ik[i, k] = model.addVar(vtype=GRB.BINARY, name=f'm_{i, k}')
                for l in self.L:
                    self.beta_lik[i, k, l] = model.addVar(vtype=GRB.BINARY, name=f'beta^{l}_{i, k}')

        model.update()

        # Define Objective Function
        obj = LinExpr()

        for j in self.B:
            obj += self.C_j[j] * self.u_j[j]

        model.setObjective(obj, GRB.MINIMIZE)
        model.update()
        print("Objective Function Defined")
        print(model.getObjective())

        # Define Constraints
        # Geometric Constraints
        for j in self.B:
            model.addConstr(quicksum(self.p_ij[i, j] for i in self.I) <= self.M * self.u_j[j], name=f'GC1_{j}')
        for i in self.I:
            model.addConstr(quicksum(self.p_ij[i, j] for j in self.B) == 1, name=f'GC2_{i}')
            model.addConstr(self.xp_i[i] <= quicksum(self.L_j[j] * self.p_ij[i, j] for j in self.B), name=f'GC3_{i}')
            model.addConstr(self.zp_i[i] <= quicksum(self.H_j[j] * self.p_ij[i, j] for j in self.B), name=f'GC4_{i}')
            model.addConstr(
                self.xp_i[i] - self.x_i[i] == self.r_iab[i, 1, 1] * self.l_i[i] + self.r_iab[i, 1, 3] * self.h_i[i], name=f'GC5_{i}')
            model.addConstr(
                self.zp_i[i] - self.z_i[i] == self.r_iab[i, 3, 1] * self.l_i[i] + self.r_iab[i, 3, 3] * self.h_i[i], name=f'GC6_{i}')
            for b in self.b:
                model.addConstr(quicksum(self.r_iab[i, a, b] for a in self.a) == 1, name=f'GC7_{i, b}')
            for a in self.a:
                model.addConstr(quicksum(self.r_iab[i, a, b] for b in self.b) == 1, name=f'GC8_{i, a}')

if __name__ == "__main__":
    run = TwoD_BPP()
    run.bpp_model()