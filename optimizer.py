import pickle

import numpy as np
import pandas as pd
from gurobipy import Model, GRB, quicksum, LinExpr, abs_
from data_preprocessor import PreprocessData
from datetime import datetime
import os

class TwoD_BPP:
    def __init__(self, b_path='B.pickle', r_path='R.pickle', subset=None):
        self.B_set = PreprocessData(b_path).process()
        self.I_set = PreprocessData(r_path).process()

        if subset is None:
            pass
        elif type(subset) is int:
            if subset < len(self.I_set):
                print(f"Using {subset} number of items")
                self.I_set = self.I_set.iloc[:subset]
            else:
                raise UserWarning(f"Subset number passed greater than available set. Using complete dataset instead")

        self.B = self.B_set.index.to_list()
        self.I = self.I_set.index.to_list()
        self.L = [1, 3]
        self.a = [1, 3]
        self.b = [1, 3]

        # Parameters of j in B
        self.C_j = self.B_set['C'].to_dict()
        self.L_j = self.B_set['L'].to_dict()
        self.H_j = self.B_set['H'].to_dict()
        self.m = len(self.B)

        # Max Params
        self.L_max = max(self.L_j.values())
        self.H_max = max(self.H_j.values())

        # Parameters of i in I
        self.l_i = self.I_set['l'].to_dict()
        self.h_i = self.I_set['h'].to_dict()
        self.r_plus_i = self.I_set['r+'].to_dict()
        self.f_i = self.I_set['f'].to_dict()
        self.rho_i = self.I_set['rho'].to_dict()
        self.phi_i = self.I_set['phi'].to_dict()
        self.n = len(self.I)

        self.M = 1e3

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
        self.o_ik = {}
        self.beta_lik = {}
        self.nu_ik = {}
        self.m_ik = {}

        # Variables of i in I and j in B
        self.p_ij = {}
        self.model = Model("2D_BPP")
        # Initialise Model

    def setup_variables(self):
        # Define Decision Variables
        for j in self.B:
            self.u_j[j] = self.model.addVar(vtype=GRB.BINARY, name=f'u_{j}')
            self.Rho_j[j] = self.model.addVar(vtype=GRB.BINARY, name=f'Rho_{j}')
            self.Phi_j[j] = self.model.addVar(vtype=GRB.BINARY, name=f'Phi_{j}')

        for i in self.I:
            self.x_i[i] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f'x_{i}')
            self.xp_i[i] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f'xp_{i}')
            self.z_i[i] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f'z_{i}')
            self.zp_i[i] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f'zp_{i}')
            self.g_i[i] = self.model.addVar(vtype=GRB.BINARY, name=f'g_{i}')
            for j in self.B:
                self.p_ij[i, j] = self.model.addVar(vtype=GRB.BINARY, name=f'p_{i, j}')
            for a in self.a:
                for b in self.b:
                    self.r_iab[i, a, b] = self.model.addVar(vtype=GRB.BINARY, name=f'r_{i, a, b}')
            for k in self.I:
                self.x_ikp[i, k] = self.model.addVar(vtype=GRB.BINARY, name=f'x^p_{i, k}')
                self.z_ikp[i, k] = self.model.addVar(vtype=GRB.BINARY, name=f'z^p_{i, k}')
                self.h_ik[i, k] = self.model.addVar(vtype=GRB.BINARY, name=f'h_{i, k}')
                self.s_ik[i, k] = self.model.addVar(vtype=GRB.BINARY, name=f's_{i, k}')
                self.eta1_ik[i, k] = self.model.addVar(vtype=GRB.BINARY, name=f'eta^1_{i, k}')
                self.eta3_ik[i, k] = self.model.addVar(vtype=GRB.BINARY, name=f'eta^3_{i, k}')
                self.o_ik[i, k] = self.model.addVar(vtype=GRB.BINARY, name=f'o_{i, k}')
                self.nu_ik[i, k] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'nu_{i, k}')
                self.m_ik[i, k] = self.model.addVar(vtype=GRB.BINARY, name=f'm_{i, k}')
                for l in self.L:
                    self.beta_lik[i, k, l] = self.model.addVar(vtype=GRB.BINARY, name=f'beta^{l}_{i, k}')

        self.model.update()

    def define_obj(self):
        # Define Objective Function
        obj = LinExpr()

        for j in self.B:
            obj += self.C_j[j] * self.u_j[j]

        self.model.setObjective(obj, GRB.MINIMIZE)
        self.model.update()
        print("Objective Function Defined")
        print(self.model.getObjective(), '\n')

    def geometric_constraints(self):
        for j in self.B:
            self.model.addConstr(quicksum(self.p_ij[i, j] for i in self.I) <= self.M * self.u_j[j], name=f'GC1_{j}')
        for i in self.I:
            self.model.addConstr(quicksum(self.p_ij[i, j] for j in self.B) == 1, name=f'GC2_{i}')
            self.model.addConstr(self.xp_i[i] <= quicksum(self.L_j[j] * self.p_ij[i, j] for j in self.B),
                                 name=f'GC3_{i}')
            self.model.addConstr(self.zp_i[i] <= quicksum(self.H_j[j] * self.p_ij[i, j] for j in self.B),
                                 name=f'GC4_{i}')
            self.model.addConstr(
                self.xp_i[i] - self.x_i[i] == self.r_iab[i, 1, 1] * self.l_i[i] + self.r_iab[i, 1, 3] * self.h_i[i],
                name=f'GC5_{i}')
            self.model.addConstr(
                self.zp_i[i] - self.z_i[i] == self.r_iab[i, 3, 1] * self.l_i[i] + self.r_iab[i, 3, 3] * self.h_i[i],
                name=f'GC6_{i}')
            for b in self.b:
                self.model.addConstr(quicksum(self.r_iab[i, a, b] for a in self.a) == 1, name=f'GC7_{i, b}')
            for a in self.a:
                self.model.addConstr(quicksum(self.r_iab[i, a, b] for b in self.b) == 1, name=f'GC8_{i, a}')

    def overlap_constraints(self):
        for i in self.I:
            for k in self.I:
                if i == k:
                    continue
                for j in self.B:
                    self.model.addConstr(
                        self.x_ikp[i, k] + self.x_ikp[k, i] + self.z_ikp[i, k] + self.z_ikp[k, i] >= self.p_ij[i, j] +
                        self.p_ij[k, j] - 1, name=f'OvC1_{i, k, j}')
                self.model.addConstr(self.xp_i[k] <= self.x_i[i] + (1 - self.x_ikp[i, k]) * self.L_max,
                                     name=f'OvC2_{i, k}')
                self.model.addConstr(self.x_i[i] + 1 <= self.xp_i[k] + self.x_ikp[i, k] * self.L_max,
                                     name=f'OvC3_{i, k}')
                self.model.addConstr(self.zp_i[k] <= self.z_i[i] + (1 - self.z_ikp[i, k]) * self.H_max,
                                     name=f'OvC4_{i, k}')

    def orientation_constraints(self):
        for i in self.I:
            self.model.addConstr(self.r_iab[i, 3, 1] <= self.r_plus_i[i], name=f'OrC1_{i}')
            self.model.addConstr(
                quicksum(quicksum(self.beta_lik[i, j, k] for j in self.I) for k in self.L) + 2 * self.g_i[i] == 2,
                name=f'OrC2_{i}')
            self.model.addConstr(self.z_i[i] <= (1 - self.g_i[i]) * self.H_max, name=f'OrC3_{i}')
            for k in self.I:
                self.model.addConstr(self.zp_i[k] - self.z_i[i] <= self.nu_ik[i, k], name=f'OrC4_{i, k}')
                self.model.addConstr(self.z_i[k] - self.zp_i[k] <= self.nu_ik[i, k], name=f'OrC5_{i, k}')
                self.model.addConstr(
                    self.nu_ik[i, k] <= self.zp_i[k] - self.z_i[i] + 2 * self.H_max * (1 - self.m_ik[i, k]),
                    name=f'OrC6_{i, k}')
                self.model.addConstr(self.nu_ik[i, k] <= self.z_i[i] - self.zp_i[k] + 2 * self.H_max * self.m_ik[i, k],
                                     name=f'OrC7_{i, k}')
                self.model.addConstr(self.h_ik[i, k] <= self.nu_ik[i, k], name=f'OrC8_{i, k}')
                self.model.addConstr(self.nu_ik[i, k] <= self.h_ik[i, k] * self.H_max, name=f'OrC9_{i, k}')
                self.model.addConstr(self.o_ik[i, k] == self.x_ikp[i, k] + self.x_ikp[k, i], name=f'OrC10_{i, k}')
                self.model.addConstr((1 - self.s_ik[i, k]) == self.h_ik[i, k] + self.o_ik[i, k], name=f'OrC11_{i, k}')
                for j in self.B:
                    self.model.addConstr(self.p_ij[i, j] - self.p_ij[k, j] <= 1 - self.s_ik[i, k],
                                         name=f'OrC12_{i, k, j}')
                    self.model.addConstr(self.p_ij[k, j] - self.p_ij[i, j] <= 1 - self.s_ik[i, k],
                                         name=f'OrC13_{i, k, j}')
                for l in self.L:
                    self.model.addConstr(self.beta_lik[i, k, l] <= self.s_ik[i, k], name=f'OrC14_{i, k, l}')
                self.model.addConstr(self.eta1_ik[i, k] <= 1 - self.beta_lik[i, k, 1], name=f'OrC15_{i, k, 1}')
                self.model.addConstr(self.eta3_ik[i, k] <= 1 - self.beta_lik[i, k, 3], name=f'OrC16_{i, k, 3}')
                self.model.addConstr(self.x_i[k] <= self.x_i[i] + self.eta1_ik[i, k] * self.L_max, name=f'OrC17_{i, k}')
                self.model.addConstr(self.xp_i[i] <= self.xp_i[k] + self.eta3_ik[i, k] * self.L_max, name=f'OrC18_{i, k}')

    def fragility_perishability_radioactivity_constraints(self):
        for k in self.I:
            self.model.addConstr(quicksum(self.s_ik[i, k] for i in self.I) <= self.n * (1 - self.f_i[k]))
        for j in self.B:
            for i in self.I:
                self.model.addConstr(self.Rho_j[j] >= self.rho_i[i] * self.p_ij[i, j], name=f'PerC_{i, j}')
                self.model.addConstr(self.Phi_j[j] >= self.phi_i[i] * self.p_ij[i, j], name=f'RadC_{i, j}')
            self.model.addConstr(self.Rho_j[j] + self.Phi_j[j] <= 1, name=f'PRC_{j}')

    def build_model(self):
        self.setup_variables()
        self.define_obj()
        self.geometric_constraints()
        self.overlap_constraints()
        self.orientation_constraints()
        self.fragility_perishability_radioactivity_constraints()
        self.model.update()

    def run_model(self, time_limit=None):
        if time_limit is None:
            pass
        elif type(time_limit) is int:
            self.model.setParam('Timelimit', time_limit)  # Set Timeout limit to 15 minutes
        else:
            raise TypeError("Incorrect type passed. Time limit should be given as an integer")

        self.model.optimize()
        status = self.model.status

        if status == GRB.Status.UNBOUNDED:
            print('The model cannot be solved because it is unbounded')

        elif status == GRB.INFEASIBLE:
            self.model.computeIIS()
            print('\nThe following constraint(s) cannot be satisfied:')
            for c in self.model.getConstrs():
                if c.IISConstr:
                    print('%s' % c.ConstrName)

        elif status == GRB.Status.OPTIMAL or True:
            f_objective = self.model.objVal
            print('***** RESULTS ******')
            print('\nObjective Function Value: \t %g' % f_objective)

        elif status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
            print('Optimization was stopped with status %d' % status)

    def write_output(self):

        now = datetime.now()

        bins_used = []
        for j in self.B:
            if self.u_j[j].X == 1:
                bins_used.append(j)
        print(bins_used)

        with open(f'results/bins_used_{now}.pickle', 'wb') as handle:
            pickle.dump(bins_used, handle, protocol=pickle.HIGHEST_PROTOCOL)

        Items_in_Bin = dict()
        for j in bins_used:
            items_lst = []
            for i in self.I:
                if self.p_ij[i, j].X == 1:
                    items_lst.append(i)
            Items_in_Bin[j] = items_lst
        print(Items_in_Bin)

        with open(f'results/Items_in_Bin_{now}.pickle', 'wb') as handle:
            pickle.dump(Items_in_Bin, handle, protocol=pickle.HIGHEST_PROTOCOL)

        I_info_solution = dict()
        for i in self.I:
            info = [self.x_i[i].X, self.z_i[i].X, self.l_i[i], self.h_i[i]]
            I_info_solution[i] = info
        print(I_info_solution)

        with open(f'results/I_info_solution_{now}.pickle', 'wb') as handle:
            pickle.dump(I_info_solution, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    run = TwoD_BPP(subset=8)
    run.build_model()
    run.run_model(time_limit=1800)
    run.write_output()
