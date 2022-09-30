import pandas as pd
import numpy as np
import random
from numpy import linalg as LA
import matplotlib.pyplot as plt


class QSystemState:
    def __init__(self, ph_spin_up, ph_spin_down, 
            el_f0_spin_up_cnt, el_f0_spin_down_cnt,
            el_f1_spin_up_cnt, el_f1_spin_down_cnt) -> None:
        self.ph_spin_up = ph_spin_up
        self.ph_spin_down = ph_spin_down
        self.el_f0_spin_up_cnt = el_f0_spin_up_cnt
        self.el_f0_spin_down_cnt = el_f0_spin_down_cnt
        self.el_f1_spin_up_cnt = el_f1_spin_up_cnt
        self.el_f1_spin_down_cnt = el_f1_spin_down_cnt

    def __repr__(self) -> str:
        return (f"{self.ph_spin_up}{self.ph_spin_down}"
                f"{self.el_f0_spin_up_cnt}{self.el_f0_spin_down_cnt}"
                f"{self.el_f1_spin_up_cnt}{self.el_f1_spin_down_cnt}")
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False
    
    def is_possible(self) -> bool:
        return self.el_f0_spin_down_cnt <= 1 and self.el_f0_spin_up_cnt <= 1 and \
            self.el_f1_spin_down_cnt <= 1 and self.el_f1_spin_up_cnt <= 1 and \
            self.el_f0_spin_down_cnt + self.el_f0_spin_up_cnt + \
                self.el_f1_spin_down_cnt + self.el_f1_spin_up_cnt == 2 and \
            self.ph_spin_down <= 1 and self.ph_spin_up <= 1
    
    def quantum_state(self, sorted_states_list):
        if str(self) not in sorted_states_list:
            return None
        pos = sorted_states_list.index(str(self))
        n = len(sorted_states_list)
        result = np.zeros(n)
        result[pos] = 1
        return result

    def gives_moleculo(self):
        return self.el_f0_spin_down_cnt == 1 and self.el_f0_spin_up_cnt == 1
            

def generate_qsystem_states(max_photon_cnt, max_f_el_cnt):
    already_found = set()
    for cur_ph_cnt in range(max_photon_cnt + 1):
        for cur_spin_down_ph_cnt in range(cur_ph_cnt + 1):
            for cur_f0_el in range(max_f_el_cnt + 1):
                for cur_f0_spin_down_el in range(cur_f0_el + 1):
                    for cur_f0_spin_up_el in range(cur_f0_el - cur_f0_spin_down_el + 1):
                        for cur_f1_el in range(max_f_el_cnt + 1):
                            for cur_f1_spin_down_el in range(cur_f1_el + 1):
                                for cur_f1_spin_up_el in range(cur_f1_el - 
                                        cur_f1_spin_down_el + 1):
                                    cur_state = \
                                        QSystemState(cur_ph_cnt - cur_spin_down_ph_cnt, cur_spin_down_ph_cnt,
                                            cur_f0_spin_up_el, cur_f0_spin_down_el,
                                            cur_f1_spin_up_el, cur_f1_spin_down_el)
                                    if cur_state.is_possible() and str(cur_state) not \
                                            in already_found:
                                        already_found.add(str(cur_state))
                                        yield cur_state


def hamming_distance(state1: QSystemState, state2: QSystemState):
    s1 = str(state1)
    s2 = str(state2)
    ans = 0
    for i in range(min(len(s1), len(s2))):
        if s1[i] != s2[i]:
            ans += 1
    return ans


def transition_energy(state1: QSystemState, state2: QSystemState) -> np.cdouble:
    #if hamming_distance(state1, state2) > 3:
    #    return 0
    global h_planc
    global w_omega
    global g_const

    #if abs(state1.ph_spin_down - state2.ph_spin_down) > 1 or \
    #        abs(state1.ph_spin_up - state2.ph_spin_up) > 1 or \
    #        abs(state1.el_f0_spin_down_cnt - state2.el_f0_spin_down_cnt) > 1 or \
    #        abs(state1.el_f0_spin_up_cnt - state2.el_f0_spin_up_cnt) > 1 or \
    #        abs(state1.el_f1_spin_down_cnt - state2.el_f1_spin_down_cnt) > 1 or \
    #        abs(state1.el_f1_spin_up_cnt - state2.el_f1_spin_up_cnt) > 1:
    #    return 0

    if state1 == state2:
        # nothing changed
        return h_planc * w_omega * (state1.ph_spin_down + state1.ph_spin_up + 
            state1.el_f1_spin_down_cnt + state1.el_f1_spin_up_cnt)
    elif state1.ph_spin_down == state2.ph_spin_down and \
            state1.el_f0_spin_down_cnt == state2.el_f0_spin_down_cnt and \
            state1.el_f1_spin_down_cnt == state2.el_f1_spin_down_cnt:
        # spin up changed
        ph_change = state1.ph_spin_up - state2.ph_spin_up
        f0_change = state1.el_f0_spin_up_cnt - state2.el_f0_spin_up_cnt
        f1_change = state1.el_f1_spin_up_cnt - state2.el_f1_spin_up_cnt
        if (not (f0_change == -f1_change and ph_change * f0_change > 0)):
            return 0
        return g_const * (max(state1.ph_spin_up, state2.ph_spin_up) ** 0.5)
    elif state1.ph_spin_up == state2.ph_spin_up and \
            state1.el_f0_spin_up_cnt == state2.el_f0_spin_up_cnt and \
            state1.el_f1_spin_up_cnt == state2.el_f1_spin_up_cnt:
        # spin down changed
        ph_change = state1.ph_spin_down - state2.ph_spin_down
        f0_change = state1.el_f0_spin_down_cnt - state2.el_f0_spin_down_cnt
        f1_change = state1.el_f1_spin_down_cnt - state2.el_f1_spin_down_cnt
        if (not (f0_change == -f1_change and ph_change * f0_change > 0)):
            return 0
        return g_const * (max(state1.ph_spin_down, state2.ph_spin_down) ** 0.5)
    else:
        return 0


def get_probability(quantum_state: np.array, final_state_array: np.array) -> float:
    filtered_array = final_state_array[:, 0] * quantum_state[:, 0]
    filtered_array = np.abs(filtered_array)
    filtered_array **= 2
    return float(filtered_array.sum())


max_photon_cnt = 2
max_f_el_cnt = 2

h_planc = 1#3e-27
w_omega = 100#1e10
g_const = 8e-2#1e-30
t_delta = 1e-4
t_max = 100

valid_states_list = list(generate_qsystem_states(max_photon_cnt, max_f_el_cnt))
valid_states_list = sorted(valid_states_list, key=lambda x: str(x))
n = len(valid_states_list)
print(f"Valid states cnt: {n}")

matrix_array = []

for i in range(len(valid_states_list)):
    cur_row = []
    for j in range(len(valid_states_list)):
        cur_row.append(transition_energy(valid_states_list[i], valid_states_list[j]))
    matrix_array.append(cur_row)

matrix = np.array(matrix_array).astype(np.cdouble)
pd.DataFrame(matrix).to_excel("./matrix.xlsx")

sorted_str_states = sorted(map(str, list(valid_states_list)))
gives_moleculo_list = [el for el in valid_states_list if el.gives_moleculo()]
gives_moleculo_filter = np.add.reduce(list(map(lambda x: x.quantum_state(sorted_str_states), gives_moleculo_list))).astype(np.cdouble).reshape((-1, 1))

# test 0
latest_quantum_state = 0.5 * QSystemState(0, 0, 0, 1, 1, 0).quantum_state(sorted_str_states).astype(np.cdouble).reshape((-1, 1)) - \
        0.5 * QSystemState(0, 0, 1, 1, 0, 0).quantum_state(sorted_str_states).astype(np.cdouble).reshape((-1, 1)) + \
        0.5 * QSystemState(0, 0, 0, 0, 1, 1).quantum_state(sorted_str_states).astype(np.cdouble).reshape((-1, 1)) - \
        0.5 * QSystemState(0, 0, 1, 0, 0, 1).quantum_state(sorted_str_states).astype(np.cdouble).reshape((-1, 1))
# test 1 - must be a sin
#latest_quantum_state = QSystemState(0, 0, 1, 0, 0, 1).quantum_state(sorted_str_states).astype(np.cdouble).reshape((-1, 1))
# test 2 - must be a const
#latest_quantum_state = QSystemState(0, 0, 1, 1, 0, 0).quantum_state(sorted_str_states).astype(np.cdouble).reshape((-1, 1))
latest_quantum_state_copy = np.copy(latest_quantum_state)

eigenvalues_array, eigenvectors_array = LA.eig(matrix)
lambdas_list = []
for i in range(len(eigenvalues_array)):
    #eigenvectors_array[:,i] = eigenvectors_array[:,i] / np.linalg.norm(np.abs(eigenvectors_array[:,i]))
    lambdas_list.append(eigenvectors_array[:,i].dot(latest_quantum_state[:,0]))

cur_t = 0

t_list = []
approx_solution_list = []
exact_solution_list = []
latest_percent_done = -1
while cur_t < t_max:
    # approx solution
    latest_quantum_state = (np.identity(n).astype(np.cdouble) - ((1j * t_delta / h_planc) * matrix) + ((1j * t_delta / h_planc / 2) * np.sqaure(matrix))).dot(latest_quantum_state)
    cur_vector_modulus = np.linalg.norm(np.abs(latest_quantum_state[:,0]))
    latest_quantum_state = latest_quantum_state / cur_vector_modulus
    approx_probability = get_probability(latest_quantum_state, gives_moleculo_filter)
    # exact solution
    exact_quantum_state = np.zeros(n).astype(np.cdouble)
    for i in range(len(lambdas_list)):
        exact_quantum_state = exact_quantum_state + ((lambdas_list[i] * 
            np.exp(-1j / h_planc * cur_t * eigenvalues_array[i])) * eigenvectors_array[:,i])
    exact_state_vector_modulus = np.linalg.norm(np.abs(exact_quantum_state))
    exact_quantum_state = exact_quantum_state / exact_state_vector_modulus
    exact_probability = get_probability(exact_quantum_state.reshape((-1, 1)), gives_moleculo_filter)
    # print
    if round(cur_t * 100 / t_max) % 10 == 0 and round(cur_t * 100 / t_max) != latest_percent_done:
        print(f"Done {round(cur_t * 100 / t_max)}%", cur_t, approx_probability, exact_probability)
        latest_percent_done = round(cur_t * 100 / t_max)
    approx_solution_list.append(approx_probability)
    exact_solution_list.append(exact_probability)
    t_list.append(cur_t)
    cur_t += t_delta

plt.plot(t_list, approx_solution_list, 'r', label="Approx Solution")
plt.plot(t_list, exact_solution_list, 'g', label="Exact Solution")
plt.title("Moleculo Appearance Probability")
plt.xlabel("Time")
plt.ylabel("Probability")
plt.legend(loc="lower right")
plt.show()
