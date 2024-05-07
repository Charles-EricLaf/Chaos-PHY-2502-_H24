import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from scipy import optimize
from scipy.integrate import solve_ivp

# Définition des constantes utilisées
kf = 28  # uM/s
Vp = 1.2  # uM/s
R5 = 1.6  # uM
R3 = 50  # uM
r2 = 100  # 1/uMs
R1 = 6  # uM
leak = 0.2  # uM/s
Kp = 0.18  # uM
k2 = 26.5  # uM/s
k1 = 44  # uM/s
r4 = 20  # 1/s
k3 = 1.6  # uM/s


# Fonctions de bases
def phi_1(conc):
    """
        Calcul du taux phi_1, en fonction de la concentration du calcium.
    Args :
        Conc : Concentration de calcium
    Returns :
        0 : Fonction
        1 : Dérivée
    """
    return (r2*conc)/(R1+conc), (r2*R1) / (R1+conc)**2

def phi_m1(conc):
    """
        Calcul du taux phi_-1, en fonction de la concentration du calcium.
    Args :
        Conc : Concentration de calcium
    Returns :
        0 : Fonction
        1 : Dérivée
    """
    return k1/(R3+conc), -k1 / (R3+conc)**2

def phi_2(conc):
    """
        Calcul du taux phi_2, en fonction de la concentration du calcium.
    Args :
        Conc : Concentration de calcium
    Returns :
        0 : Fonction
        1 : Dérivée
    """
    return (k2 + r4*conc)/(R3+conc), (R3*r4 - k2) / (R3+conc)**2

def phi_3(conc):
    """
        Calcul du taux phi_3, en fonction de la concentration du calcium.
    Args :
        Conc : Concentration de calcium
    Returns :
        0 : Fonction
        1 : Dérivée
    """
    return k3/(R5+conc), -k3 / (R5+conc)**2

def pump(conc):
    """
        Calcul du taux de calcium étant pompé dans la cellule.
    Args :
        Conc : Concentration de calcium
    Returns :
        0 : Fonction
        1 : Dérivée
    """
    return (Vp * conc**2)/(Kp**2 + conc**2), (2*Vp * Kp**2 * conc) / (Kp**2 + conc**2)**2

# Fonctions combinées
def recept(p, h, conc):
    """
        Calcul du taux de calcium qui entre dans la cellule par un récepteur InsP3 du réticulum endoplasmique.
    Args :
        p : Concentration d'InsP3
        h : Fraction des récepteurs ouverts ou fermés
        Conc : Concentration de calcium
    Returns :
        0 : Fonction
        1 : Dérivée p/r conc
        2 : Dérivée p/r h
    """
    paren = (p*h*phi_1(conc)[0]) / (phi_1(conc)[0]*p + phi_m1(conc)[0])
    fonc = kf * paren**4
    d_dc = 4 * kf * paren**3 * ((p*h * (phi_m1(conc)[0]*phi_1(conc)[1] - phi_1(conc)[0]*phi_m1(conc)[1])) / (phi_1(conc)[0]*p + phi_m1(conc)[0])**2)
    d_dh = 4 * kf * paren**3 * (p*phi_1(conc)[0]) / (phi_1(conc)[0]*p + phi_m1(conc)[0])
    return fonc, d_dc, d_dh

def InsP3(p, h, conc):
    """
        Équation différentielle d'ordre 1 décrivant la fraction de récepteurs InsP3 ouverts ou fermés.
    Args :
        p : Concentration d'InsP3
        h : Fraction des récepteurs ouverts ou fermés
        Conc : Concentration de calcium
    Returns :
        0 : Fonction
        1 : Dérivée p/r conc
        2 : Dérivée p/r h
    """
    fonc = phi_3(conc)[0]*(1-h) - ((phi_1(conc)[0]*phi_2(conc)[0]*h*p)/(phi_1(conc)[0]*p + phi_m1(conc)[0]))
    d_dc = phi_3(conc)[1]*(1-h) - ((h*p*(phi_m1(conc)[0]*phi_2(conc)[0]*phi_1(conc)[1] +
                                        (phi_1(conc)[0]**2)* phi_2(conc)[1]*p + phi_m1(conc)[0]*phi_1(conc)[0]*phi_2(conc)[1]
                                        - phi_1(conc)[0]*phi_2(conc)[0]*phi_m1(conc)[1])) / (phi_1(conc)[0]*p + phi_m1(conc)[0])**2)
    d_dh = - phi_3(conc)[0] - ((phi_1(conc)[0]*phi_2(conc)[0]*p) / (phi_1(conc)[0]*p + phi_m1(conc)[0]))
    return fonc, d_dc, d_dh

# Fonctions pour une seule cellule
def one_cell_conc(p, h, conc):
    """
        Équation différentielle d'ordre 1 décrivant la variation de concentration du calcium dans une seule cellule.
    Args :
        p : Concentration d'InsP3
        h : Fraction des récepteurs ouverts ou fermés
        Conc : Concentration de calcium
    Returns :
        0 : Fonction
        1 : Dérivée p/r conc
        2 : Dérivée p/r h
    """
    fonc = recept(p, h, conc)[0] - pump(conc)[0] + leak
    d_dc = recept(p, h, conc)[1] - pump(conc)[1]
    d_dh = recept(p, h, conc)[2]
    return fonc, d_dc, d_dh

def system_one_cell(t, y, p):
    """
        Système EDO d'ordre 1 décrivant la variation de concentration du calcium dans une seule cellule.
    Args :
        t : temps
        y : tuple (h, concentration)
        p : Concentration d'InsP3
    Returns :
        0 : Système
    """
    h, conc = y
    dc_dt = one_cell_conc(p, h, conc)[0]
    dh_dt = InsP3(p, h, conc)[0]
    return [dh_dt, dc_dt]

def points_fixes_one_cell(sol, p):
    '''
        Calcul des points fixes.
    Args :
        Sol : Solution du système d'EDO.
        p : Concentration d'InsP3.
    Returns :
        Points fixes.
    '''
    t_values = np.linspace(*t_span, 100000)
    x_values, y_values = sol.sol(t_values)
    dx_dt_values, dy_dt_values = system_one_cell(t_values, [x_values, y_values], p)
    indices_points_fixes = np.where((np.abs(dx_dt_values) < 1e-11) & (np.abs(dy_dt_values) < 1e-11))
    return x_values[indices_points_fixes], y_values[indices_points_fixes]

def points_one_cell(sol, p):
    '''
        Calcul des points.
    Args :
        Sol : Solution du système d'EDO.
        p : Concentration d'InsP3.
    Returns :
        Points (t, x, y).
    '''
    t_values = np.linspace(*t_span, 100000)
    x_values, y_values = sol.sol(t_values)
    # dx_dt_values, dy_dt_values = system(t_values, [x_values, y_values], p)
    return t_values, x_values, y_values

# Fonctions pour deux cellules
def diff_two_cells(D, conc1, conc2):
    """
        Calcul du taux de calcium étant pompé diffusé entre les deux .
    Args :
        D : Constante de proportionnalité, si 0 --> cellules sans interaction
        Conc1 : Concentration de calcium dans la cellule 1
        Conc2 : Concentration de calcium dans la cellule 2
    Returns :
        0 : Fonction
        1 : Dérivée p/r conc1
        2 : Dérivée p/r conc2
    """
    fonc =  D*(conc2-conc1)
    d_dc1 = -D
    d_dc2 = D
    return fonc, d_dc1, d_dc2

def two_cells_conc(p, h, conc1, conc2, D):
    """
        Équation différentielle d'ordre 1 décrivant la variation de concentration du calcium dans une seule cellule.
    Args :
        p : Concentration d'InsP3
        h : Fraction des récepteurs ouverts ou fermés
        Conc1 : Concentration de calcium dans la cellule 1
        Conc2 : Concentration de calcium dans la cellule 2
        D : Constante de proportionnalité, si 0 --> cellules sans interaction
    Returns :
        0 : Fonction
        1 : Dérivée p/r conc1
        2 : Dérivée p/r h
        3 : Dérivée p/r conc2
    """
    fonc = recept(p, h, conc1)[0] - pump(conc1)[0] + leak + diff_two_cells(D, conc1, conc2)[0]
    d_dc1 = recept(p, h, conc1)[1] - pump(conc1)[1] + diff_two_cells(D, conc1, conc2)[1]
    d_dh = recept(p, h, conc1)[2]
    d_dc2 = diff_two_cells(D, conc1, conc2)[2]
    return fonc, d_dc1, d_dh, d_dc2

def system_two_cell(t, y, p):
    """
        Système EDO d'ordre 1 décrivant la variation de concentration du calcium dans une seule cellule.
    Args :
        t : temps
        y : tuple (h1, concentration1, h2, concentration2)
        p : Concentration d'InsP3
    Returns :
        0 : Système
    """
    h1, conc1, h2, conc2 = y
    dc_dt1 = two_cells_conc(p, h1, conc1, conc2, D)[0]
    dh_dt1 = InsP3(p, h1, conc1)[0]
    dc_dt2 = two_cells_conc(p, h2, conc2, conc1, D)[0]
    dh_dt2 = InsP3(p, h2, conc2)[0]
    return [dh_dt1, dc_dt1, dh_dt2, dc_dt2]

def points_fixes_two_cell(sol, p):
    '''
        Calcul des points fixes.
    Args :
        Sol : Solution du système d'EDO.
        p : Concentration d'InsP3.
    Returns :
        Points fixes.
    '''
    t_values = np.linspace(*t_span, 100000)
    x1_values, y1_values, x2_values, y2_values = sol.sol(t_values)
    dx_dt1_values, dy_dt1_values, dx_dt2_values, dy_dt2_values = system_two_cell(t_values, [x1_values, y1_values, x2_values, y2_values], p)
    indices_points_fixes = np.where((np.abs(dx_dt1_values) < 1e-11) & (np.abs(dy_dt1_values) < 1e-11) & (np.abs(dx_dt2_values) < 1e-11) & (np.abs(dy_dt2_values) < 1e-11))
    return x1_values[indices_points_fixes], y1_values[indices_points_fixes], x2_values[indices_points_fixes], y2_values[indices_points_fixes]

def points_two_cell(sol):
    '''
        Calcul des points.
    Args :
        Sol : Solution du système d'EDO.
        p : Concentration d'InsP3.
    Returns :
        Points (temps, h1, conc1, h2, conc2)
    '''
    t_values = np.linspace(*t_span, 100000)
    x1_values, y1_values, x2_values, y2_values = sol.sol(t_values)
    # dx_dt1_values, dy_dt1_values, dx_dt2_values, dy_dt2_values = system_two_cell(t_values, [x1_values, y1_values, x2_values, y2_values], p)
    return t_values, x1_values, y1_values, x2_values, y2_values

def diff_three_cells(D12, D13, conc1, conc2, conc3):
    """
        Calcul du taux de calcium étant pompé diffusé entre les deux .
    Args :
        D : Constante de proportionnalité, si 0 --> cellules sans interaction
        Conc1 : Concentration de calcium dans la cellule 1
        Conc2 : Concentration de calcium dans la cellule 2
    Returns :
        0 : Fonction
        1 : Dérivée p/r conc1
        2 : Dérivée p/r conc2
        3 : Dérivée p/r conc3
    """
    fonc =  D12*(conc2-conc1) + D13*(conc3-conc1)
    d_dc1 = -D12 - D13
    d_dc2 = D12
    d_dc3 = D13
    return fonc, d_dc1, d_dc2, d_dc3

def three_cells_conc(p, h, conc1, conc2, conc3, D12, D13):
    """
        Équation différentielle d'ordre 1 décrivant la variation de concentration du calcium dans une seule cellule.
    Args :
        p : Concentration d'InsP3
        h : Fraction des récepteurs ouverts ou fermés
        Conc1 : Concentration de calcium dans la cellule 1
        Conc2 : Concentration de calcium dans la cellule 2
        Conc3 : Concentration de calcium dans la cellule 3
        D : Constante de proportionnalité, si 0 --> cellules sans interaction
    Returns :
        0 : Fonction
        1 : Dérivée p/r conc1
        2 : Dérivée p/r h
        3 : Dérivée p/r conc2
    """
    fonc = recept(p, h, conc1)[0] - pump(conc1)[0] + leak + diff_three_cells(D12, D13, conc1, conc2, conc3)[0]
    d_dc1 = recept(p, h, conc1)[1] - pump(conc1)[1] + diff_three_cells(D12, D13, conc1, conc2, conc3)[1]
    d_dh = recept(p, h, conc1)[2]
    d_dc2 = diff_three_cells(D12, D13, conc1, conc2, conc3)[2]
    d_dc3 = diff_three_cells(D12, D13, conc1, conc2, conc3)[3]
    return fonc, d_dc1, d_dh, d_dc2, d_dc3

def system_three_cell(t, y, p):
    """
        Système EDO d'ordre 1 décrivant la variation de concentration du calcium dans une seule cellule.
    Args :
        t : temps
        y : tuple (h1, concentration1, h2, concentration2, h3, concentration3)
        p : Concentration d'InsP3
    Returns :
        0 : Système
    """
    h1, conc1, h2, conc2, h3, conc3 = y
    dc_dt1 = three_cells_conc(p, h1, conc1, conc2, conc3, D12, D13)[0]
    dh_dt1 = InsP3(p, h1, conc1)[0]
    dc_dt2 = three_cells_conc(p, h2, conc2, conc1, conc3, D12, D13)[0]
    dh_dt2 = InsP3(p, h2, conc2)[0]
    dc_dt3 = three_cells_conc(p, h3, conc3, conc2, conc1, D12, D13)[0]
    dh_dt3 = InsP3(p, h3, conc3)[0]
    return [dh_dt1, dc_dt1, dh_dt2, dc_dt2, dh_dt3, dc_dt3]

def points_fixes_three_cell(sol, p):
    '''
        Calcul des points fixes.
    Args :
        Sol : Solution du système d'EDO.
        p : Concentration d'InsP3.
    Returns :
        Points fixes.
    '''
    t_values = np.linspace(*t_span, 100000)
    x1_values, y1_values, x2_values, y2_values, x3_values, y3_values = sol.sol(t_values)
    dx_dt1_values, dy_dt1_values, dx_dt2_values, dy_dt2_values, dx_dt3_values, dy_dt3_values = system_three_cell(t_values,
                                                                                            [x1_values, y1_values, x2_values, y2_values, x3_values, y3_values], p)
    indices_points_fixes = np.where((np.abs(dx_dt1_values) < 1e-11) & (np.abs(dy_dt1_values) < 1e-11) & (np.abs(dx_dt2_values) < 1e-11) &
                                    (np.abs(dy_dt2_values) < 1e-11) & (np.abs(dx_dt3_values) < 1e-11) & (np.abs(dy_dt3_values) < 1e-11))
    return (x1_values[indices_points_fixes], y1_values[indices_points_fixes], x2_values[indices_points_fixes],
             y2_values[indices_points_fixes], x3_values[indices_points_fixes], y3_values[indices_points_fixes])



def points_three_cell(sol):
    '''
        Calcul des points.
    Args :
        Sol : Solution du système d'EDO.
        p : Concentration d'InsP3.
    Returns :
        Points (temps, h1, conc1, h2, conc2)
    '''
    t_values = np.linspace(*t_span_normal, 1000)
    x1_values, y1_values, x2_values, y2_values, x3_values, y3_values = sol.sol(t_values)
    # dx_dt1_values, dy_dt1_values, dx_dt2_values, dy_dt2_values = system_two_cell(t_values, [x1_values, y1_values, x2_values, y2_values], p)
    return t_values, x1_values, y1_values, x2_values, y2_values, x3_values, y3_values


def three_cells_conc_perturbation(p, h, conc1, conc2, conc3, D12, D13, pert):
    """
        Équation différentielle d'ordre 1 décrivant la variation de concentration du calcium dans une seule cellule.
    Args :
        p : Concentration d'InsP3
        h : Fraction des récepteurs ouverts ou fermés
        Conc1 : Concentration de calcium dans la cellule 1
        Conc2 : Concentration de calcium dans la cellule 2
        Conc3 : Concentration de calcium dans la cellule 3
        D : Constante de proportionnalité, si 0 --> cellules sans interaction
    Returns :
        0 : Fonction
        1 : Dérivée p/r conc1
        2 : Dérivée p/r h
        3 : Dérivée p/r conc2
    """

    fonc = recept(p, h, conc1)[0] - pump(conc1)[0] + leak + diff_three_cells(D12, D13, conc1, conc2, conc3)[0] + pert
    d_dc1 = recept(p, h, conc1)[1] - pump(conc1)[1] + diff_three_cells(D12, D13, conc1, conc2, conc3)[1]
    d_dh = recept(p, h, conc1)[2]
    d_dc2 = diff_three_cells(D12, D13, conc1, conc2, conc3)[2]
    d_dc3 = diff_three_cells(D12, D13, conc1, conc2, conc3)[3]
    return fonc, d_dc1, d_dh, d_dc2, d_dc3

def system_three_cell_perturbation(t, y, p):
    """
        Système EDO d'ordre 1 décrivant la variation de concentration du calcium dans une seule cellule.
    Args :
        t : temps
        y : tuple (h1, concentration1, h2, concentration2, h3, concentration3)
        p : Concentration d'InsP3
    Returns :
        0 : Système
    """
    h1, conc1, h2, conc2, h3, conc3 = y
    dc_dt1 = three_cells_conc_perturbation(p, h1, conc1, conc2, conc3, D12, D13, pert)[0]
    dh_dt1 = InsP3(p, h1, conc1)[0]
    dc_dt2 = three_cells_conc(p, h2, conc2, conc1, conc3, D12, D13)[0]
    dh_dt2 = InsP3(p, h2, conc2)[0]
    dc_dt3 = three_cells_conc(p, h3, conc3, conc2, conc1, D12, D13)[0]
    dh_dt3 = InsP3(p, h3, conc3)[0]
    return [dh_dt1, dc_dt1, dh_dt2, dc_dt2, dh_dt3, dc_dt3]

def points_three_cell_perturbation(sol):
    '''
        Calcul des points.
    Args :
        Sol : Solution du système d'EDO.
        p : Concentration d'InsP3.
    Returns :
        Points (temps, h1, conc1, h2, conc2)
    '''
    t_values = np.linspace(start=timepoint, stop=timepoint+1, num=10) # vérifier
    x1_values, y1_values, x2_values, y2_values, x3_values, y3_values = sol.sol(t_values)
    # dx_dt1_values, dy_dt1_values, dx_dt2_values, dy_dt2_values = system_two_cell(t_values, [x1_values, y1_values, x2_values, y2_values], p)
    return t_values, x1_values, y1_values, x2_values, y2_values, x3_values, y3_values



# Résolution problème aux valeurs initiales
initial_conditions1 = [0.8, 0.2, 0.6, 0.5, 0.3, 0.8]  # Conditions initiales [# porte ouverte/fermée, conc Ca]
initial_conditions2 = [0.2, 0.8, 0.3, 0.3, 0.7, 0.3]
t_span = [0, 150]  # Plage de temps

# Limites du portrait de phase
xmax, ymax = 1.2, 1.2

# Meshgrid
h1, conc1 = np.meshgrid(np.linspace(0,xmax,400), np.linspace(0,ymax,400))
h2, conc2 = np.meshgrid(np.linspace(0,xmax,400), np.linspace(0,ymax,400))
h3, conc3 = np.meshgrid(np.linspace(0,xmax,400), np.linspace(0,ymax,400))

print("Début!")

# Paramètres de départ à la simulation

p = 0.27
D12 = 0.01
D13 = 0.01
D23 = 0.01
start = 50
end = 150
target = 0.5
# target_2 = 0.5
pert_time_values = np.arange(0, end+1, step=1)
pert_values = np.zeros((1, start))
pert_2 = 0
t_span_normal = [0, start]

# Trajectoire 1

sol = solve_ivp(system_three_cell, t_span_normal, initial_conditions1, dense_output=True, args=[p], method="Radau")
temps, h1_val, conc1_val, h2_val, conc2_val, h3_val, conc3_val = points_three_cell(sol)

# Équations différentielles perturbées

for timepoint in range(start, end+1):

    #  Target=2; p=0,27

    # if timepoint < start+5:
    #     pert = 1
    #
    # if start+5 <= timepoint < start+10:
    #     pert=0.8
    # else:
    #     pert=0.91

    # Target=0.5; p=0.27

    if -0.1 < target-conc1_val[-1] < -0.01:
        pert = 0

    elif target-conc1_val[-1] < -0.5:
        pert = target-conc1_val[-1]

    elif -0.5 < target-conc1_val[-1] <= -0.1:
        pert = -0.3

    elif 0.8 > target-conc1_val[-1] >= 0.5:
        pert = 0.5

    elif target-conc1_val[-1] >= 0.8:
        pert = 1

    elif 0.5 > target-conc1_val[-1] >= 0.1:
        pert = 0.2

    elif 0.1 > target-conc1_val[-1] > 0.01:
        pert = 0.01

    else :
        pert = -1*(conc1_val[-1]-conc1_val[-2])

    # Tentative de perturbations cellule 2

    # if -0.1 < target_2-conc2_val[-1] < -0.01:
    #     pert = 0
    #
    # elif target_2-conc2_val[-1] < -0.5:
    #     pert = target-conc2_val[-1]
    #
    # elif -0.5 < target_2-conc2_val[-1] <= -0.1:
    #     pert = -0.3
    #
    # elif 0.8 > target_2-conc2_val[-1] >= 0.5:
    #     pert = 0.5
    #
    # elif target_2-conc2_val[-1] >= 0.8:
    #     pert = 1
    #
    # elif 0.5 > target_2-conc2_val[-1] >= 0.1:
    #     pert = 0.2
    #
    # elif 0.1 > target_2-conc2_val[-1] > 0.01:
    #     pert = 0.01
    #
    # else :
    #     pert_2 = -1*(conc2_val[-1]-conc2_val[-2])

    # Perturbation supplémentaire cellule 2

    # if 160 >timepoint > 149:
    #     pert_2 = 0.7
    # else:
    #     pert_2 = 0

    # Target = 0.5, perturbation = -écart

    # pert = target-conc1_val[-1]

    print(pert_2)
    pert_values = np.append(pert_values, np.array([pert]))
    y_vector = [h1_val[-1], conc1_val[-1], h2_val[-1], conc2_val[-1], h3_val[-1], conc3_val[-1]]

    sol = solve_ivp(system_three_cell_perturbation, [timepoint, timepoint+1], y_vector,
                    dense_output=True, args=[p], method="Radau") # modifier conditions initiales
    temps_temp, h1_temp, conc1_temp, h2_temp, conc2_temp, h3_temp, conc3_temp = points_three_cell_perturbation(sol)

    temps = np.append(temps, temps_temp)
    h1_val = np.append(h1_val, h1_temp)
    conc1_val = np.append(conc1_val, conc1_temp)
    h2_val = np.append(h2_val, h2_temp)
    conc2_val = np.append(conc2_val, conc2_temp)
    h3_val = np.append(h3_val, h3_temp)
    conc3_val = np.append(conc3_val, conc3_temp)

    print(pert)
    print(timepoint)

fig, ax = plt.subplots(nrows=2, sharex=True)

ax[0].plot(temps, conc1_val, label=r"$c_1$", color="black", lw=2)
ax[0].set_xlim(0, end+1)
ax[0].set_ylim(0, 2.5)
plt.tick_params(direction="in")
ax[0].plot(temps, conc2_val, label=r"$c_2$", color="darkviolet", lw=2)
ax[0].plot(temps, conc3_val, label=r"$c_3$", color="teal", lw=2)
ax[0].axvline(x=start, linestyle=":", color="k")
ax[0].axhline(y=target, linestyle="--", color="k", label="Cible")
ax[0].legend(loc="upper right", fontsize=14)
ax[0].yaxis.set_tick_params(labelsize=16)
fig.text(0.06, 0.5, r'Concentration de Ca$^{2+}$ [$\mu$M]', ha='center', va='center', rotation='vertical', fontsize=18)

ax[1].plot(pert_time_values, pert_values, label="Perturbation", color="k")
ax[1].axvline(x=start, linestyle=":", color="k", label="Début perturbation")
ax[1].legend(loc="lower left", fontsize=14)
ax[1].set_xlim(0, end+1)
ax[1].set_xlabel('Temps [s]', fontsize=18)
ax[1].xaxis.set_tick_params(labelsize=16)
ax[1].yaxis.set_tick_params(labelsize=16)
plt.show()

print("Terminé!")
