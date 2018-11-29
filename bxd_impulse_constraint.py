""""
ABC constraint demonstration under both euler and velocity verlet integration.
Bond forces are simulated using the MM3 force field, no other forces are used.
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import namedtuple
import seaborn as sns

# set up sns as in thesis.
sns.set(font_scale=0.8, palette=sns.color_palette(), style='white', color_codes=False)

# bond constraints - defined as a named tuple
bond_constraint = namedtuple("bond_constraint", ["i", "j", "d", "active"])
# ABC constraint - defined as a named tuple
abc_constraint = namedtuple("abc_constraint", ["a", "b", "c", "norm", "d"])
# bond force - defined as a named tuple
bond = namedtuple("bond", ["i", "j", "k", "r0"])
# dimensionality
d = 2
natoms = 3


def vector(v, index):
    """
    From a flat list, extracts a given indexes' vector
    :param v: List
    :param index: The index to extract as a vector.
    :return: Vector of dimensionality d.
    """
    o = []
    for ii in range(d):
        o.append(v[index * d + ii])
    return np.array(o)


def KE(m, v):
    """
    Computes the kinetic energy. Assumes M is diagonal matrix, v is 1D array of velocities.
    """
    return 0.5 * np.sum(np.dot(m, v * v))


def LM(m, v):
    """
    Assumes M is a diagonal matrix, v is 1D array of velocities,
    Returns the momentum
    """

    lm = np.array([0.0, 0.0])
    natoms = int(len(v) / 2)
    for i in range(0, natoms):
        x = m[2 * i][2 * i] * vector(v, i)
        lm += x

    return (lm)


def F(r, bonds):
    """
    MM3 Force calculation.
    """

    # MM3 bond units
    bondUnit = 71.94
    bondcubic = -2.55
    bondquartic = 3.793125

    force = np.array([0.0] * d * natoms)
    V = 0.0
    for b in bonds:
        i = b.i
        j = b.j

        # distance in x and y dimension between atoms i and j
        dx = r[d * i + 0] - r[d * j + 0]
        dy = r[d * i + 1] - r[d * j + 1]
        # inter atomic distance between atoms i and j
        rad = math.sqrt(dx * dx + dy * dy)
        # displacement from equilibrium
        dr = rad - b.r0
        dr2 = dr * dr

        # MM3 gradient of bond
        V += bondUnit * b.k * dr2 * (1.0 + bondcubic * dr + bondquartic * dr2)
        dVdr = 2 * bondUnit * b.k * dr * (1 + 1.5 * bondcubic * dr + 2.0 * bondquartic * dr2)
        # chain rule derivatives, and apply forces
        dV = dVdr / rad
        force[d * i + 0] += float(-1.0 * dV * dx)
        force[d * i + 1] += float(-1.0 * dV * dy)
        force[d * j + 0] += float(dV * dx)
        index = d * j + 1
        result = dV * dy
        result = force[index] + result
        force[index] = result
    return force, V


def phi_bond(r, constraint):
    """
    Given atomic positions and a constraint, evaluates the bond constraint
    phi(r) >= 0 means distance greater than D.
    """
    i = constraint.i
    j = constraint.j
    rij = np.array([r[i * d + 0] - r[j * d + 0], r[i * d + 1] - r[j * d + 1]])
    diff = 0.5 * (np.dot(rij, rij) - constraint.d * constraint.d)
    return diff


def phi_ABC(r, constraint):
    """
    Evaluates ABC constraint. phi(r) >= 0 means distance greater than D
    from plane
    """
    a = constraint.a
    b = constraint.b
    c = constraint.c
    ra = vector(r, a)
    rb = vector(r, b)
    rc = vector(r, c)
    norm = constraint.norm
    d = constraint.d
    dist = np.linalg.norm(rb - ra) * norm[0] + np.linalg.norm(rc - rb) * norm[1] + d
    return dist


def phi_evaluate(r, constraint):
    """
    Given atomic positions and a constraint, calls the relevant method to
    evaluate it.
    """
    if type(constraint).__name__ is "bond_constraint":
        return phi_bond(r, constraint)
    if type(constraint).__name__ is "abc_constraint":
        return phi_ABC(r, constraint)
    else:
        raise ValueError("constraint type not recognised")


def bond_phi_grad(r, v, constraint):
    """
    Given atomic positions, velocities and a constraint, evaluates the gradient
    of said constraint.
    """
    i = constraint.i
    j = constraint.j
    diff = np.dot(r[i] - r[j], v[i] - v[j])
    return diff


def phi_grad(J, v):
    return np.dot(J, v)


def jacobian_bond(r, constraint):
    """
    Constructs the jacobian for constraint on bond A-B
    :param r: List of positions
    :param constraint: Constraint indices.
    :return:
    """
    i = constraint.i
    j = constraint.j
    J = []
    xi = r[i * d + 0]
    xj = r[j * d + 0]
    yi = r[i * d + 1]
    yj = r[j * d + 1]
    J.append((xi - xj))
    J.append((yi - yj))
    J.append((xj - xi))
    J.append((yj - yi))
    J = np.array(J)
    return J


def jacobian_ABC(r, constraint):
    """
    Construct the jacobian for constraint on bond A-B and B-C
    """
    a = constraint.a
    b = constraint.b
    c = constraint.c
    n1 = constraint.norm[0]
    n2 = constraint.norm[1]
    xa = r[a * d + 0]
    xb = r[b * d + 0]
    xc = r[c * d + 0]
    ya = r[a * d + 1]
    yb = r[b * d + 1]
    yc = r[c * d + 1]
    J = []
    rab = math.sqrt((xb - xa) * (xb - xa) + (yb - ya) * (yb - ya))
    rbc = math.sqrt((xc - xb) * (xc - xb) + (yc - yb) * (yc - yb))

    # construct Jacobian
    # atom A
    J.append(-n1 * (xb - xa) / rab)
    J.append(-n1 * (yb - ya) / rab)
    # atom B - linear combination of both variables
    J.append(n1 * (xb - xa) / rab - n2 * (xc - xb) / rbc)
    J.append(n1 * (yb - ya) / rab - n2 * (yc - yb) / rbc)
    # atom C
    J.append(n2 * (xc - xb) / rbc)
    J.append(n2 * (yc - yb) / rbc)

    return J


def jacobian(r, constraint):
    """
    Returns the jacobian for a given constraint.
    :param r: List of positions
    :param constraint: Constraint indices and type.
    :return:
    """
    if type(constraint).__name__ is "bond_constraint":
        return jacobian_bond(r, constraint)
    if type(constraint).__name__ is "abc_constraint":
        return jacobian_ABC(r, constraint)
    else:
        raise ValueError("constraint type not recognised")


def single_constraint(v, M_inv, J):
    """
    Solves for lambda for a single constraint
    """
    grad = phi_grad(J, v)
    b = grad
    lamda = - (grad + b) / (np.dot(J, np.dot(M_inv, np.transpose(J))))
    return lamda


def single_constraint_vv(r, v, f, M_inv, constraint, J):
    """
    More efficient solving for lambda,  evaluates phi and returns accordingly.
    :return: 0 if constraint is satisfied, lambda value otherwise.
    """
    tol = 0.0001
    lamda = 0.0
    ev = phi_evaluate(r, constraint)
    if ev <= tol:
        grad = phi_grad(J, v)
        b = grad
        lamda = - (grad + b) / (
            np.dot(J, np.dot(M_inv, np.transpose(J))))
    return lamda


# positions
r = np.array([])
# velocities
v = np.array([])

# topology - ABC system.
# atom 0 position
r = np.array([0.2, 0], dtype=np.float64)
# atom 1 position
r = np.append(r, [1.2, 0])
r = np.append(r, [2.0, 0.5])
# atom 0 velocity
v = np.array([4, 0], dtype=np.float64)
# atom 1 velocity
v = np.append(v, [-0.5, 0])
# atom 2 velocity
v = np.append(v, [0.0, 0.5])
# atom masses, arbitrarily set.
m = np.diagflat([4.2, 4.2, 2, 2, 10, 10])

f = np.array([0.0] * natoms * d)

# add constraint between ABC atoms
norm = np.array([-0.4, 1])
norm = norm / np.linalg.norm(norm)
print("Norm:", norm)
phi = abc_constraint(0, 1, 2, norm, -0.5)
# add bond force between atoms 0 and 1, 1 and 2
bonds = [bond(1, 2, 7.630, 0.9570), bond(0, 1, 7.630, 0.9470)]

nsteps = 1000
dt = 0.001
rho1_hist = []
rho2_hist = []
v_list = []
v2_list = []
v3_list = []
ke_list = []
lm_list = []
pe_list = []
tote_list = []
phi_list = []
step = 0
m_inv = np.linalg.inv(m)
V = 0
ke = 0
lm = 0
verlet = True
# perform the integration
while step < nsteps:

    if verlet:
        V_old = V
        # compute jacobian for current values
        J = jacobian(r, phi)

        r_old = r
        r = r + dt * v + 0.5 * dt * dt * np.dot(m_inv, np.transpose(f))

        # evaluate phit here, just for plotting.
        phit = phi_evaluate(r, phi)
        phi_list.append(phit)
        tol = 0.002
        f_old = f
        f, V = F(r, bonds)
        lamda = single_constraint_vv(r, v, f, m_inv, phi, J)
        invert = False
        # if lambda > 0, then inversion is required, set the positions, velocities and forces to previous step and
        # invert.
        if lamda > 0:
            r = r_old
            V = V_old
            f = f_old
            invert = True
            v = v + lamda * np.dot(m_inv, np.transpose(J))
        else:
            v_old = v
            v = v + 0.5 * dt * np.dot(m_inv, np.transpose(f + f_old))
    else:
        f, V = F(r, bonds)
        phit = phi_evaluate(r, phi)
        phi_list.append(phit)
        invert = False
        tol = 0.002
        if phit < tol:
            J_new = jacobian(r, phi)
            r = r_old
            v = v_old
            ke = ke_old
            J = jacobian(r, phi)
            lamda = single_constraint(v, m_inv, J)
            if (lamda < 0):
                print('lamda less than 0: ', lamda)
            invert = True
            v = v + lamda * np.dot(m_inv, np.transpose(J))
        else:
            v_old = v
            v = v + dt * np.dot(m_inv, f)

        r_old = r
        r = r + dt * v
    # append current bond length to hist
    ke_old = ke
    ke = KE(m, v)
    lm = LM(m, v)
    if invert:
        print("KE before inversion: ", ke_old, " ke after inversion: ", ke)
    ke_list.append(ke)

    lm_list.append(lm)
    pe_list.append(V)
    tote_list.append(V + ke)
    v_list.append(np.linalg.norm(vector(v, 0)))
    v2_list.append(np.linalg.norm(vector(v, 1)))
    v3_list.append(np.linalg.norm(vector(v, 2)))
    rho1_hist.append(np.linalg.norm(vector(r, 0) - vector(r, 1)))
    rho2_hist.append(np.linalg.norm(vector(r, 2) - vector(r, 1)))
    step += 1

# make the fancy plot used in the thesis.

fig = plt.figure(figsize=(6, 5), dpi=300)

phiAx = plt.subplot2grid((2, 2), (0, 1))
energyAx = plt.subplot2grid((2, 2), (1, 1), sharex=phiAx)
cvAx = plt.subplot2grid((2, 2,), (1, 0))

lm = np.array(lm_list)
energyAx.plot(ke_list, label="KE")
energyAx.plot(pe_list, label="PE")
energyAx.plot(tote_list, label="Total")
energyAx.set_ylabel("Energy", fontsize=10)
frame = energyAx.legend(facecolor='white', frameon=True).get_frame()
frame.set_facecolor('white')
energyAx.set_xlabel("Time step", fontsize=10)

phiAx.plot(phi_list)
phiAx.set_ylabel(r'$\phi(\vec{r})$', fontsize=10)
plt.setp(phiAx.get_xticklabels(), visible=False)

colormap = plt.get_cmap('Blues')
colorst = [colormap(i) for i in np.linspace(0.4, 0.9, len(rho1_hist))]
# plot trajectory
cvAx.scatter(np.array(rho1_hist), np.array(rho2_hist), s=1, c=colorst)

y_min = min(rho2_hist) - 0.1
y_max = max(rho2_hist) + 0.1
x_min = min(rho1_hist) - 0.1
x_max = max(rho1_hist) + 0.15
cvAx.set_xlim(x_min, x_max)
cvAx.set_ylim(y_min, y_max)
# axis tidying
xlabel = "$s_1$"
ylabel = "$s_2$"
cvAx.set_xlabel(xlabel, fontsize=10)
cvAx.set_ylabel(ylabel, fontsize=10)

# plot bound.
if phi.norm[1] == 0.0:
    y = np.array(np.linspace(y_min, y_max))
    x = [(-phi.d / phi.norm[0])] * len(y)
else:
    x = np.array(np.linspace(x_min, x_max))
    y = [(-phi.norm[0] * z - phi.d) / phi.norm[1] for z in x]

cvAx.plot(x, y, color="k", lw=4, alpha=0.5)

plt.tight_layout()
# plt.show()
output = "bxd_impulse_vv_ABC.pdf"
plt.savefig(output)