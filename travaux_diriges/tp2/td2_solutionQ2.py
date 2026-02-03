# Produit matrice-vecteur v = A.u (version MPI, découpage par colonnes)
import numpy as np
from time import time
from mpi4py import MPI

# Taille du problème
dim = 120

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nbp = comm.Get_size()

# On suppose dim divisible par nbp
if dim % nbp != 0:
    if rank == 0:
        raise ValueError(f"dim={dim} doit être divisible par nbp={nbp}")
    raise SystemExit

Nloc = dim // nbp
j0 = rank * Nloc
j1 = (rank + 1) * Nloc

# Vecteur u (même définition que dans la version séquentielle)
u = np.array([i + 1. for i in range(dim)], dtype=np.float64)

# Partie locale de la matrice (colonnes j0 à j1-1)
A_loc = np.array(
    [[(i + j) % dim + 1. for j in range(j0, j1)] for i in range(dim)],
    dtype=np.float64
)

comm.Barrier()
deb = time()

# Contribution locale au produit A.u
u_loc = u[j0:j1]
v_local = A_loc.dot(u_loc)

# Somme des contributions locales
v = np.empty(dim, dtype=np.float64)
comm.Allreduce(v_local, v, op=MPI.SUM)

comm.Barrier()
fin = time()
t_loc = fin - deb
t_par = comm.reduce(t_loc, op=MPI.MAX, root=0)

if rank == 0:
    print(f"dim={dim}, nbp={nbp}, Nloc={Nloc}")
    print(f"Temps calcul : {t_par}")
    print(f"u = {u}")
    print(f"v = {v}")




# --------------------------------------------------
# Version MPI, découpage par lignes
# --------------------------------------------------

# Nombre de lignes locales
Nloc = dim // nbp
i0 = rank * Nloc
i1 = (rank + 1) * Nloc

# Bloc local de lignes
A_loc = np.array(
    [[(i + j) % dim + 1. for j in range(dim)] for i in range(i0, i1)],
    dtype=np.float64
)

comm.Barrier()
deb = time()

# Produit local : lignes i0..i1-1
v_local = A_loc.dot(u)

# Rassemblement du vecteur complet sur toutes les tâches
v = np.empty(dim, dtype=np.float64)
comm.Allgather(v_local, v)

comm.Barrier()
fin = time()
t_loc = fin - deb
t_par = comm.reduce(t_loc, op=MPI.MAX, root=0)

if rank == 0:
    print("Version par lignes")
    print(f"dim={dim}, nbp={nbp}, Nloc={Nloc}")
    print(f"Temps calcul : {t_par}")
    print(f"u = {u}")
    print(f"v = {v}")
