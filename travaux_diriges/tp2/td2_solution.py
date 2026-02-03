# TD n° 2 - Luiza Gonçalves Soares
####### Parallélisation ensemble de Mandelbrot ######
### Question 01-A:

# Calcul de l'ensemble de Mandelbrot en python
import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm
from mpi4py import MPI


@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius:  float = 2.0

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth) / self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex, smooth=False) -> int | float:
        z: complex
        iter: int

        # Test d'appartenance à des zones de convergence connues
        if c.real * c.real + c.imag * c.imag < 0.0625:
            return self.max_iterations
        if (c.real + 1) * (c.real + 1) + c.imag * c.imag < 0.0625:
            return self.max_iterations

        # Test d'appartenance à la cardioïde principale
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real - 0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5 * (1 - ct.real / max(ctnrm2, 1.E-14)):
                return self.max_iterations

        # Itération de la suite z_{n+1} = z_n^2 + c
        z = 0
        for iter in range(self.max_iterations):
            z = z * z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z))) / log(2)
                return iter
        return self.max_iterations


def row_range(rank: int, size: int, H: int) -> tuple[int, int]:
    """
    Détermine l'intervalle de lignes traité par chaque processus.
    """
    base = H // size
    rem = H % size

    if rank < rem:
        start = rank * (base + 1)
        end = start + (base + 1)
    else:
        start = rem * (base + 1) + (rank - rem) * base
        end = start + base

    return start, end


if __name__ == "__main__":
    # Initialisation MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
    width, height = 1024, 1024

    scaleX = 3.0 / width
    scaleY = 2.25 / height

    # ---------------------------------------------------------------------
    # Question 01-A : Répartition par blocs contigus (bandes horizontales)
    # ---------------------------------------------------------------------
    y_start, y_end = row_range(rank, size, height)
    local_h = y_end - y_start

    convergence_local = np.empty((local_h, width), dtype=np.double)

    comm.Barrier()
    deb = time()

    for local_y, y in enumerate(range(y_start, y_end)):
        imag = -1.125 + scaleY * y
        for x in range(width):
            real = -2.0 + scaleX * x
            c = complex(real, imag)
            convergence_local[local_y, x] = mandelbrot_set.convergence(c, smooth=True)

    comm.Barrier()
    fin = time()
    local_time = fin - deb

    t_par = comm.reduce(local_time, op=MPI.MAX, root=0)
    blocks = comm.gather(convergence_local, root=0)

    if rank == 0:
        convergence = np.vstack(blocks)

        print(f"np={size}")
        print(f"Temps du calcul de l'ensemble de Mandelbrot : {t_par}")

        deb = time()
        image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence) * 255))
        fin = time()
        print(f"Temps de constitution de l'image : {fin - deb}")

        image.show()

    # ---------------------------------------------------------------------
    # Question 01-B : Répartition statique entrelacée (round-robin)
    # ---------------------------------------------------------------------
    local_rows = list(range(rank, height, size))
    local_h = len(local_rows)

    convergence_local = np.empty((local_h, width), dtype=np.double)

    comm.Barrier()
    deb = time()

    for i, y in enumerate(local_rows):
        imag = -1.125 + scaleY * y
        for x in range(width):
            real = -2.0 + scaleX * x
            c = complex(real, imag)
            convergence_local[i, x] = mandelbrot_set.convergence(c, smooth=True)

    comm.Barrier()
    fin = time()
    local_time = fin - deb

    t_par = comm.reduce(local_time, op=MPI.MAX, root=0)

    rows_all = comm.gather(local_rows, root=0)
    data_all = comm.gather(convergence_local, root=0)

    if rank == 0:
        convergence = np.empty((height, width), dtype=np.double)

        for rows, block in zip(rows_all, data_all):
            for i, y in enumerate(rows):
                convergence[y, :] = block[i, :]

        print(f"np={size}")
        print(f"Temps du calcul de l'ensemble de Mandelbrot : {t_par}")

        deb = time()
        image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence) * 255))
        fin = time()
        print(f"Temps de constitution de l'image : {fin - deb}")

        image.show()

    # ------------------------------------------
    # Question 01-C : Stratégie maître-esclave 
    # ------------------------------------------
    TAG_TASK = 10
    TAG_STOP = 11

    comm.Barrier()
    deb = time()

    if rank == 0:
        # Maître : distribue les lignes à la demande
        next_y = 0
        active_workers = size - 1

        while active_workers > 0:
            status = MPI.Status()
            comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            worker = status.Get_source()

            if next_y < height:
                comm.send(next_y, dest=worker, tag=TAG_TASK)
                next_y += 1
            else:
                comm.send(None, dest=worker, tag=TAG_STOP)
                active_workers -= 1

        # Le maître ne calcule pas de lignes (payload vide)
        local_payload = ([], np.empty((0, width), dtype=np.double))

    else:
        # Worker : demande une ligne, la calcule, stocke localement
        local_rows = []
        local_data = []

        while True:
            comm.send(None, dest=0)  # demande de tâche

            status = MPI.Status()
            y = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

            if status.Get_tag() == TAG_STOP:
                break

            imag = -1.125 + scaleY * y
            row = np.empty((width,), dtype=np.double)

            for x in range(width):
                real = -2.0 + scaleX * x
                c = complex(real, imag)
                row[x] = mandelbrot_set.convergence(c, smooth=True)

            local_rows.append(y)
            local_data.append(row)

        if len(local_data) > 0:
            local_data = np.vstack(local_data)
        else:
            local_data = np.empty((0, width), dtype=np.double)

        local_payload = (local_rows, local_data)

    comm.Barrier()
    fin = time()
    local_time = fin - deb

    t_par = comm.reduce(local_time, op=MPI.MAX, root=0)

    # Rassemblement final (une seule fois)
    payloads = comm.gather(local_payload, root=0)

    if rank == 0:
        convergence = np.empty((height, width), dtype=np.double)

        for rows, block in payloads:
            for i, y in enumerate(rows):
                convergence[y, :] = block[i, :]

        print(f"np={size}")
        print(f"Temps du calcul de l'ensemble de Mandelbrot : {t_par}")

        deb = time()
        image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence) * 255))
        fin = time()
        print(f"Temps de constitution de l'image : {fin - deb}")

        image.show()
