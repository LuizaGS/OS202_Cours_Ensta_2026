from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Step 1: Process 0 generates arbitrary numbers
N = 20
if rank == 0:
    # Generating random floats between 0 and 1
    data = np.random.random(N).tolist()
    print(f"Master (rank 0) generated: {data}\n")
else:
    data = None

# Step 2: Dispatching data to buckets
# Distribute of data based on its value, not its position
if rank == 0:
    buckets = [[] for _ in range(size)]
    for number in data:
        # Determine which process gets the number
        index = int(number * size)
        if index == size:
            index = size - 1
        buckets[index].append(number)
else:
    buckets = None

# Scatter sends bucket[0] to Rank 0, bucket[1] to Rank 1, etc.
local_bucket = comm.scatter(buckets, root=0)

# Step 3: Parallel Sorting
# Each process sorts its own bucket independently
local_bucket.sort()
print(f"Rank {rank} sorted its bucket: {local_bucket}")

# Step 4: Gather the sorted results
# Collect all sorted sub-lists back to Process 0
all_sorted_buckets = comm.gather(local_bucket, root=0)

# Final Step: Consolidation on Process 0
if rank == 0:
    final_table = []
    for b in all_sorted_buckets:
        final_table.extend(b)
    
    print(f"\nFinal sorted table gathered on Rank 0: {final_table}")

# mpiexec -np 4 python3 td3_solution.py