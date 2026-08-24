from sys import argv, exit
import h5py
import numpy as np

def compare_snapshots(cpu_h5_path, gpu_h5_path, tolerance=1e-3):
    has_divergence = False

    with h5py.File(cpu_h5_path, 'r') as f_cpu, h5py.File(gpu_h5_path, 'r') as f_gpu:
        if 'ID' in f_cpu and 'ID' in f_gpu:
            cpu_ids = f_cpu['ID'][:]
            gpu_ids = f_gpu['ID'][:]
            cpu_order = np.argsort(cpu_ids)
            gpu_order = np.argsort(gpu_ids)
        else:
            cpu_order = slice(None)
            gpu_order = slice(None)

        common_keys = [k for k in f_cpu.keys() if k in f_gpu]

        for key in common_keys:
            # Skip HDF5 groups / non-datasets
            if not isinstance(f_cpu[key], h5py.Dataset) or not isinstance(f_gpu[key], h5py.Dataset):
                continue

            data_cpu = np.array(f_cpu[key])[cpu_order]
            data_gpu = np.array(f_gpu[key])[gpu_order]

            # Skip empty or string datasets
            if data_cpu.size == 0 or data_cpu.dtype.kind not in ['f', 'i', 'u']:
                continue

            diff = data_cpu - data_gpu
            abs_diff = np.abs(diff)
            denom = np.linalg.norm(data_cpu)
            rel_l2 = np.linalg.norm(diff) / (denom + 1e-15)
            max_err = np.max(abs_diff)

            print(f"\n--- Field: {key} ---")
            print(f"  Shape              : {data_cpu.shape}")
            print(f"  Max Absolute Diff  : {max_err:.6e}")
            print(f"  Relative L2 Error  : {rel_l2:.6e}")
            print(f"  CPU Sample [0:3]   : {data_cpu[:3]}")
            print(f"  GPU Sample [0:3]   : {data_gpu[:3]}")

            # Enforce tolerance on physical integration fields
            if key in ['Accx', 'Accy', 'Accz', 'Potential', 'Scalarforce']:
                if rel_l2 > tolerance:
                    print(f"  [FAIL] Field '{key}' relative L2 error ({rel_l2:.6e}) exceeds tolerance ({tolerance:.6e})")
                    has_divergence = True

    if has_divergence:
        print("\n[ERROR] Numerical divergence detected! Aborting profiling pipeline.")
        exit(1)
    else:
        print("\n[PASS] All validated physical fields within acceptable numerical tolerance.")
        exit(0)

compare_snapshots(argv[1], argv[2]);