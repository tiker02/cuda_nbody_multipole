from sys import argv
import numpy as np

def generate_plummer(N, filename):
    print(f"Generating Plummer sphere with {N} particles...")
    
    # 1. Inverse transform sampling for radii
    u = np.random.rand(N)
    r = 1.0 / np.sqrt(u**(-2.0/3.0) - 1.0)
    
    # Random spherical angles to map to 3D space
    theta = np.arccos(np.random.uniform(-1.0, 1.0, N))
    phi = np.random.uniform(0.0, 2.0 * np.pi, N)
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
# ---------------------------------------------------------
    # 2. VELOCITIES (Rejection Sampling)
    # ---------------------------------------------------------
    q = np.zeros(N)
    # The maximum of q^2 * (1 - q^2)^3.5 occurs at q = sqrt(2/9)
    # The maximum value of the function is ~0.092. We use 0.1 as our bounding box.
    for i in range(N):
        while True:
            q_try = np.random.uniform(0.0, 1.0)
            g_try = np.random.uniform(0.0, 0.1)
            
            # Rejection condition
            if g_try < (q_try**2) * ((1.0 - q_try**2)**3.5):
                q[i] = q_try
                break

    # Calculate local escape velocity (in standard N-body units: G=1, M=1, a=1)
    v_esc = np.sqrt(2.0) * (1.0 + r**2)**(-0.25)
    
    # Actual velocity magnitude
    v = q * v_esc
    
    # Random isotropic direction for the velocity vector
    theta_vel = np.arccos(np.random.uniform(-1.0, 1.0, N))
    phi_vel = np.random.uniform(0.0, 2.0 * np.pi, N)
    
    vx = v * np.sin(theta_vel) * np.cos(phi_vel)
    vy = v * np.sin(theta_vel) * np.sin(phi_vel)
    vz = v * np.cos(theta_vel)
    # 3. Equal mass distribution matching your 0.002 format (Total mass = 1.0)
    mass = np.ones(N) / N

    # 4. Write directly to file matching the exact syntax
    print(f"Writing to {filename}...")
    with open(filename, 'w') as f:
        for i in range(N):
            # Layout matching: mass x y z vx vy vz
            f.write(f"{mass[i]:.16f} {x[i]:.16f} {y[i]:.16f} {z[i]:.16f} {vx[i]:.16f} {vy[i]:.16f} {vz[i]:.16f}\n")
            
    print("Done!")

if __name__ == "__main__":
    N = int(argv[1])
    generate_plummer(N, f"plummer_{N}.dat")