import os
import matplotlib.pyplot as plt
import numpy as np

NX = 41
NY = 41

def main():

    cwd = os.getcwd()
    print("Current working dir:", cwd)
    data_dir = "./"
    if os.path.isdir(data_dir):
        print(f"Contents of {data_dir}:")
        for fname in os.listdir(data_dir):
            print("  ", fname)
    else:
        print(f"ERROR: {data_dir!r} is not a directory (or doesn’t exist)")

    x = np.linspace(0, 2, NX)
    y = np.linspace(0, 2, NY)
    X, Y = np.meshgrid(x, y)

    candidates = {
        "C++": ["u++.dat", "v++.dat", "p++.dat"],
        "CUDA": ["u_cu.dat", "v_cu.dat", "p_cu.dat"],
    }

    for device, files in candidates.items():
        paths = [os.path.join(data_dir, fname) for fname in files]
        if all(os.path.exists(p) for p in paths):
            string_device = device
            ufile, vfile, pfile = paths
            break
    else:
        string_device = "Default"
        ufile, vfile, pfile = "u.dat", "v.dat", "p.dat"
        print("No C++/CUDA files found → falling back to default files")

    print(f"→ Using device = {string_device}")
    print("→ Files:", ufile, vfile, pfile)

    with open(ufile, 'r') as f:
        uraw = f.readlines()
    with open(vfile, 'r') as f:
        vraw = f.readlines()
    with open(pfile, 'r') as f:
        praw = f.readlines()

    u = np.zeros((NY, NX))
    v = np.zeros((NY, NX))
    p = np.zeros((NY, NX))

    for n in range(len(uraw)):
        plt.clf()

        u_flat = [float(val) for val in uraw[n].split()]
        v_flat = [float(val) for val in vraw[n].split()]
        p_flat = [float(val) for val in praw[n].split()]

        for j in range(NY):
            for i in range(NX):
                idx = j * NX + i
                u[j, i] = u_flat[idx]
                v[j, i] = v_flat[idx]
                p[j, i] = p_flat[idx]

        plt.contourf(X, Y, p, alpha=0.5, cmap=plt.cm.coolwarm)
        plt.quiver(X[::2, ::2], Y[::2, ::2],
                   u[::2, ::2], v[::2, ::2])

        plt.title(f"{string_device}, n = {n}")
        plt.pause(0.01)

    plt.savefig(f"./Final Report/{string_device}_plot.png")
    plt.show()

if __name__ == "__main__":
    main()
