import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

x = np.arange(0.0, 10000.0)
k = 4 * np.cos(x / 21)
e = x / 1880 - 20
d = np.sqrt(k**2 + e**2)

# Wolfram UnitStep[k^2 - 15]
m = (k**2 - 15) >= 0

# Prevent divide-by-zero artifacts from 0.3 / k
safe_k = np.where(np.abs(k) < 1e-9, np.nan, k)

fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
ax.set_xlim(100, 300)
ax.set_ylim(75, 320)
ax.axis("off")

main_points = ax.scatter([], [], s=14, alpha=0.50, linewidths=0)
ghost_points = ax.scatter([], [], s=2, c="white", alpha=0.75, linewidths=0)

frames = 96

def frame_points(t):
    q = (
        3 * np.sin(2 * k)
        + 0.3 / safe_k
        + k * np.sin(x / 4465) * (9 + 2 * np.sin(14 * e - 3 * d + 2 * t))
    )

    px = q + 50 * np.cos(d - t) + 200
    py = 875 - q * np.sin(d - t) - 39 * d

    finite = np.isfinite(px) & np.isfinite(py)

    main_mask = m & finite
    ghost_mask = (~m) & finite

    return px, py, main_mask, ghost_mask

def update(i):
    t = 2 * np.pi * i / frames
    px, py, main_mask, ghost_mask = frame_points(t)

    # Equivalent of Wolfram Blend[{White, Red}, Sin[t]^2]
    blend = np.sin(t) ** 2
    main_color = (1.0, 1.0 - blend, 1.0 - blend)

    main_points.set_offsets(np.column_stack([px[main_mask], py[main_mask]]))
    main_points.set_color([main_color])
    ghost_points.set_offsets(np.column_stack([px[ghost_mask], py[ghost_mask]]))

    return main_points, ghost_points

anim = FuncAnimation(fig, update, frames=frames, interval=40, blit=True)
anim.save("fireeel_moving_creature.gif", writer=PillowWriter(fps=24))
plt.close(fig)