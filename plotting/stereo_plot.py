import numpy as np
import matplotlib.pyplot as plt

# Geometry parameters
radius = 2.0  # metres
speaker_angles_deg = [-30, 30]
line_incline_deg = 10

# Listener positions: (x, y)
listeners = np.array([
    [0.0, 0.5],
    [0.0, 0.0],
    [0.0, -0.5],
])

# Convert loudspeaker angles to radians
speaker_angles_rad = np.deg2rad(speaker_angles_deg)

# Coordinates using the standard mathematical convention:
# 0 degrees points to the right of the listener/origin,
# positive angles rotate counter-clockwise.
speaker_x = radius * np.cos(speaker_angles_rad)
speaker_y = radius * np.sin(speaker_angles_rad)

# Dashed line parameters
line_incline_rad = np.deg2rad(line_incline_deg)
line_slope = np.tan(line_incline_rad)
x_line = np.linspace(-10, 2.5, 200)

# Create figure
fig, ax = plt.subplots(figsize=(7, 5))

# Plot dashed 10 degree inclined lines through each listener
for x_listener, y_listener in listeners:
    y_line = y_listener + line_slope * (x_line - x_listener)
    ax.plot(
        x_line,
        y_line,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        zorder=1,
    )

# Plot listeners
ax.scatter(
    listeners[:, 0],
    listeners[:, 1],
    s=90,
    c="black",
    marker="o",
    label="Listeners",
    zorder=3,
)
for x_listener, y_listener in listeners:
    ax.text(
        x_listener - 0.10,
        y_listener,
        f"({x_listener:.0f}, {y_listener:g})",
        ha="right",
        va="center",
        fontsize=9,
    )

# Plot loudspeakers
ax.scatter(
    speaker_x,
    speaker_y,
    s=180,
    c="tab:red",
    marker="s",
    label="Loudspeakers",
    zorder=4,
)
for angle, x, y in zip(speaker_angles_deg, speaker_x, speaker_y):
    vertical_offset = 0.10 if y >= 0 else -0.10
    va = "bottom" if y >= 0 else "top"
    ax.text(x + 0.10, y + vertical_offset, f"{angle:+d}°", ha="left", va=va)

# Add an annotation for the dashed-line angle
# ax.text(
#     1.0,
#     -1.05,
#     "Dashed lines inclined at +10°",
#     ha="center",
#     va="center",
#     fontsize=10,
#     color="gray",
# )

# Formatting
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("x position / m")
ax.set_ylabel("y position / m")
# ax.set_title("Three Listeners, Two Loudspeakers, and 10° Inclined Lines")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper left")

# Set limits with padding, keeping loudspeakers to the right of the listeners
ax.set_xlim(-0.6, 2.5)
ax.set_ylim(-1.4, 1.4)

plt.tight_layout()
plt.show()
