import numpy as np
import matplotlib.pyplot as plt

# Geometry parameters
speaker_x_position = 2.0  # metres; all loudspeakers lie on this x-position
speaker_angles_deg = [-35, -30, 30, 35]
marker_angle_deg = 0

# Dense listener column along x = 0
listener_y = np.linspace(-1.0, 1.0, 41)
listener_x = np.zeros_like(listener_y)

# Convert angles to radians
speaker_angles_rad = np.deg2rad(speaker_angles_deg)
marker_angle_rad = np.deg2rad(marker_angle_deg)

# Coordinates using the standard mathematical convention:
# 0 degrees points to the right of the listener column/origin,
# positive angles rotate counter-clockwise.
# All loudspeakers are placed on the x = 2 m line, with y set by angle.
speaker_x = np.full(len(speaker_angles_deg), speaker_x_position)
speaker_y = speaker_x_position * np.tan(speaker_angles_rad)

# Marker coordinates at 0 degrees on the same x = 2 m line
marker_x = speaker_x_position
marker_y = speaker_x_position * np.tan(marker_angle_rad)

# Create figure
fig, ax = plt.subplots(figsize=(7, 6))

# Plot dashed lines from every listener to the 0 degree marker
for x_listener, y_listener in zip(listener_x, listener_y):
    ax.plot(
        [x_listener, marker_x],
        [y_listener, marker_y],
        color="gray",
        linestyle="--",
        linewidth=0.8,
        alpha=0.45,
        zorder=1,
    )

# Plot dense column of listeners
ax.scatter(
    listener_x,
    listener_y,
    s=35,
    c="black",
    marker="o",
    label="Listeners",
    zorder=3,
)
ax.plot(listener_x, listener_y, color="black", linewidth=1.0, alpha=0.4, zorder=2)

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

# Plot phantom source marker
ax.scatter(
    marker_x,
    marker_y,
    s=130,
    c="tab:blue",
    marker="o",
    label="Phantom source",
    zorder=5,
)

# Formatting
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("x position / m", fontsize=16)
ax.set_ylabel("y position / m", fontsize=16)
ax.tick_params(axis="both", labelsize=14)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper left", fontsize=14)

# Set limits with padding
ax.set_xlim(-0.3, 2.7)
y_min = min(listener_y.min(), speaker_y.min()) - 0.35
y_max = max(listener_y.max(), speaker_y.max()) + 0.35
ax.set_ylim(y_min, y_max)

plt.tight_layout()
plt.show()
