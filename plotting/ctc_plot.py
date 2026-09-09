import numpy as np
import matplotlib.pyplot as plt

# Geometry parameters
radius = 2.0  # metres
speaker_angles_deg = [-30, 30]
single_marker_angle_deg = 45

# Convert angles to radians
speaker_angles_rad = np.deg2rad(speaker_angles_deg)
single_marker_angle_rad = np.deg2rad(single_marker_angle_deg)

# Speaker coordinates
speaker_x = radius * np.cos(speaker_angles_rad)
speaker_y = radius * np.sin(speaker_angles_rad)

# Single marker coordinates at +45 degrees
marker_x = radius * np.cos(single_marker_angle_rad)
marker_y = radius * np.sin(single_marker_angle_rad)

# Create figure
fig, ax = plt.subplots(figsize=(7, 6))

# Plot listener
ax.scatter(0, 0, s=120, c="black", marker="o", label="Listener")
ax.text(0, -0.15, "Listener\n(0, 0)", ha="center", va="top")

# Plot loudspeakers
ax.scatter(speaker_x, speaker_y, s=180, c="tab:red", marker="s", label="Loudspeakers")
for angle, x, y in zip(speaker_angles_deg, speaker_x, speaker_y):
    ax.text(x, y + 0.12, f"Speaker\n{angle:+d}°", ha="center", va="bottom")

# Plot single marker at +45 degrees
ax.scatter(marker_x, marker_y, s=90, c="tab:blue", marker="o", zorder=3, label="+45° marker")
ax.text(marker_x * 1.06, marker_y * 1.06, f"{single_marker_angle_deg:+d}°", ha="center", va="center", fontsize=10)

# Draw radial guide lines to speakers and marker
for x, y in zip(speaker_x, speaker_y):
    ax.plot([0, x], [0, y], color="gray", linestyle="--", linewidth=1)
ax.plot([0, marker_x], [0, marker_y], color="tab:blue", linestyle=":", linewidth=1.5)

# Formatting
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("x position / m")
ax.set_ylabel("y position / m")
# ax.set_title("Listener, Loudspeakers, and +45° Marker")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper left")

# Set limits with some padding
padding = 0.3
ax.set_xlim( - padding, radius + padding)
ax.set_ylim(-radius + padding, radius - padding)

plt.tight_layout()
plt.show()
