import numpy as np
import matplotlib.pyplot as plt

# Geometry parameters
radius = 2.0  # metres
speaker_angles_deg = [-30, 30]
marker_step_deg = 5

# Convert speaker angles to radians
speaker_angles_rad = np.deg2rad(speaker_angles_deg)

# Speaker coordinates
speaker_x = radius * np.cos(speaker_angles_rad)
speaker_y = radius * np.sin(speaker_angles_rad)

# Arc coordinates from -30 to +30 degrees
arc_angles_deg = np.arange(-25, 25 + marker_step_deg, marker_step_deg)
arc_angles_rad = np.deg2rad(arc_angles_deg)
arc_x = radius * np.cos(arc_angles_rad)
arc_y = radius * np.sin(arc_angles_rad)

# Create figure
fig, ax = plt.subplots(figsize=(7, 6))

# Plot listener
ax.scatter(0, 0, s=120, c="black", marker="o", label="Listener")
ax.text(0, -0.15, "Listener\n(0, 0)", ha="center", va="top")

# Plot loudspeakers
ax.scatter(speaker_x, speaker_y, s=180, c="tab:red", marker="s", label="Loudspeakers")
for angle, x, y in zip(speaker_angles_deg, speaker_x, speaker_y):
    ax.text(x, y + 0.12, f"Speaker\n{angle:+d}°", ha="center", va="bottom")

# Plot arc and markers every 5 degrees
ax.plot(arc_x, arc_y, color="tab:blue", linewidth=2, label="2 m arc")
ax.scatter(arc_x, arc_y, s=35, c="tab:blue", zorder=3, label="5° markers")

# Draw radial guide lines to speakers
for x, y in zip(speaker_x, speaker_y):
    ax.plot([0, x], [0, y], color="gray", linestyle="--", linewidth=1)

# Add angle labels on arc markers
for angle, x, y in zip(arc_angles_deg, arc_x, arc_y):
    if angle % 10 == 0:
        ax.text(x * 1.06, y * 1.06, f"{angle:+d}°", ha="center", va="center", fontsize=9)

# Formatting
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("x position / m")
ax.set_ylabel("y position / m")
# ax.set_title("Listener and Loudspeaker Arrangement")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper left")

# Set limits with some padding
padding = 0.5
ax.set_xlim( - padding, radius + padding)
ax.set_ylim(-radius + padding, radius - padding)

plt.tight_layout()
plt.show()
