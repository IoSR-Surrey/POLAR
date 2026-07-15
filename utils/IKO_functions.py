import numpy as np
from scipy.special import sph_harm


def num_hoa_channels(order: int) -> int:
    """Number of 3D HOA channels for given order N: (N+1)^2."""
    return (order + 1) ** 2


def compute_real_sh_matrix(order: int,
                           azimuths: np.ndarray,
                           elevations: np.ndarray) -> np.ndarray:
    """Compute real-valued spherical harmonics up to given order.

    Args:
        order: HOA order N (e.g. 3 for the IKO).
        azimuths: (K,) azimuth angles in radians, in [0, 2π).
        elevations: (K,) elevation angles in radians, in [-π/2, π/2].

    Returns:
        Y: (K, L) array of real-valued SH, where L = (order+1)^2.
             Each row corresponds to one direction; each column to one HOA channel.
    """
    assert azimuths.shape == elevations.shape
    K = azimuths.shape[0]
    L = num_hoa_channels(order)

    Y = np.zeros((K, L), dtype=np.float64)

    idx = 0
    # Loop over degree n and order m
    for n in range(order + 1):
        for m in range(-n, n + 1):
            # Convert to sph_harm convention: m, n, phi (azimuth), theta (polar)
            theta = np.pi / 2.0 - elevations  # polar angle
            phi = azimuths                    # azimuth

            # Complex spherical harmonics
            Y_complex = sph_harm(m, n, phi, theta)  # (K,) complex

            # Real-valued SH basis (N3D-like): combine real/imag parts
            if m < 0:
                Y[:, idx] = np.sqrt(2.0) * (-1) ** m * Y_complex.imag
            elif m == 0:
                Y[:, idx] = Y_complex.real
            else:  # m > 0
                Y[:, idx] = np.sqrt(2.0) * (-1) ** m * Y_complex.real

            idx += 1

    return Y  # (K, L)


def compute_beam_pattern(order: int,
                         weights: np.ndarray,
                         azimuths: np.ndarray,
                         elevations: np.ndarray) -> np.ndarray:
    """Compute the beam pattern for a given HOA weight vector.

    Args:
        order: HOA order N (e.g. 3).
        weights: (L,) real-valued HOA weights, L = (N+1)^2.
        azimuths: (K,) azimuth angles in radians.
        elevations: (K,) elevation angles in radians.

    Returns:
        pattern: (K,) real-valued beam pattern samples over the given directions.
    """
    L_expected = num_hoa_channels(order)
    assert weights.shape[0] == L_expected, f"weights length must be {L_expected} for order {order}"

    Y = compute_real_sh_matrix(order, azimuths, elevations)  # (K, L)
    pattern = Y @ weights  # (K,)
    return pattern


def example_horizontal_beam(order: int = 3,
                            num_points: int = 360,
                            target_azimuth: float = 0.0,
                            target_elevation: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Example: construct a simple beam steered towards a target direction on the horizontal plane.

    This builds a weight vector proportional to the SH basis evaluated at the target direction,
    then computes its pattern around the full circle (elevation = 0).

    Args:
        order: HOA order (default 3).
        num_points: number of sample points around the circle.
        target_azimuth: target azimuth in radians.
        target_elevation: target elevation in radians (0 for horizontal).

    Returns:
        azimuths: (K,) azimuths used for sampling.
        elevations: (K,) elevations (all zeros if horizontal).
        pattern: (K,) beam pattern values for the constructed weights.
    """
    # Sample directions around the horizontal circle
    azimuths = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    elevations = np.zeros_like(azimuths)  # constrain to 2D horizontal plane

    # SH at target direction
    Y_target = compute_real_sh_matrix(order,
                                      azimuths=np.array([target_azimuth]),
                                      elevations=np.array([target_elevation]))  # (1, L)

    # Use SH at target as a simple weight vector (normalised)
    weights = Y_target[0, :]
    weights /= np.linalg.norm(weights) + 1e-12

    pattern = compute_beam_pattern(order, weights, azimuths, elevations)  # (K,)
    return azimuths, elevations, pattern


if __name__ == "__main__":
    # Example usage: 3rd-order HOA, beam steered to azimuth 0° on the horizontal plane
    order = 3
    azimuths, elevations, pattern = example_horizontal_beam(order=order,
                                                            num_points=360,
                                                            target_azimuth=0.0,
                                                            target_elevation=0.0)

    # Simple text output of pattern summary
    print(f"Pattern min: {pattern.min():.3f}, max: {pattern.max():.3f}")
    print(f"Pattern at target direction (index 0): {pattern[0]:.3f}")
