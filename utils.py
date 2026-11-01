from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Attractor:
    """
    Fixed attractor used in the minimal-mechanism model.

    The attractor set is immutable during a simulation run.
    """
    id: int
    center: tuple
    radius: float
    strength: float
    emergent: bool = False


# ============================================================
# BASIC GEOMETRY
# ============================================================

def euclidean_distance(p1, p2):
    """Euclidean distance between two 2D points."""
    return math.hypot(
        p1[0] - p2[0],
        p1[1] - p2[1],
    )


def get_attractor_by_id(attractors, attractor_id):
    """Return the attractor with the requested ID, or None."""
    for attr in attractors:
        if attr.id == attractor_id:
            return attr
    return None


def nearest_attractor(point, attractors):
    """
    Assign a point to the nearest attractor centre.

    Returns
    -------
    (attractor_id, distance_to_center)

    Ties are resolved deterministically by attractor list order.
    """
    if not attractors:
        return None, float("inf")

    best_attr = min(
        attractors,
        key=lambda attr: euclidean_distance(
            point,
            attr.center,
        ),
    )

    return (
        best_attr.id,
        euclidean_distance(
            point,
            best_attr.center,
        ),
    )


def clip_position(pos, width, height):
    """
    Clip a position to the continuous rectangular simulation domain.

    Coordinates are kept in [0, width] x [0, height].
    """
    x = min(max(float(pos[0]), 0.0), float(width))
    y = min(max(float(pos[1]), 0.0), float(height))
    return (x, y)


# ============================================================
# ATTRACTOR-RADIUS GEOMETRY
# ============================================================

def radial_boundary_distance(point, attractor):
    """
    Absolute distance to the attractor's nominal radius.

    This is NOT the inter-basin interface metric used in the paper.
    It is retained only as a diagnostic geometric quantity.
    """
    d = euclidean_distance(
        point,
        attractor.center,
    )
    return abs(d - attractor.radius)


def point_inside_attractor(point, attractor):
    """Whether a point lies inside the attractor's nominal radius."""
    return (
        euclidean_distance(
            point,
            attractor.center,
        )
        <= attractor.radius
    )


# Backward-compatible alias for code that may still inspect radial
# attractor geometry.  New paper analyses should use
# basin_boundary_distance().
boundary_distance = radial_boundary_distance


# ============================================================
# INTER-BASIN / VORONOI GEOMETRY
# ============================================================

def pairwise_bisector_distance(
    point,
    attractor_a,
    attractor_b,
):
    """
    Exact perpendicular distance from a point to the Euclidean
    bisector separating two attractor centres.

    For centres c_a and c_b, the equal-distance boundary satisfies

        ||x-c_a||^2 = ||x-c_b||^2.

    The expression below is the exact distance to that line.
    """
    ax, ay = attractor_a.center
    bx, by = attractor_b.center
    px, py = point

    center_distance = euclidean_distance(
        attractor_a.center,
        attractor_b.center,
    )

    if center_distance == 0.0:
        raise ValueError(
            "Two attractors share the same centre; "
            "their basin boundary is undefined."
        )

    da2 = (
        (px - ax) ** 2
        + (py - ay) ** 2
    )
    db2 = (
        (px - bx) ** 2
        + (py - by) ** 2
    )

    return abs(db2 - da2) / (
        2.0 * center_distance
    )


def basin_boundary_distance(point, attractors):
    """
    Distance from a point to the nearest boundary of its
    nearest-centre basin.

    Basin assignment is the Voronoi partition induced by attractor
    centres.  If attractor i is the nearest centre, this function
    computes the minimum perpendicular distance from the point to
    any bisector separating i from another attractor.

    Returns None when fewer than two attractors exist.
    """
    if len(attractors) < 2:
        return None

    current_id, _ = nearest_attractor(
        point,
        attractors,
    )

    current = get_attractor_by_id(
        attractors,
        current_id,
    )

    if current is None:
        return None

    distances = [
        pairwise_bisector_distance(
            point,
            current,
            other,
        )
        for other in attractors
        if other.id != current.id
    ]

    if not distances:
        return None

    return min(distances)


def is_interface(
    point,
    attractors,
    interface_threshold,
):
    """
    Paper interface indicator.

    A point is classified as lying in an inter-attractor interface
    zone when its distance to the nearest Voronoi basin boundary is
    no greater than `interface_threshold`.
    """
    if interface_threshold < 0:
        raise ValueError(
            "interface_threshold must be >= 0."
        )

    distance = basin_boundary_distance(
        point,
        attractors,
    )

    if distance is None:
        return False

    return (
        distance
        <= interface_threshold
    )


# ============================================================
# BACKWARD-COMPATIBLE ANALYSIS NAMES
# ============================================================

def competitive_boundary_distance(
    point,
    attractors,
):
    """
    Backward-compatible name for the paper's basin-boundary metric.

    Unlike the older |d2-d1| proxy, this returns the exact
    perpendicular distance to the nearest Voronoi boundary.
    """
    return basin_boundary_distance(
        point,
        attractors,
    )


def competitive_is_interface(
    point,
    attractors,
    threshold,
):
    """Backward-compatible interface indicator."""
    return is_interface(
        point,
        attractors,
        threshold,
    )