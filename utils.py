from dataclasses import dataclass
import math


@dataclass
class Attractor:
    id: int
    center: tuple
    radius: float
    strength: float
    emergent: bool = False


def get_attractor_by_id(attractors, attractor_id):
    for attr in attractors:
        if attr.id == attractor_id:
            return attr
    return None


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def nearest_attractor(point, attractors):
    """
    Returns (attractor_id, distance_to_center)
    """
    if not attractors:
        return None, float("inf")

    best_attr = None
    best_dist = float("inf")

    for attr in attractors:
        d = euclidean_distance(point, attr.center)
        if d < best_dist:
            best_dist = d
            best_attr = attr

    return best_attr.id, best_dist


def clip_position(pos, width, height):
    x = min(max(pos[0], 0.0), width - 1.0)
    y = min(max(pos[1], 0.0), height - 1.0)
    return (x, y)


def boundary_distance(point, attractor):
    """
    Absolute distance from point to attractor boundary.
    """
    d = euclidean_distance(point, attractor.center)
    return abs(d - attractor.radius)


def point_inside_attractor(point, attractor):
    return euclidean_distance(point, attractor.center) <= attractor.radius


def is_interface(point, attractors, interface_threshold):
    """
    A point is in the interface zone if it is close to the boundary
    of at least one attractor.
    """
    for attr in attractors:
        if boundary_distance(point, attr) <= interface_threshold:
            return True
    return False


# ============================================================
# ANALYSIS-ONLY INTERFACE METRICS
# ============================================================

def competitive_boundary_distance(point, attractors):
    """
    Analysis-only metric.

    Distance from the decision boundary between the two nearest attractors.
    Small values mean the point lies near the inter-attractor interface
    induced by nearest-center assignment.
    """
    if len(attractors) < 2:
        return None

    dists = []
    for attr in attractors:
        d = euclidean_distance(point, attr.center)
        dists.append(d)

    dists.sort()
    return abs(dists[1] - dists[0])


def competitive_is_interface(point, attractors, threshold):
    """
    Analysis-only interface indicator based on competitive boundary distance.
    """
    bd = competitive_boundary_distance(point, attractors)
    if bd is None:
        return False
    return bd <= threshold





