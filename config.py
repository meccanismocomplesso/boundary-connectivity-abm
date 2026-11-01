# ============================================================
# Fixed-landscape minimal-mechanism configuration
# ============================================================
#
# First-paper mechanism:
#
#   fixed structured landscape
#       -> heterogeneous mobility
#       -> interface-sensitive exploration
#       -> novelty-sensitive exploration
#       -> boundary crossings
#       -> transition graph
#       -> global connectivity
#
# Endogenous attractor creation, tension dynamics and exploratory
# drift toward the current attractor are intentionally excluded.
# ============================================================

# Spatial domain
GRID_WIDTH = 50
GRID_HEIGHT = 50

# Population
N_AGENTS = 200
EXPLORATORY_RATIO = 0.10
BERNOULLI_EXPLORER_ASSIGNMENT = True

# Fixed attractor landscape
ATTRACTOR_RADIUS = 8.0
ATTRACTOR_STRENGTH = 0.30
INTERFACE_THRESHOLD = 3.0

# "structured": embedded agents follow the nearest-attractor gradient.
# "flat_control": attractor labels remain available only for operational
# state assignment and transition measurement; they exert no force on movement.
LANDSCAPE_MODE = "structured"

# Movement amplitudes
EMBEDDED_MOVE_SCALE = 0.5
EXPLORATORY_MOVE_SCALE = 2.0
EXPLORATORY_RULE = "random_walk"

# Candidate-selection parameters
N_CANDIDATES = 5
INTERFACE_BONUS_VALUE = 0.5
NOVELTY_BONUS = 1.0

# Available modes used by experiment scripts: "real" or "none".
INTERFACE_BONUS_MODE = "real"

# Infinitesimal random term used only to break exact score ties.
CANDIDATE_TIE_EPSILON = 1e-9

# Simulation protocol
MAX_STEPS = 300
N_RUNS = 20
RANDOM_SEED = 42
CI_Z = 1.96
