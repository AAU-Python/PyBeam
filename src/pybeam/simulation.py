import logging

import numpy as np
import scipy.linalg as la

_LOGGER = logging.getLogger(__name__)


def newmark(
    *,
    stiffness: np.ndarray,
    mass: np.ndarray,
    time: np.ndarray,
    initial_disp: np.ndarray,
    initial_vel: np.ndarray,
    loads: np.ndarray | None = None,
    damping: np.ndarray | None = None,
    beta: float = 1 / 4,
    gamma: float = 1 / 2,
) -> np.ndarray:
    n_dof: int = stiffness.shape[0]

    if damping is None:
        _LOGGER.info("System is undamped.")
        damping = np.zeros((n_dof, n_dof))

    if loads is None:
        _LOGGER.info("There is no loading.")
        loads = np.zeros((n_dof, len(time)))

    if initial_disp is None:
        _LOGGER.info("Initial displacements are zero.")
        initial_disp = np.zeros((n_dof, 1))

    if initial_vel is None:
        _LOGGER.info("Initial velocities are zero.")
        initial_vel = np.zeros((n_dof, 1))

    if beta == 1 / 4:
        _LOGGER.info("Simulating with constant acceleration")
    elif beta == 1 / 6:
        _LOGGER.info("Simulating with linear acceleration")

    if not stiffness.shape == mass.shape:
        raise ValueError(f"Incorrect matrix dimensions, K ({stiffness.shape}), M ({mass.shape})")

    # Helper parameters
    dt = time[1] - time[0]
    a_1 = 1 / (beta * dt**2)
    a_2 = 1 / (beta * dt)
    a_3 = 1 / (2 * beta)
    a_4 = 1 / beta

    # Effective stiffness
    eff_stiff = a_1 * mass + a_2 * gamma * damping + stiffness

    displacements = np.zeros((n_dof, len(time)))
    displacements[:, 0] = initial_disp

    velocities = np.zeros((n_dof, len(time)))
    velocities[:, 0] = initial_vel

    accelerations = np.zeros((n_dof, len(time)))
    initial_acc = la.inv(mass).dot(loads[:, 0]) - damping.dot(initial_vel) - stiffness.dot(initial_disp)
    accelerations[:, 0] = initial_acc

    for i in range(len(time) - 1):
        a_eff = mass.dot(a_1 * displacements[:, i] + a_2 * velocities[:, i] + (a_3 - 1) * accelerations[:, i])
        v_eff = damping.dot(
            gamma * a_2 * displacements[:, i]
            + (gamma * a_4 - 1) * velocities[:, i]
            + dt * (gamma * a_3 - 1) * accelerations[:, i]
        )

        displacements[:, i + 1] = la.inv(eff_stiff).dot(loads[:, i + 1] + a_eff + v_eff)
        velocities[:, i + 1] = (
            gamma * a_2 * (displacements[:, i + 1] - displacements[:, i])
            - (gamma * a_4 - 1) * velocities[:, i]
            - dt * (gamma * a_3 - 1) * accelerations[:, i]
        )
        accelerations[:, i + 1] = (
            a_1 * (displacements[:, i + 1] - displacements[:, i] - dt * velocities[:, i])
            - (a_3 - 1) * accelerations[:, i]
        )

    return displacements, velocities, accelerations
