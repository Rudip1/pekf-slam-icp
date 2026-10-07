#!/usr/bin/python3
"""SE(2) pose type with compounding operators and their Jacobians.

Poses are 3x1 column vectors ``[x, y, theta]``. ``AxB`` reads "pose of frame B
expressed in frame A".
"""
import numpy as np
from math import cos, sin


class Pose(np.ndarray):
    """Abstract pose interface that :class:`Pose3D` implements."""

    def oplus(self, BxC):
        """Pose compounding ``self (+) BxC``. Implemented by subclasses."""
        pass

    def J_1oplus(self, BxC):
        """Jacobian of ``oplus`` w.r.t. the first pose. Implemented by subclasses."""
        pass

    def J_2oplus(self):
        """Jacobian of ``oplus`` w.r.t. the second pose. Implemented by subclasses."""
        pass

    def ominus(self):
        """Pose inversion ``(-) self``. Implemented by subclasses."""
        pass

    def J_ominus(self):
        """Jacobian of ``ominus``. Implemented by subclasses."""
        pass

    def ToCartesian(self):
        """Return the pose in Cartesian form (identity for this class)."""
        return self

    def J_2c(self):
        """Jacobian of :meth:`ToCartesian` (identity matrix)."""
        return np.eye(self.shape[0])


class Pose3D(Pose):
    """Planar robot pose ``[x, y, theta]`` (3 DOF) stored as a 3x1 ndarray.

    The operators are written as plain functions of their arguments, so the
    code base calls them unbound, e.g. ``Pose3D.oplus(AxB, BxC)`` with ordinary
    numpy arrays.
    """

    def __new__(cls, input_array=np.array([[0.0, 0.0, 0.0]]).T):
        """View ``input_array`` as a :class:`Pose3D` instance.

        :param input_array: 3x1 array ``[x, y, theta]``.
        :returns: the array viewed as a :class:`Pose3D`.
        """
        obj = np.asarray(input_array).view(cls)
        return obj

    def __init__(self, input_array=np.array([[0.0, 0.0, 0.0]]).T):
        """Check that the pose is a 3x1 vector."""
        assert input_array.shape == (3, 1), "mean must be a 3x1 vector"

    def oplus(AxB, BxC):
        """Compound two poses: ``AxC = AxB (+) BxC``.

        .. math::
            {}^Ax_C = \\begin{bmatrix}
                x_{AB} + x_{BC}\\cos\\theta_{AB} - y_{BC}\\sin\\theta_{AB} \\\\
                y_{AB} + x_{BC}\\sin\\theta_{AB} + y_{BC}\\cos\\theta_{AB} \\\\
                \\theta_{AB} + \\theta_{BC}
            \\end{bmatrix}

        The resulting angle is not wrapped.

        :param AxB: pose of B in A (3x1).
        :param BxC: pose of C in B (3x1).
        :returns: pose of C in A (3x1 ndarray).
        """
        AxC = np.array([[AxB[0,0] + BxC[0,0]*cos(AxB[2,0]) - BxC[1,0]*sin(AxB[2,0])],
                        [AxB[1,0] + BxC[0,0]*sin(AxB[2,0]) + BxC[1,0]*cos(AxB[2,0])],
                        [AxB[2,0] + BxC[2,0]]])
        return AxC

    def J_1oplus(AxB, BxC):
        """Jacobian of :meth:`oplus` with respect to the first pose ``AxB``.

        :param AxB: pose of B in A (3x1).
        :param BxC: pose of C in B (3x1).
        :returns: 3x3 Jacobian ``d(AxB (+) BxC) / d(AxB)``.
        """
        J1 = np.array([[1.0,    0.0,    -BxC[0,0]*sin(AxB[2,0])-BxC[1,0]*cos(AxB[2,0])],
                       [0.0,    1.0,     BxC[0,0]*cos(AxB[2,0])-BxC[1,0]*sin(AxB[2,0])],
                       [0.0,    0.0,     1.0]])
        return J1

    def J_2oplus(AxB):
        """Jacobian of :meth:`oplus` with respect to the second pose ``BxC``.

        It only depends on the heading of ``AxB`` (a rotation block).

        :param AxB: pose of B in A (3x1).
        :returns: 3x3 Jacobian ``d(AxB (+) BxC) / d(BxC)``.
        """
        J2 = np.array([[cos(AxB[2,0]),    -sin(AxB[2,0]),   0],
                       [sin(AxB[2,0]),     cos(AxB[2,0]),   0],
                       [0,                  0,             1.0]])
        return J2

    def ominus(AxB):
        """Invert a pose: ``BxA = (-) AxB``.

        :param AxB: pose of B in A (3x1).
        :returns: pose of A in B (3x1 ndarray).
        """
        AxB = np.array([[-AxB[0,0]*cos(AxB[2,0]) - AxB[1,0]*sin(AxB[2,0])],
                        [ AxB[0,0]*sin(AxB[2,0]) - AxB[1,0]*cos(AxB[2,0])],
                        [-AxB[2,0]]])
        return AxB

    def J_ominus(AxB):
        """Jacobian of :meth:`ominus` with respect to ``AxB``.

        :param AxB: pose of B in A (3x1).
        :returns: 3x3 Jacobian ``d((-) AxB) / d(AxB)``.
        """
        J = np.array([[-cos(AxB[2,0]),    -sin(AxB[2,0]),   AxB[0,0]*sin(AxB[2,0])-AxB[1,0]*cos(AxB[2,0])],
                      [ sin(AxB[2,0]),    -cos(AxB[2,0]),   AxB[0,0]*cos(AxB[2,0])+AxB[1,0]*sin(AxB[2,0])],
                      [ 0.0,               0.0,            -1.0]])
        return J

    def ToCartesian(self):
        """Return the pose in Cartesian form (identity for this class)."""
        return self

    def J_2c(self):
        """Jacobian of :meth:`ToCartesian` (identity matrix)."""
        return np.eye(self.shape[0])
