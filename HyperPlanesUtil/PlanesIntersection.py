import numpy as np
from sympy import symbols, Eq, solve
from scipy.optimize import fsolve


def isCoincident(n1, n2, d1, d2):
    """
        Check if two hyperplanes defined by their normal vectors are coincident.

        Args:
            n1 (array-like): Normal vector of the first hyperplane.
            n2 (array-like): Normal vector of the second hyperplane.
            d1 (float): The distance from the origin along the first hyperplane's normal vector.
            d2 (float): The distance from the origin along the second hyperplane's normal vector.
        Returns:
            bool: True if the hyperplanes are coincident, False otherwise.
        """

    if d2 == 0:  # Check if d2 is zero
        return False  # Hyperplanes are not coincident if d2 is zero

    fraction = d1/d2
    coincident = all(c1 / c2 == fraction for c1, c2 in zip(n1, n2))
    return coincident


def find_intersection_hyperplane(n1, n2, d1, d2, w_base, w_inc):
    """
        Find the intersection point of two non-parallel hyperplanes.

        Args:
            n1 (array-like): Normal vector of the first hyperplane.
            n2 (array-like): Normal vector of the second hyperplane.
            d1 (float): Distance of the first hyperplane from the origin.
            d2 (float): Distance of the second hyperplane from the origin.
            w_base (float): Weight of the first hyperplane for weighted midpoint calculation.
            w_inc (float): Weight of the second hyperplane for weighted midpoint calculation.

        Returns:
            list: Intersection point coordinates [x0, x1, ..., xn] if the hyperplanes intersect,
                  or the midpoint if they are parallel.
        """
    variables = symbols(' '.join([f'x{i}' for i in range(len(n1))]))
    equation1 = Eq(sum(var * nn1 for var, nn1 in zip(variables, n1)), -d1)
    equation2 = Eq(sum(var * nn2 for var, nn2 in zip(variables, n2)), -d2)
    sol_dict = solve((equation1, equation2), (variables))

    if not sol_dict:  # parallel - no intersection point.
        return parallel_find_mid_weighted_point(n1, n2, d1, d2, w_base, w_inc)

    x0 = symbols('x0')
    x1 = symbols('x1')

    x0Val = sol_dict[x0].args[0] if (len(sol_dict[x0].args) != 0) else sol_dict[x0]
    x1Val = sol_dict[x1].args[0] if (len(sol_dict[x1].args) != 0) else sol_dict[x1]

    result = [x0Val, x1Val]
    return result


def find_intersection_hyperplaneND(n1, n2, d1, d2, w_base, w_inc):
    """
        Find the intersection point of two hyperplanes in N-dimensional space.

        Args:
            n1 (array-like): Normal vector of the first hyperplane.
            n2 (array-like): Normal vector of the second hyperplane.
            d1 (float): Distance of the first hyperplane from the origin.
            d2 (float): Distance of the second hyperplane from the origin.
            w_base (float): Weight of the first hyperplane for weighted midpoint calculation.
            w_inc (float): Weight of the second hyperplane for weighted midpoint calculation.

        Returns:
            list: Intersection point coordinates [x0, x1, ..., xn] if the hyperplanes intersect,
                  or the midpoint if they are parallel.
        """
    nLength = len(n1)
    if (nLength == 2):
        return find_intersection_hyperplane(n1, n2, d1, d2, w_base, w_inc)

    if are_parallel_planes(n1, n2):
        return parallel_find_mid_weighted_point(n1, n2, d1, d2, w_base, w_inc)

    variables = symbols(' '.join([f'x{i}' for i in range(nLength)]))
    n1 = n1[1:]
    n2 = n2[1:]

    # Define the system of equations for the intersection point
    def equations(variables):
        variables1 = sum(var * nn1 for var, nn1 in zip(variables, n1)) + d1
        variables2 = sum(var * nn2 for var, nn2 in zip(variables, n2)) + d2
        return [variables1, variables2]

    guess = np.ones(2)
    intersection_point = fsolve(equations, guess)

    return [0] + intersection_point.tolist() + [0] * (nLength - 3)


def parallel_find_mid_weighted_point(n1, n2, d1, d2, w_base, w_inc):
    """
        Calculate the midpoint between two parallel hyperplanes using weighted averages.

        Args:
            n1 (array-like): Normal vector of the first hyperplane.
            n2 (array-like): Normal vector of the second hyperplane.
            d1 (float): Distance of the first hyperplane from the origin.
            d2 (float): Distance of the second hyperplane from the origin.
            w_base (float): Weight of the first hyperplane for weighted midpoint calculation.
            w_inc (float): Weight of the second hyperplane for weighted midpoint calculation.

        Returns:
            list: Midpoint coordinates [x0, x1, ..., xn] between the two parallel hyperplanes.
        """
    w_base_normalized = w_base / (w_base + w_inc)
    w_inc_normalized = w_inc / (w_base + w_inc)
    y_midpoint_line1 = -(d1) / n1[1]
    y_midpoint_line2 = -(d2) / n2[1]
    y_midpoint = (y_midpoint_line1 * w_base_normalized + y_midpoint_line2 * w_inc_normalized) / (
            w_base_normalized + w_inc_normalized)
    return [0, y_midpoint] + [0] * (len(n1) - 2)


def are_parallel_planes(n1, n2, tol=1e-12):
    """
        Check if two hyperplanes defined by their normal vectors are parallel.

        Args:
            n1 (array-like): Normal vector of the first hyperplane.
            n2 (array-like): Normal vector of the second hyperplane.
            tol (float, optional): Tolerance for angle comparison.

        Returns:
            bool: True if the hyperplanes are parallel, False otherwise.
        """
    # Check if the normal vectors are parallel
    n1 = np.array(n1)
    n2 = np.array(n2)

    # Normalize the vectors
    normalized_n1 = n1 / np.linalg.norm(n1)
    normalized_n2 = n2 / np.linalg.norm(n2)

    # Check if the normalized vectors are parallel
    dot_product = np.dot(normalized_n1, normalized_n2)
    angle_diff = np.arccos(np.clip(dot_product, -1.0, 1.0))

    return np.isclose(angle_diff, 0.0, atol=tol)



