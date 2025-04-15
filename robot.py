import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from utils import safe_divide
import warnings


class ContinuumRobot:
    def __init__(self, control_points, degree=3, segment_count=3):
        self.control_points = np.array(control_points)
        self.p = degree
        self.segment_count = segment_count
        self.n = len(control_points) - 1
        self.update_knot_vector()

        self.b_memo = {}  # Memoization for basis functions
        self.db_memo = {}  # Memoization for derivative of basis functions

        self.obstacles = []  # List of obstacles (center, radius)
        self.robot_radius = 0.05 # Radius of the robot (for collision detection)

    def update_knot_vector(self):
        self.n = len(self.control_points) - 1
        self.knot_vector = self._generate_knot_vector(self.n, self.p)

    def _generate_knot_vector(self, n, p):
        if n > p:
            internal_knots = np.linspace(
                0, 1, n-p+2)[1:-1]  # exclude extra 0 and 1
        else:
            internal_knots = np.array([])
        knot_vector = np.concatenate((
            np.zeros(p+1),
            internal_knots,
            np.ones(p+1)
        ))
        return knot_vector

    # original 'typical' de Boor function
    def _basis_functions(self, u, i, p=None):
        U = self.knot_vector
        if p is None:
            p = self.p

        if (u, i, p) in self.b_memo:
            return self.b_memo[(u, i, p)]

        if p == 0:
            result = 1.0 if U[i] <= u < U[i +
                                          1] or (abs(u - U[-1]) < 1e-10 and i == len(U) - p - 2) else 0.0
            self.b_memo[(u, i, p)] = result
            return result
        else:
            coef1 = safe_divide(u - U[i], U[i + p] - U[i])
            coef2 = safe_divide(U[i + p + 1] - u, U[i + p + 1] - U[i + 1])
            result = coef1 * \
                self._basis_functions(u, i, p - 1) + coef2 * \
                self._basis_functions(u, i + 1, p - 1)
            self.b_memo[(u, i, p)] = result
            return result

    def _find_span(self, u, p, U, n):
        # special case for endpoint
        if abs(u - U[-1]) < 1e-10:
            return n

        # binary search
        low, high = p, n+1
        mid = (low + high) // 2

        while u < U[mid] or u >= U[mid+1]:
            if u < U[mid]:
                high = mid
            else:
                low = mid
            mid = (low + high) // 2

        return mid

    # non-recursive version of de Boor function for efficency
    def _basis_functions_efficient(self, span, u, p, U):
        N = np.zeros(p+1)
        left = np.zeros(p+1)
        right = np.zeros(p+1)
        N[0] = 1.0
        for j in range(1, p+1):
            left[j] = u - U[span+1-j]
            right[j] = U[span+j] - u
            saved = 0.0
            for r in range(j):
                temp = N[r] / (right[r+1] + left[j-r])
                N[r] = saved + right[r+1] * temp
                saved = left[j-r] * temp
            N[j] = saved
        return N

    # def _basis_function_derivative(self, u, i, p=None):
        # """
        # Calculate the derivative of basis function N_i,p at parameter u.

        # Args:
        #     u: Parameter value
        #     i: Index of control point
        #     p: Degree of the basis function (default: uses the robot's degree)

        # Returns:
        #     Value of the basis function derivative at u
        # """
        # U = self.knot_vector
        # if p is None:
        #     p = self.p

        # # Use memoized value if available
        # if (u, i, p) in self.db_memo:
        #     return self.db_memo[(u, i, p)]

        # # Base case
        # if p == 0:
        #     result = 0.0
        # else:
        #     # Recursive case for derivative
        #     coef1 = safe_divide(p, U[i + p] - U[i])
        #     coef2 = safe_divide(p, U[i + p + 1] - U[i + 1])
        #     term1 = coef1 * \
        #         self._basis_functions(
        #             u, i, p - 1) if abs(coef1) > 1e-10 else 0.0
        #     term2 = coef2 * \
        #         self._basis_functions(
        #             u, i + 1, p - 1) if abs(coef2) > 1e-10 else 0.0
        #     result = term1 - term2

        # self.db_memo[(u, i, p)] = result
        # return result

    # def _get_tangent_vector(self, u, control_points=None):
        # if control_points is None:
        #     control_points = self.control_points

        # tangent = np.zeros(3)
        # for i in range(self.n + 1):
        #     db_i = self._basis_function_derivative(u, i)
        #     tangent += db_i * control_points[i]

        # # Normalize tangent vector
        # norm = np.linalg.norm(tangent)
        # if norm > 1e-10:
        #     tangent = tangent / norm

        # return tangent

    def _get_point_on_curve(self, u, control_points=None):
        if control_points is None:
            control_points = self.control_points

        span = self._find_span(u, self.p, self.knot_vector, self.n)
        N = self._basis_functions_efficient(span, u, self.p, self.knot_vector)

        point = np.zeros(3)
        for i in range(self.p+1):
            idx = span-self.p+i
            if 0 <= idx <= self.n:  # Ensure index is valid
                point += N[i] * control_points[idx]
        return point

    def _get_tangent_vector(self, u, control_points=None, delta=0.0001):
        if control_points is None:
            control_points = self.control_points

        pt1 = self._get_point_on_curve(u, control_points)

        if u + delta > 1.0:
            pt2 = pt1
            pt1 = self._get_point_on_curve(u - delta, control_points)
        else:
            pt2 = self._get_point_on_curve(u + delta, control_points)

        tangent = pt2 - pt1
        norm = np.linalg.norm(tangent)
        
        tangent = tangent / norm
        
        return tangent

    def add_obstacle(self, center, radius):
        self.obstacles.append((np.array(center), radius))

    def check_obstacle_collision(self, curve_points):
        for point in curve_points:
            for obstacle in self.obstacles:
                center, radius = obstacle
                if np.linalg.norm(point - center) < (radius + self.robot_radius):
                    return True
        return False

    def backbone_curve(self, control_points=None, num_points=100):
        if control_points is None:
            control_points = self.control_points
        u_values = np.linspace(0, 1, num_points)
        curve_points = [self._get_point_on_curve(
            u, control_points) for u in u_values]
        return np.array(curve_points)

    def forward_kinematics(self, control_points=None, num_points=100):
        backbone_curve = self.backbone_curve(control_points, num_points)
        ee_position = backbone_curve[-1]
        tangent = self._get_tangent_vector(1.0)
        return backbone_curve, ee_position, tangent

    def transformation_matrix(self, u):
        position = self._get_point_on_curve(u)
        tangent = self._get_tangent_vector(u)
        z_axis = tangent / \
            np.linalg.norm(tangent) if np.linalg.norm(
                tangent) > 1e-10 else np.array([0, 0, 1])
        if abs(z_axis[2]) < 0.9:
            x_axis = np.cross(np.array([0, 0, 1]), z_axis)
        else:
            x_axis = np.cross(np.array([0, 1, 0]), z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        R = np.column_stack((x_axis, y_axis, z_axis))
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = position
        return T

    def inverse_kinematics(self, target_position, max_iter=1000, ftol=1e-10):
        initial_cp = self.control_points.copy()

        def objective_function(flattened_control_points):
            n_points = len(flattened_control_points) // 3
            cp_flat = flattened_control_points.reshape(n_points, 3)

            control_points = np.vstack([initial_cp[0:1], cp_flat])

            backbone_curve, end_effector_position, _ = self.forward_kinematics(
                control_points)

            distance_penalty = 50.0 * \
                np.linalg.norm(end_effector_position - target_position)**2

            collision_penalty = 0.0
            if self.check_obstacle_collision(backbone_curve):
                collision_penalty = 1000.0

            smoothness_penalty = 0.0
            for i in range(1, len(control_points)):
                smoothness_penalty += 0.01 * \
                    np.linalg.norm(control_points[i] - control_points[i-1])**2

            total_cost = distance_penalty + collision_penalty + smoothness_penalty
            return total_cost

        initial_guess = initial_cp[1:].flatten()
        bounds = []
        for i in range(len(initial_guess)):
            # allow movement within a reasonable range
            bounds.append((initial_guess[i] - 5.0, initial_guess[i] + 5.0))

        print("Solving inverse kinematics...")
        result = minimize(
            objective_function,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': max_iter,
                'ftol': ftol,
                'disp': False
            }
        )
        cp_flat = result.x.reshape(-1, 3)
        optimized_control_points = np.vstack([initial_cp[0:1], cp_flat])
        self.control_points = optimized_control_points

        self.update_knot_vector()
        _, ee_position, _ = self.forward_kinematics()
        final_distance = np.linalg.norm(ee_position - target_position)

        print(
            f"Optimization complete. Final distance to target: {final_distance:.6f}")
        print(f"End effector position: {ee_position}")
        print(f"Target position: {target_position}")

        return optimized_control_points, final_distance
    
    def get_unique_knots(self):
        distinct_knots = []
        last_knot = None
        for i, knot in enumerate(self.knot_vector):
            if last_knot is None or abs(knot - last_knot) > 1e-10:
                if 0.0 < knot < 1.0:
                    distinct_knots.append(knot)
            last_knot = knot
        return distinct_knots
            

    def visualize(self, target_position=None, initial_control_points=None, show_frames=False, show_knots=True):
        view_elev=20
        view_azim=45
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        current_curve, ee_position, _ = self.forward_kinematics()
        if initial_control_points is not None:
            initial_curve, initial_ee, _ = self.forward_kinematics(
                initial_control_points)
            ax.plot(initial_curve[:, 0], initial_curve[:, 1], initial_curve[:, 2],
                    '--', color='blue', linewidth=1, label='Initial Backbone')
            ax.plot(initial_control_points[:, 0], initial_control_points[:, 1], initial_control_points[:, 2],
                    'o-', color='blue', alpha=0.5, label='Initial Control Points')
            ax.scatter(initial_ee[0], initial_ee[1], initial_ee[2],
                       color='blue', s=50, label='Initial End Effector')

        # current configuration
        ax.plot(current_curve[:, 0], current_curve[:, 1], current_curve[:, 2],
                '-', color='cyan', linewidth=2, label='Robot Backbone')
        ax.plot(self.control_points[:, 0], self.control_points[:, 1], self.control_points[:, 2],
                'o--', color='magenta', alpha=0.5, label='Control Points')
        ax.scatter(ee_position[0], ee_position[1], ee_position[2],
                   color='orange', s=100, label='End Effector')

        # target position
        if target_position is not None:
            ax.scatter(target_position[0], target_position[1], target_position[2],
                       color='red', s=100, marker='x', label='Target')

            # draw line from ee to target to show error
            ax.plot([ee_position[0], target_position[0]],
                    [ee_position[1], target_position[1]],
                    [ee_position[2], target_position[2]],
                    '--', color='red', alpha=0.7)

            # display numerical error (distance between ee and target)
            distance = np.linalg.norm(ee_position - target_position)
            ax.text(ee_position[0], ee_position[1], ee_position[2] + 0.2,
                    f'Distance: {distance:.4f}', color='red')

        # plot obstacles
        for obstacle in self.obstacles:
            center, radius = obstacle
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color='red', alpha=0.2)

        # set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Continuum Robot Visualization')
        ax.legend()
        ax.set_box_aspect([1, 1, 1])
        
        if show_knots:
            distinct_knots = self.get_unique_knots()
            knot_points = [self._get_point_on_curve(u) for u in distinct_knots]
            if len(knot_points) > 0:
                knot_points = np.array(knot_points)
                ax.scatter(knot_points[:, 0], knot_points[:, 1], knot_points[:, 2], 
                           color='black', s=100, marker='.', label='Knot Points')
                for i, (point, u) in enumerate(zip(knot_points, distinct_knots)):
                    ax.text(point[0], point[1], point[2] + 0.1, f'u={u:.2f}', color='green')
      
        if show_frames:
            u_values = [0.0] + self.get_unique_knots() + [1.0]
            frame_scale = 0.1
            for u in u_values:
                T = self.transformation_matrix(u)
                position = T[0:3, 3]
                x_axis = T[0:3, 0] * frame_scale
                y_axis = T[0:3, 1] * frame_scale
                z_axis = T[0:3, 2] * frame_scale
                ax.quiver(position[0], position[1], position[2],
                          x_axis[0], x_axis[1], x_axis[2],
                          color='r', arrow_length_ratio=0.2)
                ax.quiver(position[0], position[1], position[2],
                          y_axis[0], y_axis[1], y_axis[2],
                          color='g', arrow_length_ratio=0.2)
                ax.quiver(position[0], position[1], position[2],
                          z_axis[0], z_axis[1], z_axis[2],
                          color='b', arrow_length_ratio=0.2)

            global_scale = 0.1
            ax.quiver(0, 0, 0, global_scale, 0, 0, color='r',
                      arrow_length_ratio=0.2, label='Global X')
            ax.quiver(0, 0, 0, 0, global_scale, 0, color='g',
                      arrow_length_ratio=0.2, label='Global Y')
            ax.quiver(0, 0, 0, 0, 0, global_scale, color='b',
                      arrow_length_ratio=0.2, label='Global Z')

        ax.view_init(elev=view_elev, azim=view_azim)
        plt.tight_layout()
        plt.show()


def run_demo():
    initial_control_points = [
        [0.0, 0.0, 0.0],    
        [0.2, 0.0, 0.3],   
        [0.4, 0.0, 0.6],    
        [0.6, 0.0, 0.9],   
        [0.8, 0.0, 1.2],
        [1.0, 0.0, 1.5],   
    ]

    robot = ContinuumRobot(initial_control_points, degree=3)
    initial_cp = robot.control_points.copy()

    # robot.add_obstacle([0.3, 0.3, 0.4], 0.15)
    # robot.add_obstacle([0.7, 0.2, 0.6], 0.1)

    target_position = np.array([0.5, 0.5, 0.7])

    print("Transformation matrices along the robot backbone:")
    u_values = u_values = [0.0] + robot.get_unique_knots() + [1.0]
    for i, u in enumerate(u_values):
        T = robot.transformation_matrix(u)
        position = T[0:3, 3]
        print(f"\nPoint {i} (u={u}), Position: {position}")
        print("Transformation matrix:")
        print(np.array2string(T, precision=4, suppress_small=True))

    print("\nInitial configuration:")
    robot.visualize(target_position, initial_cp,
                    show_frames=True, show_knots=True)

    robot.inverse_kinematics(target_position)

    print("Final configuration:")
    robot.visualize(target_position, initial_cp,
                    show_frames=True, show_knots=True)

    print("\nTransformation matrices after inverse kinematics:")
    for i, u in enumerate(u_values):
        T = robot.transformation_matrix(u)
        position = T[0:3, 3]
        print(f"\nPoint {i} (u={u}), Position: {position}")
        print("Transformation matrix:")
        print(np.array2string(T, precision=4, suppress_small=True))

    return robot


if __name__ == "__main__":
    run_demo()
