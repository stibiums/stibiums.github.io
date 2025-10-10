#!/usr/bin/env python3
"""
Generate illustrations for CV Chapter 7: Epipolar Geometry
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {
    'camera': '#2E86AB',
    'point': '#A23B72',
    'line': '#F18F01',
    'plane': '#C73E1D',
    'epipole': '#06A77D'
}

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def draw_camera_2d(ax, center, direction, size=0.3, color='blue', label=''):
    """Draw a simple 2D camera icon"""
    # Camera body
    rect = Rectangle((center[0]-size/2, center[1]-size/3),
                     size, size*2/3,
                     facecolor=color, alpha=0.7, edgecolor='black', linewidth=2)
    ax.add_patch(rect)

    # Lens direction
    if direction == 'right':
        ax.arrow(center[0]+size/2, center[1], size*0.5, 0,
                head_width=0.15, head_length=0.1, fc=color, ec='black', linewidth=1.5)
    elif direction == 'left':
        ax.arrow(center[0]-size/2, center[1], -size*0.5, 0,
                head_width=0.15, head_length=0.1, fc=color, ec='black', linewidth=1.5)

    if label:
        ax.text(center[0], center[1]-size/2-0.2, label,
               fontsize=14, fontweight='bold', ha='center')

def draw_image_plane(ax, x, y_range, color='gray', alpha=0.3, label=''):
    """Draw a vertical image plane"""
    ax.fill_between([x, x], [y_range[0], y_range[0]],
                    [y_range[1], y_range[1]],
                    color=color, alpha=alpha, linewidth=2, edgecolor='black')
    if label:
        ax.text(x, y_range[1]+0.2, label, fontsize=12, ha='center', fontweight='bold')

def generate_single_view_ambiguity():
    """Generate illustration of single view depth ambiguity"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Camera at origin
    camera_pos = np.array([0, 0])
    draw_camera_2d(ax, camera_pos, 'right', size=0.4, color=colors['camera'], label='O')

    # Image plane
    image_x = 2
    draw_image_plane(ax, image_x, [-2, 2], alpha=0.2, label='Image Plane')

    # Multiple 3D points projecting to same image point
    image_point_y = 0.8
    ax.plot(image_x, image_point_y, 'o', color=colors['point'], markersize=12, zorder=5)
    ax.text(image_x+0.3, image_point_y, 'x', fontsize=14, fontweight='bold')

    # Ray from camera through image point
    ray_end = 6
    ax.plot([0, ray_end], [0, ray_end*image_point_y/image_x],
           '--', color=colors['line'], linewidth=2, alpha=0.7)

    # Multiple possible 3D points
    depths = [3, 4, 5]
    for i, d in enumerate(depths):
        X = d * image_point_y / image_x
        ax.plot(d, X, 'o', color=colors['point'], markersize=10, alpha=0.8)
        ax.text(d, X+0.3, f'X{i+1}', fontsize=12)
        # Dashed line from 3D point to image
        ax.plot([d, image_x], [X, image_point_y], ':',
               color=colors['point'], linewidth=1.5, alpha=0.5)

    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Z (depth)', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Single View Depth Ambiguity', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('single_view_ambiguity.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: single_view_ambiguity.png")

def generate_baseline():
    """Generate baseline illustration"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Two cameras
    O = np.array([0, 0])
    O_prime = np.array([4, 0])

    draw_camera_2d(ax, O, 'right', size=0.4, color=colors['camera'], label='O')
    draw_camera_2d(ax, O_prime, 'left', size=0.4, color=colors['camera'], label="O'")

    # Baseline
    ax.plot([O[0], O_prime[0]], [O[1], O_prime[1]],
           color=colors['line'], linewidth=4, label='Baseline')
    ax.annotate('', xy=O_prime, xytext=O,
               arrowprops=dict(arrowstyle='<->', color=colors['line'], lw=3))
    ax.text(2, -0.5, 'Baseline', fontsize=14, ha='center', fontweight='bold',
           color=colors['line'])

    # Image planes
    draw_image_plane(ax, 1, [-2, 2], alpha=0.15)
    draw_image_plane(ax, 3, [-2, 2], alpha=0.15)

    ax.set_xlim(-1, 5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Stereo Camera Baseline', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('baseline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: baseline.png")

def generate_epipoles():
    """Generate epipoles illustration"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Two cameras
    O = np.array([0, 0])
    O_prime = np.array([5, 0])

    draw_camera_2d(ax, O, 'right', size=0.4, color=colors['camera'], label='O')
    draw_camera_2d(ax, O_prime, 'left', size=0.4, color=colors['camera'], label="O'")

    # Baseline
    ax.plot([O[0], O_prime[0]], [O[1], O_prime[1]],
           color=colors['line'], linewidth=3, alpha=0.6)

    # Image planes
    img1_x = 1.5
    img2_x = 3.5
    draw_image_plane(ax, img1_x, [-2.5, 2.5], alpha=0.2, label='Image 1')
    draw_image_plane(ax, img2_x, [-2.5, 2.5], alpha=0.2, label='Image 2')

    # Epipoles (baseline intersects image planes)
    e_y = 0  # at same height as cameras
    e_prime_y = 0

    ax.plot(img1_x, e_y, 'o', color=colors['epipole'], markersize=15, zorder=5)
    ax.text(img1_x+0.3, e_y+0.3, 'e', fontsize=16, fontweight='bold',
           color=colors['epipole'])

    ax.plot(img2_x, e_prime_y, 'o', color=colors['epipole'], markersize=15, zorder=5)
    ax.text(img2_x-0.5, e_prime_y+0.3, "e'", fontsize=16, fontweight='bold',
           color=colors['epipole'])

    # Extended baseline for clarity
    ax.plot([-0.5, img1_x], [0, 0], '--', color=colors['line'], linewidth=2, alpha=0.5)
    ax.plot([img2_x, 5.5], [0, 0], '--', color=colors['line'], linewidth=2, alpha=0.5)

    ax.set_xlim(-1, 6)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Epipoles: Baseline Intersection with Image Planes',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('epipoles.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: epipoles.png")

def generate_epipolar_plane():
    """Generate epipolar plane illustration"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Camera centers
    O = np.array([0, 0, 0])
    O_prime = np.array([5, 0, 0])

    # 3D point
    X = np.array([2.5, 3, 2])

    # Draw cameras
    ax.scatter(*O, color=colors['camera'], s=200, marker='o', label='O')
    ax.scatter(*O_prime, color=colors['camera'], s=200, marker='s', label="O'")
    ax.scatter(*X, color=colors['point'], s=200, marker='^', label='X')

    # Baseline
    ax.plot([O[0], O_prime[0]], [O[1], O_prime[1]], [O[2], O_prime[2]],
           color=colors['line'], linewidth=3, label='Baseline')

    # Epipolar plane (triangle formed by O, O', X)
    vertices = np.array([O, O_prime, X, O])
    ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2],
           color=colors['plane'], linewidth=2)

    # Fill the plane
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    plane = Poly3DCollection([vertices[:-1]], alpha=0.3,
                            facecolor=colors['plane'], edgecolor=colors['plane'])
    ax.add_collection3d(plane)

    # Lines from cameras to X
    ax.plot([O[0], X[0]], [O[1], X[1]], [O[2], X[2]],
           '--', color=colors['point'], linewidth=2, alpha=0.7)
    ax.plot([O_prime[0], X[0]], [O_prime[1], X[1]], [O_prime[2], X[2]],
           '--', color=colors['point'], linewidth=2, alpha=0.7)

    # Labels
    ax.text(*O, '  O', fontsize=12, fontweight='bold')
    ax.text(*O_prime, "  O'", fontsize=12, fontweight='bold')
    ax.text(*X, '  X', fontsize=12, fontweight='bold')

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('Epipolar Plane', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('epipolar_plane.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: epipolar_plane.png")

def generate_epipolar_lines():
    """Generate epipolar lines illustration"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Setup for both images
    for ax, title in zip([ax1, ax2], ['Image 1', 'Image 2']):
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.5, 3.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Image frame
        rect = Rectangle((0, 0), 4, 3, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(rect)

    # Epipoles
    e1 = np.array([3.5, 1.5])
    e2 = np.array([0.5, 1.5])

    ax1.plot(*e1, 'o', color=colors['epipole'], markersize=12, zorder=5)
    ax1.text(e1[0]-0.3, e1[1]+0.3, 'e', fontsize=14, fontweight='bold')

    ax2.plot(*e2, 'o', color=colors['epipole'], markersize=12, zorder=5)
    ax2.text(e2[0]+0.3, e2[1]+0.3, "e'", fontsize=14, fontweight='bold')

    # Corresponding points and epipolar lines
    points1 = np.array([[1, 2], [1.5, 1], [2, 2.5]])

    for i, p1 in enumerate(points1):
        # Point in image 1
        ax1.plot(*p1, 'o', color=colors['point'], markersize=10)
        ax1.text(p1[0]-0.3, p1[1]+0.2, f'x{i+1}', fontsize=12)

        # Epipolar line in image 1 (through e1 and p1)
        direction1 = e1 - p1
        t_vals = np.linspace(-0.5, 1.5, 100)
        line1 = p1[:, np.newaxis] + direction1[:, np.newaxis] * t_vals
        ax1.plot(line1[0], line1[1], '--', color=colors['line'],
                linewidth=1.5, alpha=0.5)

        # Corresponding epipolar line in image 2
        # Simulate corresponding line through e2
        angle = np.arctan2(direction1[1], direction1[0]) + np.random.uniform(-0.3, 0.3)
        direction2 = np.array([np.cos(angle), np.sin(angle)])
        line2 = e2[:, np.newaxis] + direction2[:, np.newaxis] * np.linspace(-2, 2, 100)
        ax2.plot(line2[0], line2[1], '-', color=colors['line'],
                linewidth=2, alpha=0.7, label=f"l'{i+1}")

    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('epipolar_lines.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: epipolar_lines.png")

def generate_parallel_cameras():
    """Generate parallel camera configuration"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Two cameras at same height
    O = np.array([0, 0])
    O_prime = np.array([4, 0])

    draw_camera_2d(ax, O, 'right', size=0.4, color=colors['camera'], label='O')
    draw_camera_2d(ax, O_prime, 'right', size=0.4, color=colors['camera'], label="O'")

    # Parallel optical axes
    ax.arrow(O[0]+0.2, O[1], 2, 0, head_width=0.15, head_length=0.1,
            fc='gray', ec='black', linewidth=1.5, alpha=0.6)
    ax.arrow(O_prime[0]+0.2, O_prime[1], 2, 0, head_width=0.15, head_length=0.1,
            fc='gray', ec='black', linewidth=1.5, alpha=0.6)

    # Image planes (parallel and vertical)
    img1_x = 1.2
    img2_x = 5.2
    draw_image_plane(ax, img1_x, [-2, 2], alpha=0.15)
    draw_image_plane(ax, img2_x, [-2, 2], alpha=0.15)

    # Epipoles at infinity (arrows pointing left and right)
    ax.annotate('', xy=(-1.5, 0), xytext=(-0.8, 0),
               arrowprops=dict(arrowstyle='<-', color=colors['epipole'], lw=3))
    ax.text(-1.5, -0.4, 'e at ∞', fontsize=12, fontweight='bold', color=colors['epipole'])

    ax.annotate('', xy=(7.5, 0), xytext=(6.8, 0),
               arrowprops=dict(arrowstyle='->', color=colors['epipole'], lw=3))
    ax.text(7, -0.4, "e' at ∞", fontsize=12, fontweight='bold', color=colors['epipole'])

    # Parallel epipolar lines
    for y in [-1.2, 0, 1.2]:
        ax.plot([img1_x, img1_x+0.8], [y, y], '-',
               color=colors['line'], linewidth=2, alpha=0.6)
        ax.plot([img2_x, img2_x+0.8], [y, y], '-',
               color=colors['line'], linewidth=2, alpha=0.6)

    ax.set_xlim(-2, 8)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Parallel Camera Configuration', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('parallel_cameras.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: parallel_cameras.png")

def generate_converging_cameras():
    """Generate converging camera configuration"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Two cameras
    O = np.array([0, -1])
    O_prime = np.array([4, 1])

    draw_camera_2d(ax, O, 'right', size=0.4, color=colors['camera'], label='O')
    draw_camera_2d(ax, O_prime, 'left', size=0.4, color=colors['camera'], label="O'")

    # Converging optical axes
    meeting_point = np.array([3, 0])
    ax.plot([O[0], meeting_point[0]], [O[1], meeting_point[1]],
           '--', color='gray', linewidth=2, alpha=0.6)
    ax.plot([O_prime[0], meeting_point[0]], [O_prime[1], meeting_point[1]],
           '--', color='gray', linewidth=2, alpha=0.6)

    # Image planes (not parallel)
    img1_x = 1
    img2_x = 3

    # Angled image planes
    ax.plot([img1_x, img1_x+0.2], [-2, 2], color='black', linewidth=2)
    ax.plot([img2_x-0.2, img2_x], [-2, 2], color='black', linewidth=2)
    ax.fill_between([img1_x, img1_x+0.2], [-2, -2], [2, 2],
                    color='gray', alpha=0.15)
    ax.fill_between([img2_x-0.2, img2_x], [-2, -2], [2, 2],
                    color='gray', alpha=0.15)

    # Baseline
    ax.plot([O[0], O_prime[0]], [O[1], O_prime[1]],
           color=colors['line'], linewidth=3, alpha=0.6)

    # Epipoles (visible in images)
    e = np.array([img1_x, 0.5])
    e_prime = np.array([img2_x, -0.3])

    ax.plot(*e, 'o', color=colors['epipole'], markersize=12, zorder=5)
    ax.text(e[0]+0.3, e[1], 'e', fontsize=14, fontweight='bold', color=colors['epipole'])

    ax.plot(*e_prime, 'o', color=colors['epipole'], markersize=12, zorder=5)
    ax.text(e_prime[0]-0.5, e_prime[1], "e'", fontsize=14, fontweight='bold',
           color=colors['epipole'])

    # Radial epipolar lines
    for angle in np.linspace(30, 150, 5):
        rad = np.radians(angle)
        dx, dy = np.cos(rad), np.sin(rad)
        ax.plot([e[0], e[0]+dx*1.5], [e[1], e[1]+dy*1.5],
               '-', color=colors['line'], linewidth=1.5, alpha=0.5)

    for angle in np.linspace(200, 340, 5):
        rad = np.radians(angle)
        dx, dy = np.cos(rad), np.sin(rad)
        ax.plot([e_prime[0], e_prime[0]+dx*1.5], [e_prime[1], e_prime[1]+dy*1.5],
               '-', color=colors['line'], linewidth=1.5, alpha=0.5)

    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Converging Camera Configuration', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('converging_cameras.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: converging_cameras.png")

def generate_forward_motion():
    """Generate forward motion configuration"""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Camera at two positions along z-axis
    O = np.array([0, 0])
    O_prime = np.array([0, -1.5])

    # Draw cameras pointing right
    draw_camera_2d(ax, O, 'right', size=0.3, color=colors['camera'], label='O')
    draw_camera_2d(ax, O_prime, 'right', size=0.3, color=colors['camera'], label="O'")

    # Motion arrow
    ax.annotate('', xy=O_prime, xytext=O,
               arrowprops=dict(arrowstyle='->', color='red', lw=4))
    ax.text(-0.5, -0.75, 'Motion', fontsize=12, fontweight='bold', color='red')

    # Image plane
    img_x = 2
    ax.plot([img_x, img_x], [-2.5, 2.5], color='black', linewidth=2)
    ax.fill_between([img_x, img_x+0.1], [-2.5, -2.5], [2.5, 2.5],
                    color='gray', alpha=0.15)

    # Epipole at center (principal point)
    e = np.array([img_x, 0])
    ax.plot(*e, 'o', color=colors['epipole'], markersize=15, zorder=5)
    ax.text(e[0]+0.3, e[1], 'e = e\'', fontsize=14, fontweight='bold',
           color=colors['epipole'])
    ax.text(e[0]+0.3, e[1]-0.4, '(principal point)', fontsize=10,
           color=colors['epipole'])

    # Radial epipolar lines from center
    for angle in np.linspace(0, 360, 16, endpoint=False):
        rad = np.radians(angle)
        dx, dy = np.cos(rad), np.sin(rad)
        ax.plot([e[0], e[0]+dx*2], [e[1], e[1]+dy*2],
               '-', color=colors['line'], linewidth=1.5, alpha=0.6)

    ax.set_xlim(-1.5, 4.5)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Forward Motion: Epipole at Principal Point',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('forward_motion.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: forward_motion.png")

def generate_geometric_error():
    """Generate geometric vs algebraic error illustration"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Epipolar line
    x_line = np.linspace(0, 4, 100)
    y_line = 0.5 * x_line + 0.5

    # Setup both subplots
    for ax, title in zip([ax1, ax2],
                        ['Algebraic Error', 'Geometric Error (Better)']):
        ax.plot(x_line, y_line, '-', color=colors['line'], linewidth=3, label='Epipolar line')
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.5, 3.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)

    # Point
    point = np.array([2, 2.5])

    # Algebraic error (vertical)
    ax1.plot(*point, 'o', color=colors['point'], markersize=12, label='Point x\'')
    algebraic_proj = np.array([point[0], 0.5*point[0] + 0.5])
    ax1.plot([point[0], point[0]], [point[1], algebraic_proj[1]],
            'r--', linewidth=3, label='Algebraic error')
    ax1.arrow(point[0]+0.1, (point[1]+algebraic_proj[1])/2, 0.5, 0,
             head_width=0.15, head_length=0.1, fc='red', ec='red')

    # Geometric error (perpendicular)
    ax2.plot(*point, 'o', color=colors['point'], markersize=12, label='Point x\'')
    # Find perpendicular projection
    line_vec = np.array([1, 0.5])
    line_vec = line_vec / np.linalg.norm(line_vec)
    point_vec = point - np.array([0, 0.5])
    proj_length = np.dot(point_vec, line_vec)
    geometric_proj = np.array([0, 0.5]) + proj_length * line_vec

    ax2.plot(*geometric_proj, 's', color='green', markersize=10, label='Projection')
    ax2.plot([point[0], geometric_proj[0]], [point[1], geometric_proj[1]],
            'g--', linewidth=3, label='Geometric error')

    # Perpendicular mark
    perp_size = 0.15
    perp_vec = np.array([-line_vec[1], line_vec[0]]) * perp_size
    corner = geometric_proj + perp_vec
    ax2.plot([geometric_proj[0], corner[0], corner[0]+line_vec[0]*perp_size],
            [geometric_proj[1], corner[1], corner[1]+line_vec[1]*perp_size],
            'k-', linewidth=1)

    ax1.legend(fontsize=10)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('geometric_error.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: geometric_error.png")

def generate_rank2_constraint():
    """Generate rank-2 constraint illustration"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Simulate SVD singular values
    sigma_init = np.array([3.2, 2.1, 0.8])
    sigma_final = np.array([3.2, 2.1, 0.0])

    x = np.arange(3)
    width = 0.35

    # Initial F
    bars1 = ax1.bar(x, sigma_init, width, color=colors['camera'], alpha=0.7,
                    edgecolor='black', linewidth=2)
    ax1.set_ylabel('Singular Values', fontsize=12)
    ax1.set_xlabel('Index', fontsize=12)
    ax1.set_title('Initial F Estimate (Full Rank)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['σ₁', 'σ₂', 'σ₃'])
    ax1.set_ylim(0, 3.5)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars1, sigma_init)):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.1,
                f'{val:.1f}', ha='center', fontweight='bold', fontsize=11)

    # Final F (rank 2)
    bars2 = ax2.bar(x, sigma_final, width, color=colors['camera'], alpha=0.7,
                    edgecolor='black', linewidth=2)
    bars2[2].set_color('red')
    bars2[2].set_alpha(0.3)

    ax2.set_ylabel('Singular Values', fontsize=12)
    ax2.set_xlabel('Index', fontsize=12)
    ax2.set_title('Final F (Rank 2)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['σ₁', 'σ₂', 'σ₃'])
    ax2.set_ylim(0, 3.5)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars2, sigma_final)):
        if i < 2:
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.1,
                    f'{val:.1f}', ha='center', fontweight='bold', fontsize=11)
        else:
            ax2.text(bar.get_x() + bar.get_width()/2, 0.2,
                    '0.0', ha='center', fontweight='bold', fontsize=11, color='red')

    # Arrow showing the transformation
    fig.text(0.5, 0.95, 'SVD → Set σ₃ = 0 → Reconstruct',
            ha='center', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('rank2_constraint.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: rank2_constraint.png")

def generate_four_solutions():
    """Generate four possible camera configurations"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    configs = [
        ('R₁, t', [0, 0], [3, 0], [1.5, 1.5]),
        ('R₁, -t', [0, 0], [-3, 0], [1.5, 1.5]),
        ('R₂, t', [0, 0], [3, 0], [1.5, -1.5]),
        ('R₂, -t', [0, 0], [-3, 0], [1.5, -1.5])
    ]

    for ax, (title, O, t, X) in zip(axes, configs):
        O = np.array(O)
        O_prime = np.array(t)
        X = np.array(X)

        # Camera 1
        ax.scatter(*O, color=colors['camera'], s=200, marker='o', zorder=5)
        ax.text(O[0]-0.3, O[1]-0.4, 'O', fontsize=12, fontweight='bold')

        # Camera 2
        ax.scatter(*O_prime, color=colors['camera'], s=200, marker='s', zorder=5)
        ax.text(O_prime[0]+0.2, O_prime[1]-0.4, "O'", fontsize=12, fontweight='bold')

        # 3D point
        ax.scatter(*X, color=colors['point'], s=150, marker='^', zorder=5)
        ax.text(X[0]+0.2, X[1]+0.2, 'X', fontsize=12, fontweight='bold')

        # Rays to point
        ax.plot([O[0], X[0]], [O[1], X[1]], '--',
               color=colors['point'], linewidth=2, alpha=0.6)
        ax.plot([O_prime[0], X[0]], [O_prime[1], X[1]], '--',
               color=colors['point'], linewidth=2, alpha=0.6)

        # Baseline
        ax.plot([O[0], O_prime[0]], [O[1], O_prime[1]],
               color=colors['line'], linewidth=3, alpha=0.6)

        # Check if valid (X in front of both cameras)
        # Simplified check: X.y should be positive for valid config
        if X[1] > 0 and ((O_prime[0] > O[0] and X[0] > O[0]) or
                        (O_prime[0] < O[0] and X[0] < O_prime[0])):
            valid = True
            color = 'green'
            status = '✓ Valid'
        else:
            valid = False
            color = 'red'
            status = '✗ Invalid'

        ax.set_xlim(-4, 4)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{title}: {status}', fontsize=12, fontweight='bold', color=color)
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)

    plt.suptitle('Four Possible Solutions from Essential Matrix',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('four_solutions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: four_solutions.png")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Generating Epipolar Geometry Illustrations")
    print("="*60 + "\n")

    generate_single_view_ambiguity()
    generate_baseline()
    generate_epipoles()
    generate_epipolar_plane()
    generate_epipolar_lines()
    generate_parallel_cameras()
    generate_converging_cameras()
    generate_forward_motion()
    generate_geometric_error()
    generate_rank2_constraint()
    generate_four_solutions()

    print("\n" + "="*60)
    print("All images generated successfully!")
    print("="*60 + "\n")
