import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import pytransform3d
from pytransform3d.plot_utils import plot_mesh
from pytransform3d.transformations import plot_transform
from IPython import display


def quaternion_conjugate(q):
        return q * np.array([-1.0,-1.0,-1.0,1.0])

def quaternion_product(q1,q2):

    q12 = np.zeros(4)
    q12[-1] = q1[-1]*q2[-1] - np.dot(q1[:-1],q2[:-1])
    q12[:-1] = q1[-1]*q2[:-1] + q2[-1]*q1[:-1] + np.cross(q1[:-1],q2[:-1])
    return q12

def quaternion_error(q1,q2):
    return quaternion_product(q1,quaternion_conjugate(q2))

def exponential_map(r):

    theta = np.linalg.norm(r)
    if theta == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0])

    n = r / np.linalg.norm(r) 

    q = np.zeros(4)
    q[-1] = np.cos(theta / 2.0)
    q[:-1] = np.sin(theta/ 2.0) * n
    
    return q

def logarithmic_map(q):
    
    # As described in "Filtering in a unit quaternion space for model-based object tracking" by Ude

    if np.linalg.norm(q[:-1]) < np.finfo(float).eps:
        return np.zeros(3)

    n = q[:-1] / np.linalg.norm(q[:-1])
    theta = 2.0 * np.arctan2(np.linalg.norm(q[:-1]),q[-1])
    
    return theta*n # rotation vector

def quaternion_diff(q,dt):
    
    w = np.zeros([q.shape[0], 3])
    w[0,:] = logarithmic_map(quaternion_error(q[1,:], q[0,:])) / dt
    for n in range(1, q.shape[0]-1):
        w[n,:] = logarithmic_map(quaternion_error(q[n+1,:], q[n-1,:])) / (2.0*dt)
    w[-1,:] = logarithmic_map(quaternion_error(q[-1,:], q[-2,:])) / dt

    return w

def skew(w):
    return np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])

# Nonholonomic constraint equation
def nonholonomic_constraint_equation(dx,q):

    Rotmat = R.from_quat(q).as_matrix() # "scalar last" format
    yb = np.array([0,1,0]) # constraint vector in body frame
    A = np.dot(Rotmat,yb)  # constraint vector in spatial frame
    
    return np.dot(A, dx)

# Nonholonomic constraint forces 
def nonholonomic_constraint_force(q,dx,w,ddx): 

    Rotmat = R.from_quat(q).as_matrix() # "scalar last" format
    yb = np.array([0,1,0]) # constraint vector in body frame
    A = np.dot(Rotmat,yb)  # constraint vector in spatial frame
    
    # Constraint quadratic velocity vector, b
    Rotmat_dot = np.matmul(skew(w),Rotmat)
    A_dot = np.dot(Rotmat_dot,yb)
    b = np.dot(-A_dot,dx)

    e = b - np.dot(A,ddx) # Constraint "error vector"
    K = np.linalg.pinv(A[:,np.newaxis]).reshape(-1) # Contraint "gain"

    return K*e # Constraint force

# Objective function
def objective_function(x, xyz, dxyz, ddxyz, q, w, dw, dt):

    # Evaluate the unconstrained system state
    dxyz_unc = dxyz + ddxyz * dt
    xyz_unc = xyz + dxyz_unc * dt
    w_unc = w + dw * dt
    q_unc = quaternion_product(
                    exponential_map(w_unc * dt), q)
    
    # Evaluate the current solution for orientation
    w_opt = w + x * dt
    q_opt = quaternion_product(
                    exponential_map(w_opt * dt), q)

    # Evaluate the constraint forces
    f_con = nonholonomic_constraint_force(q_unc,dxyz_unc,w_opt,ddxyz)

    # Evaluate SO(3) distance between original and current orientation 
    Rotmat_unc = R.from_quat(q_unc).as_matrix()
    Rotmat_opt = R.from_quat(q_opt).as_matrix()
    so3_distance = np.linalg.norm(Rotmat_unc - Rotmat_opt,'fro')

    return np.sum(f_con**2) + so3_distance**2

# Plot rollout
def plot_rollout(xyz_traj, q_traj, constraint, label=None):

    if isinstance(xyz_traj, list) is False:
        xyz_traj = [xyz_traj]
        q_traj = [q_traj]
        constraint = [constraint]
        label = [label]

    N = len(xyz_traj)
    mk = ['-','--',':','-.','-*']

    fig = plt.figure(figsize=(16,3))

    plt.subplot(151)
    for i in range(N):
        plt.plot(xyz_traj[i],ls=mk[i],alpha=0.4,lw=6.0)
    plt.xlabel('Sample',fontsize=16)
    plt.ylabel('Unit distance',fontsize=16)
    plt.title('Position',fontsize=20)

    plt.subplot(152) 
    for i in range(N):
        plt.plot(q_traj[i],ls=mk[i],alpha=0.4,lw=6.0) 
    plt.xlabel('Sample',fontsize=16)
    plt.ylabel('Radians',fontsize=16)
    plt.title('Orientation',fontsize=20)

    plt.subplot(153)
    for i in range(N):
        plt.plot(xyz_traj[i][:,0],xyz_traj[i][:,1],ls=mk[i],alpha=0.4,lw=6.0)
    plt.xlabel('X',fontsize=16)
    plt.ylabel('Y',fontsize=16)
    # plt.xlim([-2,4])
    # plt.ylim([-7,0.5])
    plt.title('XY trajectory',fontsize=20)

    plt.subplot(154)
    for i in range(N):
        plt.plot(xyz_traj[i][:,1],xyz_traj[i][:,2],ls=mk[i],alpha=0.4,lw=6.0)
    plt.xlabel('Y',fontsize=16)
    plt.ylabel('Z',fontsize=16)
    plt.title('YZ trajectory',fontsize=20)

    plt.subplot(155)
    for i in range(N):
        plt.plot(constraint[i],ls=mk[i],alpha=0.4,label='DMP',lw=6.0); 
    plt.title('Constraint',fontsize=20)
    plt.xlabel('Sample',fontsize=16)
    
    plt.legend(label,fontsize=12)
    plt.tight_layout()
    plt.show()


def H_inv(H):

    Rotmat = H[:3,:3]
    p = H[:3,3]

    H_inv = np.zeros([4,4])
    H_inv[:3,:3] = Rotmat.T
    H_inv[:3,3] = - np.dot(Rotmat.T,p)
    H_inv[3,3] = 1.0

    return H_inv

def scalpel_trajectory(pose_traj, skip_frames=10, elevation=90, azimuth=90, title=None):

    indx = np.arange(0,pose_traj.shape[0],skip_frames)

    fig = plt.figure(figsize=(10,10))
    for i in indx:

        Hsh = np.eye(4) # scalpel handle frame wrt scalpel tip frame
        Hsh[:3,3] = [1.18,0.0,-0.0817]
        Hws = np.eye(4) # scalpel tip frame wrt world frame
        Hws[:3,3] = pose_traj[i,:3]
        Hws[:3,:3] = R.from_quat(pose_traj[i,3:]).as_matrix()
        Hwm = np.eye(4); Hwm[:3,:3] = R.from_euler('x', 180, degrees=True).as_matrix()
        Hms = np.matmul(H_inv(Hwm),Hws)
        Hmh = np.matmul(Hms,Hsh)
        Hwh = np.matmul(Hwm,Hmh)
        
        ax = plot_mesh(filename=("scalpel.stl"),s=0.03 * np.ones(3), alpha=0.2, A2B=Hwh)
        plot_transform(ax=ax, A2B=Hws, s=0.4, lw=3)
        plt.plot(pose_traj[:,0],pose_traj[:,1],pose_traj[:,2],lw=5,ls=':',markersize=2,alpha=0.3)
        # plt.plot(pose_traj[:i,0],pose_traj[:i,1],pose_traj[:i,2],lw=5,markersize=2)

        ax.set_xlim([-2,2])
        ax.set_ylim([-2,2])
        ax.set_zlim([-2,2])
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        ax.set_zticks([])
        if title is not None:
            ax.set_title(title,fontsize=20)

        display.clear_output(wait=True)
        display.display(plt.gcf())

    plt.show()