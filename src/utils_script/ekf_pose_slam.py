#!/usr/bin/python3
"""Pose-based EKF SLAM filter.

The state vector stacks every stored scan viewpoint followed by the current
robot pose, ``[x_0, y_0, theta_0, ..., x_k, y_k, theta_k]^T``, and the
covariance holds all cross-correlations between them. Scan-to-scan ICP results
are used as relative-pose observations between the current pose and earlier
viewpoints.
"""

import numpy as np
from scipy.linalg import block_diag
from  utils_script.pose import Pose3D
from math import cos, sin
import rospy
import scipy
import threading
from scipy.spatial import KDTree
import math 

class PoseSLAMEKF:
    """Extended Kalman Filter over a growing vector of robot poses.

    Provides the differential-drive prediction, state augmentation when a new
    scan is stored, overlap search, ICNN gating of ICP measurements, the EKF
    update and a yaw (compass/IMU) update.
    """
    def __init__(self, x0, P0, Q, compass_Rk , compass_Vk , wheel_base , wheel_radius , overlap_distancce = 2):    
        """Store the initial estimate and the noise models.

        :param x0: initial state vector (3x1 for a single pose).
        :param P0: initial state covariance.
        :param Q: 2x2 process noise covariance used with :meth:`F2k`.
        :param compass_Rk: 1x1 covariance of the heading measurement.
        :param compass_Vk: 1x1 heading noise Jacobian.
        :param wheel_base: distance between the wheels [m].
        :param wheel_radius: wheel radius [m].
        :param overlap_distancce: overlap distance threshold [m]. Stored but
            not read by the methods of this class.
        """
        
        self.xk = x0
        self.Pk= P0
        self.Qk = Q
        self.compass_Rk  = compass_Rk
        self.compass_Vk = compass_Vk
        self.wheel_base = wheel_base 
        self.wheel_radius = wheel_radius
        self.xb_dim = 3
        self.alpha = 0.95
        self.overlap_distancce = overlap_distancce
        self.mutex = threading.Lock()

    def wrap_angle(self, angle):
        """this function wraps the angle between -pi and pi

        :param angle: the angle to be wrapped
        :type angle: float

        :return: the wrapped angle
        :rtype: float
        """
        return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )
    
    def Add_New_Pose(self, xk, Pk):
        """Clone the current pose into the state as a new scan viewpoint.

        The clone is fully correlated with the current pose.

        :param xk: state vector.
        :param Pk: state covariance.
        :return: augmented ``(xk, Pk)``, three entries longer.
        """
        xk_new = np.zeros((len(xk)+3,1))
        xk_new[:-3,:] = xk
        xk_new[-3:,:] = xk[-3:,:]
        Pk_new = np.zeros((len(xk)+3,len(xk)+3))

        last_row = Pk[-3:,:] # 
        last_col = Pk[:,-3:]
        Pk_new[:-3,:-3] = Pk
        Pk_new[-3:,:-3] = last_row # make them full correlation at firist 
        Pk_new[:-3,-3:] = last_col  # make them full correlation at firist
        Pk_new[-3:,-3:] = Pk[-3:,-3:]

        return xk_new , Pk_new
    
    def Prediction(self, xk_1 , Pk_1 , uk , dt):             
        """EKF prediction of the current pose; stored viewpoints are unchanged.

        :param xk_1: previous state vector.
        :param Pk_1: previous state covariance.
        :param uk: 3x1 odometry increment ``[v*dt, 0, w*dt]`` in the robot frame.
        :param dt: time step [s].
        :return: predicted ``(xk, Pk)``.
        """
        t0 = rospy.Time.now().to_sec()
        self.xk = xk_1
        self.dt = dt
        self.uk = uk

        xk_robot = self.get_robot_pose(xk_1)
        theta = xk_robot[-1]

        Jfx = self.F1k(xk_robot)
        Jfw= self.F2k(xk_robot)
        
        # Predict the mean of the robot pose
        xk_robot = Pose3D.oplus(xk_robot, uk)
        xk_robot[-1] = self.wrap_angle(xk_robot[-1])

        # Add the predicted robot pose to the state vector
        xk = np.block([[xk_1[:-3,:]], [xk_robot]])

        A = Pk_1[:-3,:-3]  # Extract the covariance of the scan pose
        B = Pk_1[-3:,-3:]  # Extract the covariance of the robot pose
        C = Pk_1[-3:,:-3]  # Extract the covariance of the side
        
        P =  Jfx@B@Jfx.T + Jfw@self.Qk@Jfw.T
        Pk = np.block([[A,C.T@Jfx.T],[Jfx@C,P]])
     
        return xk ,Pk
    
    def get_robot_pose(self , xk):
        """Return the current robot pose, i.e. the last three state entries (3x1)."""
        return xk[-3:,-3:]
    
    def F1k(self,xk_robot):
        """Jacobian of the motion model w.r.t. the robot pose (3x3).

        Uses the control input stored by the last :meth:`Prediction` call.
        """
        Jfx = Pose3D.J_1oplus(xk_robot,self.uk)
     
        # F1x = np.block([[np.eye(len(self.xk)-3), np.zeros((len(self.xk)-3, 3))],
        #                 [np.zeros((3, len(self.xk)-3)), Jfx]])
        return Jfx
    
    def F2k(self,xk_robot):
        """Jacobian of the motion model w.r.t. the 2D process noise (3x2).

        Uses the time step stored by the last :meth:`Prediction` call.
        TODO: document the units of the noise vector that ``Qk`` describes.
        """
        theta = xk_robot[-1,-1]
       
        # Jfw = np.array([[self.dt*self.wheel_radius*cos(theta)/2,    self.dt*self.wheel_radius*cos(theta)/2],
        #                 [self.dt*self.wheel_radius*sin(theta)/2,    self.dt*self.wheel_radius*sin(theta)/2],
        #                 [self.dt*self.wheel_radius/self.wheel_base, -self.dt*self.wheel_radius/self.wheel_base]]).reshape(3, 2)
     
        Jfw = np.array([[self.dt*cos(theta)/2,    self.dt*cos(theta)/2],
                        [self.dt*sin(theta)/2,    self.dt*sin(theta)/2],
                        [self.dt/self.wheel_base, -self.dt/self.wheel_base]])
        
        return Jfw
        
        
        # return np.eye(3)
        
        
    def OverlappingScan(self,xk ,max_num_scans, overlap_distance ):
        """Find stored viewpoints close enough to the newest one to be matched.

        Runs a KD-tree nearest-neighbour query (x, y only) from the newest stored
        viewpoint (the second-to-last pose in the state) to all earlier ones.

        :param xk: state vector.
        :param max_num_scans: maximum number of neighbours to query.
        :param overlap_distance: keep neighbours closer than this [m].
        :returns: list of viewpoint indices (pose index, not state index).

        TODO: the caller in ``slam_icp_node.py`` passes the distance threshold
        and the scan count in the opposite order to this signature; check which
        values were intended.
        """
        H0= [] 
        xk= xk.reshape(-1,3)
        xk_pose = xk[: , 0:2]
        current_pose = xk_pose[-2:-1,:]
        matched_pose = xk_pose[:-2,:]
       

        tree = KDTree(matched_pose)
        # Query the tree for the neareast neighbors
        dist, indices = tree.query(current_pose, k = min( len(matched_pose), max_num_scans))
        # Check if the distances are less than the overlap distance
        dist = dist.flatten()
        indices = indices.flatten()
        for i in range(len(dist)):
            if dist[i] < overlap_distance:
                H0.append(indices[i])

        return H0
    def remove_pose(self, xk , Pk , len = 3):
        """Drop ``len`` viewpoints from the state to bound its size.

        Removes the poses at indices 1, 3, 5, ... (every second of the oldest
        poses) from the state vector and the matching rows/columns of the
        covariance.

        :param xk: state vector.
        :param Pk: state covariance.
        :param len: number of poses to remove.
        :returns: ``(xk, Pk)`` without the removed poses.
        """
        indices = []
        for l in range(len):
            index = 2*l+1 #get the last even poses index from state vector 
            indices.extend( range(index*3 , index*3+3))
          
        xk = np.delete(xk, indices, axis=0)
        Pk = np.delete(Pk, indices, axis=0)
        Pk = np.delete(Pk, indices, axis=1)
   
        return xk , Pk
        
       

    def h(self, xk ,Hp):
        """Stack the expected relative-pose observations for every index in ``Hp``.

        :param xk: state vector.
        :param Hp: list of matched viewpoint indices.
        :return: stacked expected observations (3*len(Hp) x 1).
        """
        hf = np.zeros((0,1))
        for i in range(len(Hp)):
            hf = np.block([[hf],[self.hfj(xk, Hp[i])]])
        return hf

    def hfj(self, xk, j ):
        """Expected pose of the robot relative to viewpoint ``j``: ``(-)x_j (+) x_k``.

        :param xk: state vector.
        :param j: index of the matched viewpoint.
        :return: 3x1 relative pose, used as ICP initial guess and as ``h(x)``.
        """
        # h(xk_bar,vk)=(-) Xk) [+] x_J+ vk
        # Get Pose vector from the filter state
        NxBk = self.get_robot_pose(xk) # current pose 
        index = int(j*3)
        NxBj = xk[index:index+3,:].reshape((3,1)) # matched pose
        
        # Matched Scan as referance frame 
        Jxn  = Pose3D.ominus(NxBj)
        hfj  = Pose3D.oplus(Jxn ,NxBk)

        return hfj
    
    def Jhf(self,xk,j):
        
        """Jacobian of :meth:`hfj` with respect to the full state vector.

        Non-zero only in the columns of viewpoint ``j`` and of the current pose.

        :param xk: state vector.
        :param j: index of the matched viewpoint.
        :return: 3 x len(xk) Jacobian.
        """

        NxBk = self.get_robot_pose(xk)
        index = int(j*3)
        J1_oplus, J1_ominus, J2_oplus = self.Jhfjx(xk, j)
        J = np.zeros((self.xb_dim,np.shape(xk)[0]))
        J[:,index:index+3] = J1_oplus@J1_ominus
        J[:,-3:] = J2_oplus 
        return J

    def Jhfjx(self, xk, j):
        
        """Partial Jacobians of ``(-)x_j (+) x_k``.

        :param xk: state vector.
        :param j: index of the matched viewpoint.
        :return: ``(J1_oplus, J1_ominus, J2_oplus)``: compounding Jacobian w.r.t.
            ``(-)x_j``, inversion Jacobian w.r.t. ``x_j``, and compounding
            Jacobian w.r.t. ``x_k``.
        """
        # Get Pose vector from the filter state
        NxBk  = self.get_robot_pose(xk) # current pose 
       
        index = int(j*3)
        NxBj  = xk[index:index+3,:].reshape((3,1)) # matched pose
        
        JxN   = Pose3D.ominus(NxBj)  
        # hfj  = Pose3D.oplus(JxN,NxBk)
       
        J1_oplus  = Pose3D.J_1oplus(JxN, NxBk)
        J1_ominus = Pose3D.J_ominus(NxBj)
        J2_oplus  = Pose3D.J_2oplus(JxN)

        
        return J1_oplus, J1_ominus, J2_oplus

    
    def jPk(self, xk , Pk, j):
        """Covariance of the expected relative pose :meth:`hfj`.

        Propagates the marginal covariances of viewpoint ``j`` and the current
        pose; their cross-covariance is not included.

        :param xk: state vector.
        :param Pk: state covariance.
        :param j: index of the matched viewpoint.
        :return: 3x3 covariance.
        """
        index = int(j*3)
        NPk = Pk[-3:,-3:]  # Extract the covariance of the robot pose
        NPj = Pk[index:index+3,index:index+3]  # Extract the covariance of the matched scan pose
        J1_oplus, J1_ominus, J2_oplus = self.Jhfjx(xk, j)
        jPk = J1_oplus@J1_ominus@NPj@J1_ominus.T@J1_oplus.T + J2_oplus@NPk@J2_oplus.T
        
        return jPk
    
    def ObservationMatrix(self,xk,Hp ,zk ,Rk):
        """Build the stacked observation Jacobians for the accepted ICP matches.

        :param xk: state vector.
        :param Hp: accepted viewpoint indices.
        :param zk: stacked ICP measurements.
        :param Rk: block-diagonal ICP measurement covariance.
        :return: ``(zk, Rk, Hk, Vk)`` where ``Hk`` is the state Jacobian and
            ``Vk`` the (identity) measurement-noise Jacobian.
        """
        Hk, Vk = np.zeros((0,np.shape(xk)[0])), np.zeros((0,0))
        xk_robot = self.get_robot_pose(xk)
        Vr =  np.diag(np.ones(self.xb_dim))
       
        for i,j in enumerate(Hp):

            # Add jacobian with respect to the state vector
            Hk = np.block([[Hk], [self.Jhf(xk, j)]])
            # Add jacbian with respect to the feature observation noise
            Vk = scipy.linalg.block_diag(Vk,Vr)

        
        return zk ,Rk , Hk, Vk 
  
    def SquaredMahalanobisDistance(self, hfj, Pfj, zfi, Rfi):
        """Squared Mahalanobis distance of the innovation ``zfi - hfj``.

        :param hfj: expected relative pose.
        :param Pfj: covariance of the expected relative pose.
        :param zfi: measured relative pose (ICP).
        :param Rfi: measurement covariance.
        :return: 1x1 array with the squared distance.
        """
        # Compute inovation
        v_ij = zfi - hfj
        # Compute inovetion unceranity
        S_ij = Rfi + Pfj
   
        # Compute squared mahalandobis distance
        D2_ij = v_ij.T @ np.linalg.inv(S_ij) @ v_ij
        return D2_ij

    def IndividualCompatibility(self, D2_ij, dof, alpha):
        """Chi-squared gate on a squared Mahalanobis distance.

        :param D2_ij: squared Mahalanobis distance.
        :param dof: degrees of freedom.
        :param alpha: confidence level of the gate.
        :return: True if ``D2_ij`` is below the ``alpha`` quantile of chi2(dof).
        """
        # print("D2_ij", D2_ij)
        isCompatible = D2_ij < scipy.stats.chi2.ppf(alpha, dof)

        return isCompatible

    def ICNN(self, hf, Phf, zf, Rf):
        """Individual-compatibility test for one ICP measurement.

        :param hf: expected relative pose (3x1).
        :param Phf: its covariance (3x3).
        :param zf: ICP relative pose (3x1).
        :param Rf: ICP covariance (3x3).
        :return: True if the measurement passes the gate at ``self.alpha``,
            otherwise None.
        """

        D2_ij = self.SquaredMahalanobisDistance(hf, Phf, zf, Rf)
        if self.IndividualCompatibility(D2_ij, self.xb_dim, self.alpha):
            return True
    
    def Update(self, xk, Pk, Hk, Vk, zk, Rk  , Hp):
        """EKF correction with the stacked ICP measurements.

        The covariance is updated as ``(I - K H) P (I - K H)^T``.

        :param xk: state vector.
        :param Pk: state covariance.
        :param Hk: observation Jacobian w.r.t. the state.
        :param Vk: observation Jacobian w.r.t. the measurement noise.
        :param zk: stacked ICP measurements.
        :param Rk: measurement covariance.
        :param Hp: matched viewpoint indices (to evaluate ``h(xk)``).
        :return: updated ``(xk, Pk)``.
        """

        Kk = Pk @ Hk.T @ np.linalg.inv(Hk @ Pk @ Hk.T + Vk@Rk@Vk.T)
        print(f"Hk: {Hk.shape}, Vk: {Vk.shape}, Rk: {Rk.shape}, Pk: {Pk.shape}")
        
        xk = xk + Kk @ (zk - self.h(xk, Hp))
        Pk = (np.eye(len(xk)) - Kk @ Hk) @ Pk@(np.eye(len(xk)) - Kk @ Hk).T 
      
        return xk, Pk
    
    def heading_update(self , xk , Pk , yaw):
        """EKF correction of the current heading with an absolute yaw reading.

        :param xk: state vector.
        :param Pk: state covariance.
        :param yaw: 1x1 measured yaw [rad] (IMU orientation).
        :return: updated ``(xk, Pk)``.
        """
        # Create a row vector of zeros of size 1 x 3*num_poses
        # print("imu update")   
        Hk = np.zeros((1, len(xk)))
        # Replace the last element of the row vector with 1
        Hk[0, -1] = 1
        predicted_compass_meas = xk[-1]
        # Compute the kalman gain
        K = Pk @ Hk.T @ np.linalg.inv((Hk @ Pk @ Hk.T) + (self.compass_Vk @ self.compass_Rk @ self.compass_Vk.T))
        # Compute the innovation
        innovation = np.array(self.wrap_angle(yaw[0] - predicted_compass_meas)).reshape(1, 1)
        # Update the state vector
        xk = xk + K@innovation
        # Create the identity matrix        
        I = np.eye(len(xk))
        # Update the covariance matrix
        Pk = (I - K @ Hk) @ Pk @ (I - K @ Hk).T

        return xk, Pk
    
    