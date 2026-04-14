
"""
@author: Juanita Sánchez
"""
import numpy as np                   
import matplotlib.pyplot as plt       
import os

#1 LOAD DATA

base_dir = os.path.dirname(os.path.abspath(__file__))
data = np.loadtxt(os.path.join(base_dir, 'example2.txt'))

# Extract coordinates X and Y 
X_original = data[:, 1]    
Y_original = data[:, 2]

# Count number of observations
n_obs = len(X_original)


# 2 ADD NOISE

noise = 5.0
np.random.seed(12345)
noise_X = noise * np.random.randn(n_obs)
noise_Y = noise * np.random.randn(n_obs)
X_obs = X_original + noise_X
Y_obs = Y_original + noise_Y


# 3 FILTER PARAMETERS
sigma_position = 1.0      
sigma_velocity = 1.0      
sigma_observation = 25.0  

sigma_pos_initial = 10.0  
sigma_vel_initial = 1.0   

# Time interval between observations
dt = 1.0                 


# 4 FILTER MATRICES

T = np.array([
    [1, 0, dt, 0],    
    [0, 1, 0, dt],    
    [0, 0, 1,  0],   
    [0, 0, 0,  1]     
])



A = np.array([
    [1, 0, 0, 0],     
    [0, 1, 0, 0]      
])

 

C_model = np.diag([
    sigma_position**2,    
    sigma_position**2,     
    sigma_velocity**2,    
    sigma_velocity**2      
])


C_obs = np.diag([
    sigma_observation**2,  
    sigma_observation**2   
])


C_error = np.diag([
    sigma_pos_initial**2,  
    sigma_pos_initial**2,  
    sigma_vel_initial**2,  
    sigma_vel_initial**2  
])


I = np.eye(4)


# 5 INITIALIZE STATE

x = np.array([
    X_obs[0] + 1.0,  
    Y_obs[0] + 1.0,   
    0.0,             
    0.0               
])


print(f"  X  = {x[0]:.2f} m")
print(f"  Y  = {x[1]:.2f} m")
print(f"  Vx = {x[2]:.2f} m/s")
print(f"  Vy = {x[3]:.2f} m/s")
print()

# Create arrays to store results
X_estimated = np.zeros(n_obs)
Y_estimated = np.zeros(n_obs)
Vx_estimated = np.zeros(n_obs)
Vy_estimated = np.zeros(n_obs)

# Store first state
X_estimated[0] = x[0]
Y_estimated[0] = x[1]
Vx_estimated[0] = x[2]
Vy_estimated[0] = x[3]


# 6 MAIN KALMAN FILTER LOOP

# Process each observation 
for i in range(1, n_obs):
    

    x_predicted = T @ x
    
    K = C_model + T @ C_error @ T.T
    
    G = K @ A.T @ np.linalg.inv(A @ K @ A.T + C_obs)

    y_obs = np.array([X_obs[i], Y_obs[i]])
    
    x = (I - G @ A) @ x_predicted + G @ y_obs
    
    C_error = (I - G @ A) @ K

    X_estimated[i] = x[0]
    Y_estimated[i] = x[1]
    Vx_estimated[i] = x[2]
    Vy_estimated[i] = x[3]


print("RESULTS")
print()

# Show first 5 observations
print("First 5 observations:")
print("  i      X_obs      Y_obs      X_est      Y_est      Vx         Vy")
for i in range(5):
    print(f"{i+1:3d}   {X_obs[i]:8.2f}  {Y_obs[i]:8.2f}  "
          f"{X_estimated[i]:8.2f}  {Y_estimated[i]:8.2f}  "
          f"{Vx_estimated[i]:8.4f}  {Vy_estimated[i]:8.4f}")
print()

# Show last 5 observations
print("Last 5 observations:")
print("  i      X_obs      Y_obs      X_est      Y_est      Vx         Vy")
for i in range(n_obs-5, n_obs):
    print(f"{i+1:3d}   {X_obs[i]:8.2f}  {Y_obs[i]:8.2f}  "
          f"{X_estimated[i]:8.2f}  {Y_estimated[i]:8.2f}  "
          f"{Vx_estimated[i]:8.4f}  {Vy_estimated[i]:8.4f}")
print()

#Final uncertainties
sigma_final_x = np.sqrt(C_error[0, 0])
sigma_final_y = np.sqrt(C_error[1, 1])
sigma_final_vx = np.sqrt(C_error[2, 2])
sigma_final_vy = np.sqrt(C_error[3, 3])

print("Final uncertainties:")
print(f"  σ_X  = {sigma_final_x:.2f} m")
print(f"  σ_Y  = {sigma_final_y:.2f} m")
print(f"  σ_Vx = {sigma_final_vx:.2f} m/s")
print(f"  σ_Vy = {sigma_final_vy:.2f} m/s")
print()

# Calculate average velocity
average_velocity_x = np.mean(Vx_estimated[10:])
average_velocity_y = np.mean(Vy_estimated[10:])
total_velocity = np.sqrt(average_velocity_x**2 + average_velocity_y**2)

print("Average estimated velocity:")
print(f"  Vx = {average_velocity_x:.4f} m/s")
print(f"  Vy = {average_velocity_y:.4f} m/s")
print(f"  |V| = {total_velocity:.4f} m/s")
print()


# PLOT

plt.figure(figsize=(10, 8))

plt.scatter(X_obs, Y_obs, 
           c='blue', s=15, alpha=0.5, 
           label=f'Noisy observations (n={n_obs})')

plt.scatter(X_estimated, Y_estimated, 
           c='red', s=20, alpha=0.8,
           label='Kalman filter estimates')

plt.xlabel('X (m)', fontsize=12)
plt.ylabel('Y (m)', fontsize=12)
plt.title(f'Kalman Filter: {n_obs} Noisy Observations + Filtered Trajectory', 
          fontsize=14, fontweight='bold')
plt.legend(loc='upper left', fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(100, 900)
plt.ylim(450, 850)

plt.savefig('plots/Kalman_Filter_Results.png', dpi=150, bbox_inches='tight')
plt.show()




































