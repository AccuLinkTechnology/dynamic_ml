import matplotlib.pyplot as plt
import time

# Simulated system model
def system_model(power, current_temp):
    # Simple model where power affects temperature with some delay
    new_temp = current_temp + (power - (current_temp * 0.1)) * 0.1
    return new_temp

# PID Controller class
class PIDController:
    def __init__(self, kp, ki, kd, setpoint):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral = 0
        self.previous_error = 0
    
    def compute(self, current_value):
        error = self.setpoint - current_value
        self.integral += error
        derivative = error - self.previous_error
        
        # PID formula
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        self.previous_error = error
        return output

# Initialize system and PID controller
setpoint = 50  # Desired temperature
pid = PIDController(kp=2.0, ki=1.0, kd=0.5, setpoint=setpoint)

current_temp = 20  # Initial temperature
temperatures = [current_temp]

# Run the simulation
for _ in range(100):
    power = pid.compute(current_temp)
    current_temp = system_model(power, current_temp)
    temperatures.append(current_temp)
    time.sleep(0.1)  # Simulate time delay

# Plot the results
plt.plot(temperatures)
plt.axhline(y=setpoint, color='r', linestyle='--')
plt.title('Temperature Control using PID')
plt.xlabel('Time steps')
plt.ylabel('Temperature')
plt.show()