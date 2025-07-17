
force = 0.12
total_force = force * 4 * 4
total_lift_up_mass = total_force / 9.81 # kg
drone_mass_sum = 0.035 * 4
bar_mass_sum = 0.001 * 4
payload_max_mass = total_lift_up_mass - drone_mass_sum - bar_mass_sum
print(payload_max_mass)
