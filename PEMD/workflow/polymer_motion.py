#!/usr/bin/env python

from PEMD.workflow.polymer_ion_dynamics import PolymerIonDynamics

work_dir="./"
tpr_file="nvt_prod.tpr"
xtc_unwrap_file="nvt_prod_unwrap.xtc"
xtc_wrap_file="nvt_prod.xtc"
cation_atoms="resname LIP and name Li"
anion_atoms="resname NSC and name OBT"
polymer_atoms="resname MOL and name O"
run_start=0
run_end=1001
dt=0.001
dt_collection=5000
time_window_M1=2001
time_window_M2=501


# 初始化实例
polymer_dynamics = PolymerIonDynamics(
    work_dir,
    tpr_file,
    xtc_unwrap_file,
    xtc_wrap_file,
    cation_atoms,  # 根据你的系统修改选择器
    anion_atoms,
    polymer_atoms,
    run_start,
    run_end,
    dt,
    dt_collection,
)

# 计算 Tau3 (跳跃时间)
tau3 = polymer_dynamics.calculate_tau3()
print(f"Tau3: {tau3} ns")

# 计算 Tau1 (通过外推MSD)
times_M1 = polymer_dynamics.times_range(time_window_M1)
msd_M1 = polymer_dynamics.calculate_msd_parallel(polymer_dynamics.calculate_delta_n_square, time_window_M1)
tau1 = polymer_dynamics.extrapolate_msd(tau3, times_M1, msd_M1)

# 计算 TauR (通过拟合Rouse模型)
total_time = run_end - run_start
times_MR = polymer_dynamics.times_range(total_time)
msd_oe = polymer_dynamics.calculate_oe_msd(times_MR)
re_all = polymer_dynamics.ms_endtoend_distance()
tauR = polymer_dynamics.fit_rouse_model(re_all, times_MR, msd_oe)

# 计算 Tau2 (通过拟合Rouse模型)
times_M2 = polymer_dynamics.times_range(time_window_M2)
msd_M2 = polymer_dynamics.calculate_msd_parallel(polymer_dynamics.calculate_msd_M2, time_window_M2)
tau2 = polymer_dynamics.fit_rouse_model(re_all, times_M2, msd_M2)

# 创建输出文件
output_file_path = "polymer_motion.txt"
with open(output_file_path, 'w') as file:
    # 计算 Tau1 (通过外推MSD) 并写入文件
    file.write(f"Tau1: {tau1} ns\n")
    print(f"Tau1: {tau1} ns")

    # 计算 TauR (通过拟合Rouse模型) 并写入文件
    file.write(f"TauR: {tauR} ns\n")
    print(f"TauR: {tauR} ns")

    # 计算 Tau2 (通过拟合Rouse模型) 并写入文件
    file.write(f"Tau2: {tau2} ns\n")
    print(f"Tau2: {tau2} ns")

    # 计算 Tau3 (跳跃时间) 并写入文件
    file.write(f"Tau3: {tau3} ns\n")
    print(f"Tau3: {tau3} ns")



