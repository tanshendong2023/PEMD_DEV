"""
Polymer model MD tools.

Developed by: Tan Shendong
Date: 2024.03.26
"""


import os
import time
from foyer import Forcefield
from simple_slurm import Slurm
from PEMD.simulation import sim_lib
from PEMD.model import model_lib, build
from PEMD.simulation.lammps import PEMDLAMMPS
from PEMD.core.forcefields import Forcefield
from PEMD.simulation.gromacs import PEMDGROMACS


def relax_poly_chain(
        work_dir,
        pdb_file,
        core,
        atom_typing
):

    file_prefix, file_extension = os.path.splitext(pdb_file)
    relax_dir = os.path.join(work_dir, 'relax_polymer_lmp')
    os.makedirs(relax_dir, exist_ok=True)

    Forcefield.get_gaff2(
        relax_dir,
        pdb_file,
        atom_typing
    )

    lmp = PEMDLAMMPS(
        relax_dir,
        core,
    )

    lmp.generate_input_file(
        file_prefix
    )

    lmp.run_local()

    return sim_lib.lmptoxyz(
        relax_dir,
        pdb_file,
    )

def anneal_amorph_poly(
        work_dir,
        molecules,
        temperature,
        T_high_increase,
        anneal_rate,
        anneal_npoints,
        packmol_pdb,
        density,
        add_length,
):
    MD_dir = os.path.join(work_dir, 'MD_dir')
    os.makedirs(MD_dir, exist_ok=True)

    gmx = PEMDGROMACS(
        MD_dir,
        molecules,
        temperature,
    )

    gmx.gen_top_file(
        top_filename = 'topol.top'
    )

    gmx.gen_em_mdp_file(
        filename = 'em.mdp'
    )

    gmx.gen_nvt_mdp_file(
        filename = 'nvt.mdp'
    )

    gmx.gen_npt_anneal_mdp_file(
        T_high_increase,
        anneal_rate,
        anneal_npoints,
        filename = 'npt_anneal.mdp'
    )

    gmx.gen_npt_mdp_file(
        filename = 'npt_eq.mdp'
    )

    gmx.commands_pdbtogro(
        packmol_pdb,
        density,
        add_length
    ).run_local()

    gmx.commands_em(
        input_gro = 'conf.gro'
    ).run_local()

    gmx.commands_nvt(
        input_gro = 'em.gro'
    ).run_local()

    gmx.commands_npt_anneal(
        input_gro = 'nvt.gro'
    ).run_local()

    gmx.commands_npt(
        input_gro = 'npt_anneal.gro'
    ).run_local()

    gmx.commands_extract_volume(
        edr_file = 'npt_eq.edr',
        output_file = 'volume.xvg'
    ).run_local()

    volumes_path = os.path.join(MD_dir, 'volume.xvg')
    volumes = model_lib.read_volume_data(volumes_path)

    (
        average_volume,
        frame_time
     ) = model_lib.analyze_volume(
        volumes,
        start=4000,
        dt_collection=5
    )

    gmx.commands_extract_structure(
        tpr_file = 'npt_eq.tpr',
        xtc_file = 'npt_eq.xtc',
        save_gro_file = f'pre_eq.gro',
        frame_time = frame_time
    ).run_locol()

def run_gmx_prod(
        work_dir,
        molecules,
        temperature,
        nstep_ns,
):

    MD_dir = os.path.join(work_dir, 'MD_dir')
    os.makedirs(MD_dir, exist_ok=True)

    gmx = PEMDGROMACS(
        MD_dir,
        molecules,
        temperature,
    )

    # generation nvt production mdp file, 200ns
    nstep = int(nstep_ns*1000000)   # ns to fs
    gmx.gen_nvt_mdp_file(
        nsteps_nvt = nstep,
        filename = 'nvt_prod.mdp',
    )

    gmx.commands_nvt(
        input_gro = 'pre_eq.gro',
    ).run_local()

    gmx.commands_wraptounwrap(
        output_str = 'nvt_prod'
    ).run_local()


def pre_run_gmx(model_info, density, add_length, out_dir, packout_name, core, partition, T_target,
                T_high_increase=500, anneal_rate=0.05, top_filename='topol.top', module_soft='GROMACS',
                output_str='pre_eq'):

    current_path = os.getcwd()

    # unit_name = model_info['polymer']['compound']
    # length = model_info['polymer']['length'][1]

    MD_dir = os.path.join(current_path, out_dir)
    os.chdir(MD_dir)

    numbers = model_lib.print_compounds(model_info,'numbers')
    compounds = model_lib.print_compounds(model_info,'compound')
    resnames = model_lib.print_compounds(model_info,'resname')

    pdb_files = []
    for com in compounds:
        filepath = os.path.join(MD_dir, f"{com}.pdb")
        pdb_files.append(filepath)

    box_length = (build.calculate_box_size(numbers, pdb_files, density) + add_length) / 10  # A to nm

    # generate top file
    gen_top_file(compounds, resnames, numbers, top_filename)

    # generation minimization mdp file
    gen_min_mdp_file(file_name='em.mdp')

    # generation nvt mdp file
    gen_nvt_mdp_file(nsteps_nvt=200000,
                     nvt_temperature=f'{T_target}',
                     file_name='nvt.mdp', )

    # Setup annealing
    T_high = T_target + T_high_increase
    annealing_time_steps = int((T_high - T_target) / anneal_rate)  # Calculating number of steps for annealing process
    nsteps_annealing = (1000 * 2 + 2 * annealing_time_steps) * 1000
    annealing_time = f'0 1000 {1000 + 1 * annealing_time_steps} {1000 + 2 * annealing_time_steps} {1000 * 2 + 2 * annealing_time_steps}'
    annealing_temp = f'{T_target} {T_target} {T_high} {T_target} {T_target}'

    gen_npt_anneal_mdp_file(nsteps_annealing=nsteps_annealing,
                            npt_temperature=f'{T_target}',
                            annealing_npoints=5,
                            annealing_time=annealing_time,
                            annealing_temp=annealing_temp,
                            file_name='npt_anneal.mdp', )

    # generation nvt mdp file
    gen_npt_mdp_file(nsteps_npt=5000000,
                     npt_temperature=f'{T_target}',
                     file_name = 'npt_eq.mdp',)

    # generation slurm file
    if partition=='gpu':
        slurm = Slurm(J='gmx-gpu',
                      cpus_per_gpu=f'{core}',
                      gpus=1,
                      p=f'{partition}',
                      output=f'{MD_dir}/slurm.%A.out'
                      )

        slurm.add_cmd(f'module load {module_soft}')
        slurm.add_cmd(f'gmx editconf -f {packout_name} -o conf.gro -box {box_length} {box_length} {box_length}')
        slurm.add_cmd(f'gmx grompp -f em.mdp -c conf.gro -p {top_filename} -o em.tpr -maxwarn 1')
        slurm.add_cmd(f'gmx mdrun -ntmpi 1 -ntomp {core} -v -deffnm em')
        slurm.add_cmd(f'gmx grompp -f nvt.mdp -c em.gro -p {top_filename} -o nvt.tpr -maxwarn 1')
        slurm.add_cmd(f'gmx mdrun -ntmpi 1 -ntomp {core} -v -deffnm nvt')
        slurm.add_cmd(f'gmx grompp -f npt_anneal.mdp -c nvt.gro -p {top_filename} -o npt_anneal.tpr -maxwarn 1')
        slurm.add_cmd(f'gmx mdrun -ntmpi 1 -ntomp {core} -v -deffnm npt_anneal')
        slurm.add_cmd(f'gmx grompp -f npt_eq.mdp -c npt_anneal.gro -p {top_filename} -o npt_eq.tpr -maxwarn 1')
        slurm.add_cmd(f'gmx mdrun -ntmpi 1 -ntomp {core} -v -deffnm npt_eq')

    else:
        slurm = Slurm(J='gmx',
                      N=1,
                      n=f'{core}',
                      p=f'{partition}',
                      output=f'{MD_dir}/slurm.%A.out'
                      )

        slurm.add_cmd(f'module load {module_soft}')
        slurm.add_cmd(f'gmx_mpi editconf -f {packout_name} -o conf.gro -box {box_length} {box_length} {box_length}')
        slurm.add_cmd(f'gmx_mpi grompp -f em.mdp -c conf.gro -p {top_filename} -o em.tpr')
        slurm.add_cmd('gmx_mpi mdrun -v -deffnm em')
        slurm.add_cmd(f'gmx_mpi grompp -f nvt.mdp -c em.gro -p {top_filename} -o nvt.tpr')
        slurm.add_cmd('gmx_mpi mdrun -v -deffnm nvt')
        slurm.add_cmd(f'gmx_mpi grompp -f npt_anneal.mdp -c nvt.gro -p {top_filename} -o npt_anneal.tpr')
        slurm.add_cmd('gmx_mpi mdrun -v -deffnm npt_anneal')
        slurm.add_cmd(f'gmx_mpi grompp -f npt_eq.mdp -c npt_anneal.gro -p {top_filename} -o npt_eq.tpr')
        slurm.add_cmd(f'mpirun gmx_mpi mdrun -v -deffnm npt_eq')

    job_id = slurm.sbatch()

    while True:
        status = sim_lib.get_slurm_job_status(job_id)
        if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
            print("MD simulation finish, executing the XX task...")
            break
        else:
            print("MD simulation not finish, waiting...")
            time.sleep(10)

    model_lib.extract_volume(partition, module_soft, edr_file='npt_eq.edr', output_file='volume.xvg', option_id='21')

    volumes_path = os.path.join(MD_dir, 'volume.xvg')
    volumes = model_lib.read_volume_data(volumes_path)

    average_volume, frame_time = model_lib.analyze_volume(volumes, start=4000, dt_collection=5)

    model_lib.extract_structure(partition, module_soft, tpr_file='npt_eq.tpr', xtc_file='npt_eq.xtc',
                               save_gro_file=f'{output_str}.gro', frame_time=frame_time)

    os.chdir(current_path)


def run_gmx_prod(out_dir, core, partition, T_target, input_str, top_filename, module_soft='GROMACS', nstep_ns=200,
                 output_str='nvt_prod',):

    current_path = os.getcwd()
    MD_dir = os.path.join(current_path, out_dir)
    os.chdir(MD_dir)

    # generation nvt production mdp file, 200ns
    nstep=nstep_ns*1000000
    gen_nvt_mdp_file(nsteps_nvt=f'{nstep}',
                     nvt_temperature=f'{T_target}',
                     file_name='nvt_prod.mdp',)

    # generation slurm file
    if partition == 'gpu':
        slurm = Slurm(J='gmx-gpu',
                      cpus_per_gpu=f'{core}',
                      gpus=1,
                      p=f'{partition}',
                      output=f'{MD_dir}/slurm.%A.out'
                      )

        slurm.add_cmd(f'module load {module_soft}')
        slurm.add_cmd(f'gmx grompp -f nvt_prod.mdp -c {input_str}.gro -p {top_filename} -o {output_str}.tpr -maxwarn 1')
        slurm.add_cmd(f'gmx mdrun -ntmpi 1 -ntomp {core} -v -deffnm {output_str}')
        slurm.add_cmd(f'echo 0 | gmx trjconv -s {output_str}.tpr -f {output_str}.xtc -o {output_str}_unwrap.xtc -pbc '
                      f'nojump -ur compact')

    else:
        slurm = Slurm(J='gmx',
                      N=1,
                      n=f'{core}',
                      p=f'{partition}',
                      output=f'{MD_dir}/slurm.%A.out'
                      )

        slurm.add_cmd(f'module load {module_soft}')
        slurm.add_cmd(f'gmx_mpi grompp -f nvt_prod.mdp -c {input_str}.gro -p {top_filename} -o {output_str}.tpr')
        slurm.add_cmd(f'mpirun gmx_mpi mdrun -v -deffnm {output_str}')
        slurm.add_cmd(f'echo 0 | gmx_mpi trjconv -s {output_str}.tpr -f {output_str}.xtc -o {output_str}_unwrap.xtc '
                      f'-pbc nojump -ur compact')

    job_id = slurm.sbatch()

    os.chdir(current_path)

    while True:
        status = sim_lib.get_slurm_job_status(job_id)
        if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
            print("MD simulation finish, executing the XX task...")
            break
        else:
            print("MD simulation not finish, waiting...")
            time.sleep(10)


def run_gmx_tg(out_dir, input_str, out_str, partition,top_filename='topol.top', module_soft='GROMACS',
               anneal_rate=0.01, core=64, Tinit=1000, Tfinal=100,):

    current_path = os.getcwd()
    MD_dir = os.path.join(current_path, out_dir)
    os.chdir(MD_dir)

    # heating up for anneal, heating rate 0.05K/ps
    dT = Tinit - Tfinal
    total_time_ps = int(dT/0.05) # unit: ps
    gen_npt_anneal_mdp_file(nsteps_annealing=int(dT/0.05*1000),
                            npt_temperature=f'{Tfinal}',
                            annealing_npoints=2,
                            annealing_time=f'0 {total_time_ps}',
                            annealing_temp=f'{Tfinal} {Tinit}',
                            file_name='npt_heating.mdp', )

    # 每个温度点之间的温度差
    delta_temp = 20  # 每20K进行一次退火

    # 退火所需的时间（ps），每升高1K需要的时间是1/0.01=100ps，20K需要2000ps, unit:k/ps
    time_per_delta_temp = delta_temp / anneal_rate

    # 温度列表和时间点列表
    temperatures = [Tinit]
    time_points = [0]  # 从0开始

    # 当前温度和时间
    current_temp = Tinit
    current_time = 0

    while current_temp > Tfinal:
        # 添加当前温度，进行2ns的恒定温度模拟
        temperatures.append(current_temp)
        # print(temperatures)
        current_time += 2000  # 2ns的模拟
        # print(current_time)
        time_points.append(int(current_time))
        # print(time_points)

        # 退火到下一个温度点
        if current_temp - delta_temp >= Tfinal:  # 确保不会低于最终温度
            current_temp -= delta_temp
            temperatures.append(current_temp)
            current_time += time_per_delta_temp  # 退火过程
            time_points.append(int(current_time))
        else:
            break

    temperatures.append(Tfinal)
    current_time += 2000  # 最终温度再进行2ns的恒定温度模拟
    time_points.append(int(current_time))

    # 生成npt退火mdp文件
    gen_npt_anneal_mdp_file(nsteps_annealing=int(current_time * 1000),  # 总步数
                            npt_temperature=f'{Tinit}',  # 最终温度
                            annealing_npoints=len(temperatures),  # 温度点的数量
                            annealing_time=' '.join(str(t) for t in time_points),  # 每个温度点的开始时间，单位：步
                            annealing_temp=' '.join(str(temp) for temp in temperatures),  # 对应的温度点
                            file_name='npt_anneal_tg.mdp')

    # generation slurm file
    slurm = Slurm(J='gromacs',
                  N=1,
                  n=f'{core}',
                  p=f'{partition}',
                  output=f'{MD_dir}/slurm.%A.out'
                  )

    slurm.add_cmd(f'module load {module_soft}')
    slurm.add_cmd(f'gmx_mpi grompp -f npt_heating.mdp -c {input_str}.gro -p {top_filename} -o npt_heating.tpr')
    slurm.add_cmd('gmx_mpi mdrun -v -deffnm npt_heating')
    slurm.add_cmd(f'gmx_mpi grompp -f npt_anneal_tg.mdp -c npt_heating.gro -p {top_filename} -o {out_str}.tpr')
    slurm.add_cmd(f'gmx_mpi mdrun -v -deffnm {out_str}')

    job_id = slurm.sbatch()

    os.chdir(current_path)

    while True:
        status = sim_lib.get_slurm_job_status(job_id)
        if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
            print("MD simulation finish, executing the XX task...")
            break
        else:
            print("MD simulation not finish, waiting...")
            time.sleep(10)






























