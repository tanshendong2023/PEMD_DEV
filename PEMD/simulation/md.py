"""
Polymer model MD tools.

Developed by: Tan Shendong
Date: 2024.03.26
"""


import os
import time
import shutil
import subprocess
import parmed as pmd
from PEMD.simulation import qm
from foyer import Forcefield
from simple_slurm import Slurm
from LigParGenPEMD import Converter
from PEMD.model import model_lib, build
from PEMD.simulation import sim_lib
import importlib.resources as pkg_resources


def gen_poly_gmx_oplsaa(
        poly_name,
        poly_resname,
        scaling_factor,
        charge,
        length,
):

    current_path = os.getcwd()
    MD_dir = os.path.join(current_path, 'MD_dir')
    os.makedirs(MD_dir, exist_ok=True)

    files_to_check = [
        ("pdb", f"{poly_name}.pdb"),
        ("itp", f"{poly_name}_bonded.itp"),
        ("itp", f"{poly_name}_nonbonded.itp")
    ]

    copied_any_file = False  # Flag to check if any file was successfully copied
    # 使用importlib.resources检查和复制文件
    for folder, filename in files_to_check:
        package_path = f'PEMD.forcefields.{folder}'
        if pkg_resources.is_resource(package_path, filename):
            with pkg_resources.path(package_path, filename) as src_path:
                dest_path = os.path.join(MD_dir, filename)
                shutil.copy(str(src_path), dest_path)
                print(f"Copied {filename} to MD_dir successfully.")
                copied_any_file = True
        else:
            print(f"File {filename} does not exist in {package_path}.")

    if copied_any_file:
        itp_filename = os.path.join(MD_dir, f"{poly_name}_bonded.itp")
        qm.scale_chg_itp(poly_name, itp_filename, scaling_factor, charge)
        print("Scale charge successfully.")
    else:
        desc_dir = os.path.join(current_path, f'{poly_name}_N{length}')
        relax_polymer_lmp_dir = os.path.join(desc_dir, 'relax_polymer_lmp')
        os.makedirs(relax_polymer_lmp_dir, exist_ok=True)

        file_base = f'{poly_name}_N{length}'
        xyz_filename = os.path.join(relax_polymer_lmp_dir, f"{file_base}_gmx.xyz")
        pdb_filename = os.path.join(relax_polymer_lmp_dir, f"{file_base}_gmx.pdb")
        mol2_filename = os.path.join(relax_polymer_lmp_dir, f"{file_base}_gmx.mol2")

        model_lib.convert_xyz_to_pdb(xyz_filename, pdb_filename, poly_name, poly_resname)
        model_lib.convert_xyz_to_mol2(xyz_filename, mol2_filename, poly_name, poly_resname)

        untyped_str = pmd.load_file(mol2_filename, structure=True)
        with pkg_resources.path("PEMD.forcefields", "oplsaa.xml") as oplsaa_path:
            oplsaa = Forcefield(forcefield_files=str(oplsaa_path))
        typed_str = oplsaa.apply(untyped_str)

        top_filename = os.path.join(MD_dir, f"{file_base}.top")
        gro_filename = os.path.join(MD_dir, f"{file_base}.gro")
        typed_str.save(top_filename)
        typed_str.save(gro_filename)

        shutil.copyfile(pdb_filename, os.path.join(MD_dir, f'{poly_name}.pdb'))

        nonbonditp_filename = os.path.join(MD_dir, f'{poly_name}_nonbonded.itp')
        bonditp_filename = os.path.join(MD_dir, f'{poly_name}_bonded.itp')

        model_lib.extract_from_top(top_filename, nonbonditp_filename, nonbonded=True, bonded=False)
        model_lib.extract_from_top(top_filename, bonditp_filename, nonbonded=False, bonded=True)

        os.remove(top_filename)
        os.remove(gro_filename)

        return nonbonditp_filename, bonditp_filename


def gen_ff_from_smiles(poly_name, poly_resname, smiles):
    """
    Generate PDB and parameter files from SMILES using external tools.
    """
    print({poly_name}, ": Generating OPLS parameter file ...")

    current_path = os.getcwd()
    MD_dir = os.path.join(current_path, 'MD_dir')

    try:
        Converter.convert(smiles=smiles,
                          resname=poly_resname,
                          charge=0,
                          opt=0,
                          outdir=MD_dir,)
        print(poly_name, ": OPLS parameter file generated.")
        os.rename('plt.pdb', f"{poly_name}.pdb")
    except Exception as e:  # Using Exception here to catch all possible exceptions
        print(f"Problem running LigParGen for {poly_name}: {e}")

    os.chdir(current_path)


def process_compound(compound_key, model_info, data_ff, out_dir, epsilon):

    current_path = os.getcwd()
    MD_dir = os.path.join(current_path, out_dir)
    if compound_key in model_info:
        compound_info = model_info[compound_key]
        compound_name = compound_info['compound']
        if compound_name in data_ff:
            files_to_copy = [
                f"pdb/{compound_name}.pdb",
                f"itp/{compound_name}_bonded.itp",
                f"itp/{compound_name}_nonbonded.itp"
            ]
            for file_path in files_to_copy:
                try:
                    resource_dir = pkg_resources.files('PEMD.forcefields')
                    resource_path = resource_dir.joinpath(file_path)
                    os.makedirs(MD_dir, exist_ok=True)
                    shutil.copy(str(resource_path), MD_dir)
                    print(f"Copied {file_path} to {out_dir} successfully.")
                except Exception as e:
                    print(f"Failed to copy {file_path}: {e}")

            corr_factor = compound_info['scale']
            target_sum_chg = compound_info['charge']
            filename = os.path.join(MD_dir, f"{compound_name}_bonded.itp")
            qm.scale_chg_itp(compound_name, out_dir, filename, corr_factor, target_sum_chg)
            print(f"scale charge successfully.")

        else:
            smiles = compound_info['smiles']
            corr_factor =  compound_info['scale']
            structures = qm.conformer_search_xtb(model_info, smiles, epsilon, core=32, polymer=False, work_dir=out_dir,
                                                 max_conformers=1000, top_n_MMFF=100, top_n_xtb=10, )

            sorted_df = qm.conformer_search_gaussian(structures, model_info, polymer=False, work_dir=out_dir,
                                                     core = 32, memory= '64GB', function='B3LYP', basis_set='6-311+g(d,p)',
                                                     dispersion_corr='em=GD3BJ', )

            qm.calc_resp_gaussian(sorted_df, model_info, epsilon, epsinf=2.1, polymer=False, work_dir=out_dir,
                                  numconf=5, core=32, memory='64GB', method='resp2', )

            print(f"Resp charge fitting for small moelcule successfully.")

            gen_ff_from_smiles(compound_info, out_dir)
            top_filename = os.path.join(MD_dir, f"{compound_name}.itp")
            nonbonditp_filename = os.path.join(MD_dir, f'{compound_name}_nonbonded.itp')
            bonditp_filename = os.path.join(MD_dir, f'{compound_name}_bonded.itp')
            model_lib.extract_from_top(top_filename, nonbonditp_filename, nonbonded=True, bonded=False)
            model_lib.extract_from_top(top_filename, bonditp_filename, nonbonded=False, bonded=True)
            print(f"{compound_key} generated from SMILES by ligpargen successfully.")

            qm.apply_chg_tomole(compound_name, out_dir, corr_factor, method='resp2', target_sum_chg=0, )
            print("apply charge to molecule force field successfully.")
    else:
        print(f"{compound_key} not found in model_info.")


def gen_oplsaa_ff_molecule(model_info, out_dir, epsilon):

    current_path = os.getcwd()
    MD_dir = os.path.join(current_path, out_dir)
    os.makedirs(MD_dir, exist_ok=True)  # Ensure the directory exists
    data_ff = ['Li', 'TFSI','SN','BMIM', 'EMIM', 'FSI', 'NO3']

    # Process each type of compound if present in model_info

    keys_list = [key for key in model_info.keys() if key != 'polymer']
    for compound_key in keys_list:
        process_compound(compound_key, model_info, data_ff, out_dir, epsilon)


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
        # if com == model_info['polymer']['compound']:
        #     ff_dir = os.path.join(current_path, f'{unit_name}_N{length}', 'ff_dir')
        #     filepath = os.path.join(ff_dir, f"{com}.pdb")
        #     nonbonditp_filepath = os.path.join(ff_dir, f'{com}_nonbonded.itp')
        #     bonditp_filepath = os.path.join(ff_dir, f'{com}_bonded.itp')
        #     shutil.copy(nonbonditp_filepath, MD_dir)
        #     shutil.copy(bonditp_filepath, MD_dir)
        # else:
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


# Define a function to execute commands and capture output
def run_command(command, input_text=None, output_file=None):
    # 确保命令是字符串列表
    if isinstance(command, str):
        command = command.split()

    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, text=True) as process:
        stdout, stderr = process.communicate(input=input_text)

    if output_file and stdout:
        with open(output_file, 'w') as file:
            file.write(stdout)

    return stdout, stderr


# generate top file for MD simulation
def gen_top_file(compound, resname, numbers, top_filename):
    file_contents = "; gromcs generation top file\n"
    file_contents += "; Created by PEMD\n\n"

    file_contents += "[ defaults ]\n"
    file_contents += ";nbfunc  comb-rule   gen-pairs   fudgeLJ  fudgeQQ\n"
    file_contents += "1        3           yes         0.5      0.5\n\n"

    file_contents += ";LOAD atomtypes\n"
    file_contents += "[ atomtypes ]\n"

    for com in compound:
        file_contents += f'#include "{com}_nonbonded.itp"\n'
    file_contents += "\n"
    for com in compound:
        file_contents += f'#include "{com}_bonded.itp"\n'
    file_contents += "\n"

    file_contents += "[ system ]\n"
    file_contents += ";name "
    for i in compound:
        file_contents += f"{i}"

    file_contents += "\n\n"

    file_contents += "[ molecules ]\n"
    for com, num in zip(resname, numbers):
        file_contents += f"{com} {num}\n"

    file_contents += "\n"

    # write to file
    with open(top_filename, 'w') as file:
        file.write(file_contents)
    print(f"Top file generation successful：{top_filename}")


# generation minimization mdp file
def gen_min_mdp_file(file_name = 'em.mdp'):
    file_contents = "; em.mdp - used as input into grompp to generate em.tpr\n"
    file_contents += "; Created by PEMD\n\n"

    file_contents += "integrator      = steep\n"
    file_contents += "nsteps          = 50000\n"
    file_contents += "emtol           = 1000.0\n"
    file_contents += "emstep          = 0.01\n\n"

    file_contents += ("; Parameters describing how to find the neighbors of each atom and how to calculate the "
                      "interactions\n")
    file_contents += "nstlist         = 1\n"
    file_contents += "cutoff-scheme   = Verlet\n"
    file_contents += "ns_type         = grid\n"
    file_contents += "rlist           = 1.0\n"
    file_contents += "coulombtype     = PME\n"
    file_contents += "rcoulomb        = 1.0\n"
    file_contents += "rvdw            = 1.0\n"
    file_contents += "pbc             = xyz\n"

    file_contents += "; output control is on\n"
    file_contents += "energygrps      = System\n"

    # write to file
    with open(file_name, 'w') as file:
        file.write(file_contents)
    print(f"Minimization mdp file generation successful：{file_name}")


# generation nvt mdp file
def gen_nvt_mdp_file(nsteps_nvt, nvt_temperature, file_name = 'nvt.mdp', ):
    file_contents = "; nvt.mdp - used as input into grompp to generate nvt.tpr\n"
    file_contents += "; Created by PEMD\n\n"

    file_contents += "; RUN CONTROL PARAMETERS\n"
    file_contents += "integrator            = md\n"
    file_contents += "dt                    = 0.001 \n"
    file_contents += f"nsteps                = {nsteps_nvt}\n"
    file_contents += "comm-mode             = Linear\n\n"

    file_contents += "; OUTPUT CONTROL OPTIONS\n"
    file_contents += "nstxout               = 5000\n"
    file_contents += "nstvout               = 5000\n"
    file_contents += "nstfout               = 5000\n"
    file_contents += "nstlog                = 5000\n"
    file_contents += "nstenergy             = 5000\n"
    file_contents += "nstxout-compressed    = 5000\n\n"

    file_contents += "; NEIGHBORSEARCHING PARAMETERS\n"
    file_contents += "cutoff-scheme         = verlet\n"
    file_contents += "ns_type               = grid\n"
    file_contents += "nstlist               = 20\n"
    file_contents += "rlist                 = 1.4\n"
    file_contents += "rcoulomb              = 1.4\n"
    file_contents += "rvdw                  = 1.4\n"
    file_contents += "verlet-buffer-tolerance = 0.005\n\n"

    file_contents += "; OPTIONS FOR ELECTROSTATICS AND VDW\n"
    file_contents += "coulombtype           = PME\n"
    file_contents += "vdw_type              = PME\n"
    file_contents += "fourierspacing        = 0.15\n"
    file_contents += "pme_order             = 4\n"
    file_contents += "ewald_rtol            = 1e-05\n\n"

    file_contents += "; OPTIONS FOR WEAK COUPLING ALGORITHMS\n"
    file_contents += "tcoupl                = v-rescale\n"
    file_contents += "tc-grps               = System\n"
    file_contents += "tau_t                 = 1.0\n"
    file_contents += f"ref_t                 = {nvt_temperature}\n"
    file_contents += "Pcoupl                = no\n"
    file_contents += "Pcoupltype            = isotropic\n"
    file_contents += "tau_p                 = 1.0\n"
    file_contents += "compressibility       = 4.5e-5\n"
    file_contents += "ref_p                 = 1.0\n\n"

    file_contents += "; GENERATE VELOCITIES FOR STARTUP RUN\n"
    file_contents += "gen_vel               = no\n\n"

    file_contents += "; OPTIONS FOR BONDS\n"
    file_contents += "constraints           = hbonds\n"
    file_contents += "constraint_algorithm  = lincs\n"
    file_contents += "unconstrained_start   = no\n"
    file_contents += "shake_tol             = 0.00001\n"
    file_contents += "lincs_order           = 4\n"
    file_contents += "lincs_warnangle       = 30\n"
    file_contents += "morse                 = no\n"
    file_contents += "lincs_iter            = 2\n"

    # write to file
    with open(file_name, 'w') as file:
        file.write(file_contents)
    print(f"NVT mdp file generation successful：{file_name}")


# generation npt mdp file
def gen_npt_mdp_file(nsteps_npt, npt_temperature, file_name = 'npt.mdp', ):
    file_contents = "; npt.mdp - used as input into grompp to generate npt.tpr\n"
    file_contents += "; Created by PEMD\n\n"

    file_contents += "; RUN CONTROL PARAMETERS\n"
    file_contents += "integrator            = md\n"
    file_contents += "dt                    = 0.001 \n"
    file_contents += f"nsteps                = {nsteps_npt}\n"
    file_contents += "comm-mode             = Linear\n\n"

    file_contents += "; OUTPUT CONTROL OPTIONS\n"
    file_contents += "nstxout               = 5000\n"
    file_contents += "nstvout               = 5000\n"
    file_contents += "nstfout               = 5000\n"
    file_contents += "nstlog                = 5000\n"
    file_contents += "nstenergy             = 5000\n"
    file_contents += "nstxout-compressed    = 5000\n\n"

    file_contents += "; NEIGHBORSEARCHING PARAMETERS\n"
    file_contents += "cutoff-scheme         = verlet\n"
    file_contents += "ns_type               = grid\n"
    file_contents += "nstlist               = 20\n"
    file_contents += "rlist                 = 1.4\n"
    file_contents += "rcoulomb              = 1.4\n"
    file_contents += "rvdw                  = 1.4\n"
    file_contents += "verlet-buffer-tolerance = 0.005\n\n"

    file_contents += "; OPTIONS FOR ELECTROSTATICS AND VDW\n"
    file_contents += "coulombtype           = PME\n"
    file_contents += "vdw_type              = PME\n"
    file_contents += "fourierspacing        = 0.15\n"
    file_contents += "pme_order             = 4\n"
    file_contents += "ewald_rtol            = 1e-05\n\n"

    file_contents += "; OPTIONS FOR WEAK COUPLING ALGORITHMS\n"
    file_contents += "tcoupl                = v-rescale\n"
    file_contents += "tc-grps               = System\n"
    file_contents += "tau_t                 = 1.0\n"
    file_contents += f"ref_t                 = {npt_temperature}\n"
    file_contents += "Pcoupl                = Berendsen\n"
    file_contents += "Pcoupltype            = isotropic\n"
    file_contents += "tau_p                 = 1.0\n"
    file_contents += "compressibility       = 4.5e-5\n"
    file_contents += "ref_p                 = 1.0\n\n"

    file_contents += "; GENERATE VELOCITIES FOR STARTUP RUN\n"
    file_contents += "gen_vel               = no\n\n"

    file_contents += "; OPTIONS FOR BONDS\n"
    file_contents += "constraints           = hbonds\n"
    file_contents += "constraint_algorithm  = lincs\n"
    file_contents += "unconstrained_start   = no\n"
    file_contents += "shake_tol             = 0.00001\n"
    file_contents += "lincs_order           = 4\n"
    file_contents += "lincs_warnangle       = 30\n"
    file_contents += "morse                 = no\n"
    file_contents += "lincs_iter            = 2\n"

    # write to file
    with open(file_name, 'w') as file:
        file.write(file_contents)
    print(f"NPT mdp file generation successful：{file_name}")


# generation npt anneal mdp file
def gen_npt_anneal_mdp_file(nsteps_annealing, npt_temperature, annealing_npoints, annealing_time, annealing_temp,
                            file_name):

    file_contents = "; npt_anneal.mdp - used as input into grompp to generate npt_anneal.tpr\n"
    file_contents += "; Created by PEMD\n\n"

    file_contents += "; RUN CONTROL PARAMETERS\n"
    file_contents += "integrator            = md\n"
    file_contents += "dt                    = 0.001 \n"
    file_contents += f"nsteps                = {nsteps_annealing}\n"
    file_contents += "comm-mode             = Linear\n\n"

    file_contents += "; OUTPUT CONTROL OPTIONS\n"
    file_contents += "nstxout               = 5000\n"
    file_contents += "nstvout               = 5000\n"
    file_contents += "nstfout               = 5000\n"
    file_contents += "nstlog                = 5000\n"
    file_contents += "nstenergy             = 5000\n"
    file_contents += "nstxout-compressed    = 5000\n\n"

    file_contents += "; NEIGHBORSEARCHING PARAMETERS\n"
    file_contents += "cutoff-scheme         = verlet\n"
    file_contents += "ns_type               = grid\n"
    file_contents += "nstlist               = 20\n"
    file_contents += "rlist                 = 1.4\n"
    file_contents += "rcoulomb              = 1.4\n"
    file_contents += "rvdw                  = 1.4\n"
    file_contents += "verlet-buffer-tolerance = 0.005\n\n"

    file_contents += "; OPTIONS FOR ELECTROSTATICS AND VDW\n"
    file_contents += "coulombtype           = PME\n"
    file_contents += "vdw_type              = PME\n"
    file_contents += "fourierspacing        = 0.15\n"
    file_contents += "pme_order             = 4\n"
    file_contents += "ewald_rtol            = 1e-05\n\n"

    file_contents += "; OPTIONS FOR WEAK COUPLING ALGORITHMS\n"
    file_contents += "tcoupl                = v-rescale\n"
    file_contents += "tc-grps               = System\n"
    file_contents += "tau_t                 = 1.0\n"
    file_contents += f"ref_t                 = {npt_temperature}\n"
    file_contents += "Pcoupl                = Berendsen\n"
    file_contents += "Pcoupltype            = isotropic\n"
    file_contents += "tau_p                 = 1.0\n"
    file_contents += "compressibility       = 4.5e-5\n"
    file_contents += "ref_p                 = 1.0\n\n"

    file_contents += "; Simulated annealing\n"
    file_contents += "annealing             = single\n"
    file_contents += f"annealing-npoints     = {annealing_npoints}\n"
    file_contents += f"annealing-time        = {annealing_time}\n"
    file_contents += f"annealing-temp        = {annealing_temp}\n\n"

    file_contents += "; GENERATE VELOCITIES FOR STARTUP RUN\n"
    file_contents += "gen_vel               = no\n\n"

    file_contents += "; OPTIONS FOR BONDS\n"
    file_contents += "constraints           = hbonds\n"
    file_contents += "constraint_algorithm  = lincs\n"
    file_contents += "unconstrained_start   = no\n"
    file_contents += "shake_tol             = 0.00001\n"
    file_contents += "lincs_order           = 4\n"
    file_contents += "lincs_warnangle       = 30\n"
    file_contents += "morse                 = no\n"
    file_contents += "lincs_iter            = 2\n"

    # write to file
    with open(file_name, 'w') as file:
        file.write(file_contents)
    print(f"NPT anneal mdp file generation successful：{file_name}")



























