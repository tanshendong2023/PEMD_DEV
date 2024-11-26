
import os
import subprocess


class PEMDLAMMPS:
    def __init__(
        self,
        lammps_dir,
        core,
    ):
        self.lammps_dir = lammps_dir
        self.core = core

    def generate_input_file(self, name):

        file_contents = f"################## Initialization ##################\n"
        file_contents += f"units		real\n"
        file_contents += f"boundary		s s s\n"
        file_contents += f"dimension	3\n\n"

        file_contents += f"atom_style		full\n"
        file_contents += f"bond_style		harmonic\n"
        file_contents += f"angle_style      harmonic\n"
        file_contents += f"dihedral_style   fourier\n"
        file_contents += f"improper_style   cvff\n"
        file_contents += f"pair_style       lj/cut 2.0\n\n"

        file_contents += f"################## Model ##################\n"
        file_contents += f"read_data        {self.lammps_dir}/{name}_gaff2.lmp\n\n"

        file_contents += f"################## Therm ##################\n"
        file_contents += f"thermo           100\n"
        file_contents += f"thermo_style     custom step temp pxx pyy pzz ebond eangle edihed eimp epair ecoul evdwl pe ke etotal lx ly lz vol density\n\n"

        file_contents += f"################## Confinement ##################\n"
        file_contents += f"fix              xwalls all wall/reflect xlo EDGE xhi EDGE\n"
        file_contents += f"fix              ywalls all wall/reflect ylo EDGE yhi EDGE\n"
        file_contents += f"fix              zwalls all wall/reflect zlo EDGE zhi EDGE\n\n"

        file_contents += f"################## Minimze ##################\n"
        file_contents += f"velocity         all create 300 3243242\n"
        file_contents += f"minimize         1e-8 1e-8 10000 10000\n\n"

        file_contents += f"################## Annealing ##################\n"
        file_contents += f"dump             1 all custom 500 {self.lammps_dir}/soft.lammpstrj id type mass mol x y z\n"
        file_contents += f"fix              1 all nvt temp 800 800 100 drag 2\n"
        file_contents += f"run              1000000\n"
        file_contents += f"write_data       {self.lammps_dir}/{name}_gaff2.data\n"
        file_contents += f"write_dump       all xyz {self.lammps_dir}/{name}_lmp.xyz\n\n"

        input_file_path = os.path.join(self.lammps_dir, 'lammps.in')
        with open(input_file_path, 'w') as file:
            file.write(file_contents)

    def run_local(self, ):

        input_file = os.path.join(self.lammps_dir, 'lammps.in')
        log_file = os.path.join(self.lammps_dir, 'log.lammps')
        output_file = os.path.join(self.lammps_dir, 'out.lmp')
        error_file = os.path.join(self.lammps_dir, 'lmp.err')

        command = (
            f"mpiexec -np {self.core} lmp -in {input_file} -log {log_file} > {output_file} 2> {error_file}"
        )

        try:
            result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
            print("The LAMMPS simulation has successfully completed.\n")
            return result.stdout  # Return the standard output from the Gaussian command
        except subprocess.CalledProcessError as e:
            print(f"Error executing LAMMPS: {e}")
            return e.stderr

    # def gen_slurm(self, script_name, job_name, nodes, ntasks_per_node, partition,):
    #     slurm_script = PEMDSlurm(
    #         self.work_dir,
    #         script_name,
    #     )
    #
    #     # 添加命令以改变目录到 lammps_dir 并运行 LAMMPS 脚本
    #     change_dir_and_run_command = f"cd {self.lammps_dir} && bash runLAMMPS.sh"
    #     slurm_script.add_command(change_dir_and_run_command)
    #
    #     # 生成 SLURM 脚本
    #     script_path = slurm_script.generate_script(
    #         job_name,
    #         nodes,
    #         ntasks_per_node,
    #         partition,
    #     )
    #
    #     return script_path
