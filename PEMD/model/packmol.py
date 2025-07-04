import os
import random
import subprocess
from shutil import which
from PEMD.model import polymer

class PEMDPackmol:
    def __init__(
            self,
            work_dir,
            molecule_list,
            density,
            add_length,
            packinp_name='pack.inp',
            packpdb_name='pack_cell.pdb'
    ):
        self.work_dir = work_dir

        if isinstance(molecule_list, dict):
            self.molecule_list = [
                {'name': name, 'number': number}
                for name, number in molecule_list.items()
            ]
        else:
            self.molecule_list = molecule_list

        self.density = density
        self.add_length = add_length
        self.packinp_name = packinp_name
        self.packpdb_name = packpdb_name

        self.compounds = [molecule['name'] for molecule in self.molecule_list]
        self.numbers = [molecule['number'] for molecule in self.molecule_list]

    def generate_input_file(self):

        os.makedirs(self.work_dir, exist_ok=True)
        packinp_path = os.path.join(self.work_dir, self.packinp_name)

        # compounds = list(self.molecule_list.keys())
        # numbers = list(self.molecule_list.values())

        pdb_filenames = []
        pdb_filepaths = []
        for com in self.compounds:
            filename = f"{com}.pdb"
            filepath = os.path.join(self.work_dir, f"{com}.pdb")
            pdb_filenames.append(filename)
            pdb_filepaths.append(filepath)

        box_length = polymer.calculate_box_size(self.numbers, pdb_filepaths, self.density) + self.add_length

        file_contents = "tolerance 2.0\n"
        file_contents += f"add_box_sides 1.2\n"
        file_contents += f"output {self.packpdb_name}\n"
        file_contents += "filetype pdb\n\n"
        file_contents += f"seed {random.randint(1, 100000)}\n\n"

        for num, file in zip(self.numbers, pdb_filenames):
            file_contents += f"structure {file}\n"
            file_contents += f"  number {num}\n"
            file_contents += f"  inside box 0.0 0.0 0.0 {box_length:.2f} {box_length:.2f} {box_length:.2f}\n"
            file_contents += "end structure\n\n"

        with open(packinp_path, 'w') as file:
            file.write(file_contents)
        print(f"Packmol input file generation successful: {packinp_path}")

        return packinp_path

    def run_local(self, ):
        current_path = os.getcwd()
        if not which("packmol"):
            raise RuntimeError(
                "Running Packmol requires the executable 'packmol' to be in the PATH. Please "
                "download Packmol from https://github.com/leandromartinez98/packmol and follow the "
                "instructions in the README to compile. Don't forget to add the Packmol binary to your PATH."
            )

        try:
            os.chdir(self.work_dir)
            p = subprocess.run(
                f"packmol < {self.packinp_name}",
                check=True,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Write Packmol output to the specified output file in the current directory
            with open('pack.out', mode="w") as out:
                out.write(p.stdout)

            print("Packmol executed successfully.")

        except subprocess.CalledProcessError as exc:
            if exc.returncode not in [172, 173]:
                raise ValueError(
                    f"Packmol failed with error code {exc.returncode}.\n"
                    f"Stdout:\n{exc.stdout}\n"
                    f"Stderr:\n{exc.stderr}"
                ) from exc
            else:
                print(f"Packmol ended with 'STOP {exc.returncode}', but this error is being ignored.")
                print(f"Stdout:\n{exc.stdout}\nStderr:\n{exc.stderr}")

        finally:
            # Change back to the original working directory
            os.chdir(current_path)