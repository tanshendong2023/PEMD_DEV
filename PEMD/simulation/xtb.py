# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# Module Docstring
# ******************************************************************************

import subprocess
import os
import logging
from jinja2 import Template
import re
import json

class PEMDXtb:
    def __init__(
            self,
            work_dir,
            chg=0,
            mult=1,
            gfn=2,
    ):
        self.work_dir = work_dir
        self.chg = chg
        self.mult = mult
        self.gfn = gfn

        # Create the working directory if it does not exist
        os.makedirs(self.work_dir, exist_ok=True)

        # Configure logging
        log_file = os.path.join(work_dir, 'xtb.log')
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info("PEMDXtb instance created.")

    def run_local(self, xyz_filename, outfile_headname, optimize=True):

        uhf = self.mult - 1  # Number of unpaired electrons

        # Define the command template
        command_template = Template(
            "xtb {{ xyz_filename }} {% if optimize %}--opt {% endif %} "
            "--chrg={{ chg }} "
            "--uhf={{ uhf }} "
            "--gfn {{ gfn }} "
            "--ceasefiles "
            "--namespace {{ work_dir }}/{{ outfile_headname }}"
        )

        # Render the command
        command_str = command_template.render(
            xyz_filename=xyz_filename,
            optimize=optimize,
            chg=self.chg,
            uhf=uhf,
            gfn=self.gfn,
            work_dir=self.work_dir,
            outfile_headname=outfile_headname
        )

        logging.info(f"Executing command: {command_str}")
        # print(f"Executing command: {command_str}")

        try:
            result = subprocess.run(command_str, shell=True, check=True, text=True, capture_output=True)
            logging.info(f"XTB output: {result.stdout}")
            # print(f"XTB output: {result.stdout}")

            # Extract energy information
            energy_info = self.extract_energy(result.stdout, optimized=optimize)
            if energy_info:
                logging.info(f"Extracted energy information: {energy_info}")
                # print(f"Extracted energy information: {energy_info}")
                # Save the energy data to JSON
                energy_file = os.path.join(self.work_dir, f"{outfile_headname}_energy.json")
                with open(energy_file, 'w') as ef:
                    json.dump(energy_info, ef, indent=4)
                logging.info(f"Energy information saved to {energy_file}")
            else:
                logging.error("Failed to extract energy information.")

            return {
                'output': result.stdout,
                'energy_info': energy_info,
                'error': None
            }

        except subprocess.CalledProcessError as e:
            logging.error(f"Error executing XTB: {e.stderr}")
            print(f"Error executing XTB: {e.stderr}")
            return {
                'output': None,
                'energy_info': None,
                'error': e.stderr
            }

    def extract_energy(self, stdout, optimized=False):

        energy_info = {}
        patterns = {
            'total_energy': r"::\s*total energy\s*([-+]?\d*\.\d+|\d+) Eh",
        }

        for key, pattern in patterns.items():
            matches = re.findall(pattern, stdout, re.IGNORECASE)
            if matches:
                try:
                    if optimized:
                        energy = float(matches[-1])  # Use the last match
                    else:
                        energy = float(matches[0])   # Use the first match
                    energy_info[key] = energy
                except ValueError:
                    logging.error(
                        f"Extracted energy value could not be converted to float: "
                        f"{matches[-1] if optimized else matches[0]}"
                    )
            else:
                logging.warning(f"'{key}' not found in the output.")

        if energy_info:
            return energy_info
        else:
            return None


