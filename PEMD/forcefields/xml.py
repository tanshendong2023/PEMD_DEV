import pandas as pd
import re
from rdkit import Chem
import math


class XMLGenerator:
    def __init__(self, itp_filename, mol_or_smiles, xml_filename):
        self.itp_filename = itp_filename
        if isinstance(mol_or_smiles, Chem.Mol):
            self.mol = mol_or_smiles
            self.smiles = None
        else:
            self.smiles = mol_or_smiles
            self.mol = None
        self.xml_filename = xml_filename
        self.atoms_df = None
        self.atomtypes_df = None
        self.bonds_df = None
        self.angles_df = None
        self.dihedrals_df = None
        self.smarts_df = None
        self.class_element_map = {}
        self.xml_blocks = {}
        self.unique_type_to_class = {}

    def parse_itp_section(self, itp_content, section_name):
        # Use a regular expression to extract the requested section. Ensure that
        # the final section at the end of the file is also captured correctly.
        pattern = rf"\[\s*{section_name}\s*\](.*?)\n\["  # Match up to the next section header
        match = re.search(pattern, itp_content, re.DOTALL)
        if not match:
            # If the next section header is missing, capture everything to the end
            pattern = rf"\[\s*{section_name}\s*\](.*)"
            match = re.search(pattern, itp_content, re.DOTALL)
            if not match:
                print(f"Unable to find section [{section_name}].")
                return None

        section_content = match.group(1).strip()

        # Split lines while skipping comments and blanks
        lines = section_content.split('\n')
        lines = [line for line in lines if line.strip() and not line.strip().startswith(';')]

        # Parse according to the section name
        if section_name.lower() == 'atoms':
            # Define column headers
            columns = ['nr', 'type', 'resnr', 'residue', 'atom', 'cgnr', 'charge', 'mass']
            data = []
            for line in lines:
                parts = line.split()
                if len(parts) < 8:
                    continue  # Skip incomplete rows
                nr = int(parts[0])
                type_ = parts[1]
                resnr = int(parts[2])
                residue = parts[3]
                atom = parts[4]
                cgnr = int(parts[5])
                charge = float(parts[6])
                mass = float(parts[7])
                data.append([nr, type_, resnr, residue, atom, cgnr, charge, mass])
            self.atoms_df = pd.DataFrame(data, columns=columns)
            return self.atoms_df

        elif section_name.lower() == 'atomtypes':
            # Define column headers
            columns = ['name', 'class', 'mass', 'charge', 'ptype', 'sigma', 'epsilon']
            data = []
            for line in lines:
                parts = line.split()
                if len(parts) < 7:
                    continue  # Skip incomplete rows
                name = parts[0]
                class_ = parts[1]
                mass = float(parts[2])
                charge = float(parts[3])
                ptype = parts[4]
                sigma = float(parts[5])
                epsilon = float(parts[6])
                data.append([name, class_, mass, charge, ptype, sigma, epsilon])
            self.atomtypes_df = pd.DataFrame(data, columns=columns)
            return self.atomtypes_df

        elif section_name.lower() == 'bonds':
            # Define column headers
            columns = ['ai', 'aj', 'funct', 'c0', 'c1']
            data = []
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue  # Skip incomplete rows
                ai = int(parts[0])
                aj = int(parts[1])
                funct = int(parts[2])
                c0 = float(parts[3])
                c1 = float(parts[4])
                data.append([ai, aj, funct, c0, c1])
            self.bonds_df = pd.DataFrame(data, columns=columns)
            return self.bonds_df

        elif section_name.lower() == 'angles':
            # Define column headers
            columns = ['ai', 'aj', 'ak', 'funct', 'c0', 'c1']
            data = []
            for line in lines:
                parts = line.split()
                if len(parts) < 6:
                    continue  # Skip incomplete rows
                ai = int(parts[0])
                aj = int(parts[1])
                ak = int(parts[2])
                funct = int(parts[3])
                c0 = float(parts[4])
                c1 = float(parts[5])
                data.append([ai, aj, ak, funct, c0, c1])
            self.angles_df = pd.DataFrame(data, columns=columns)
            return self.angles_df

        elif section_name.lower() == 'dihedrals':
            # Define column headers
            columns = ['ai', 'aj', 'ak', 'al', 'funct', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5']
            data = []
            for line in lines:
                parts = line.split()
                if len(parts) < 11:
                    continue  # Skip incomplete rows
                ai = int(parts[0])
                aj = int(parts[1])
                ak = int(parts[2])
                al = int(parts[3])
                funct = int(parts[4])
                c0 = float(parts[5])
                c1 = float(parts[6])
                c2 = float(parts[7])
                c3 = float(parts[8])
                c4 = float(parts[9])
                c5 = float(parts[10])
                data.append([ai, aj, ak, al, funct, c0, c1, c2, c3, c4, c5])
            self.dihedrals_df = pd.DataFrame(data, columns=columns)
            return self.dihedrals_df

        else:
            print(f"Unknown section name: {section_name}")
            return None

    def generate_first_coordination_smarts_with_details(self, include_h=True):

        if self.mol is not None:
            mol = self.mol
        else:
            mol = Chem.MolFromSmiles(self.smiles)
            if mol is None:
                print("Invalid SMILES string.")
                return None
            mol = Chem.AddHs(mol)

        data = []
        # Iterate over every atom in the molecule (including hydrogens)
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            atom_symbol = atom.GetSymbol()

            # Compute the coordination number Xn (including hydrogens)
            xn = atom.GetDegree()

            # Initialize the SMARTS string with the central atom and coordination number
            smarts = f"[{atom_symbol};X{xn}]"

            # # Iterate over all bonds of the atom to construct the SMARTS description
            # for bond in atom.GetBonds():
            #     # Identify the bonded neighbor
            #     if bond.GetBeginAtomIdx() == atom_idx:
            #         neighbor = bond.GetEndAtom()
            #     else:
            #         neighbor = bond.GetBeginAtom()
            #
            #     neighbor_symbol = neighbor.GetSymbol()
            #
            #     if neighbor_symbol == 'H' and not include_h:
            #         continue  # Skip hydrogens if they are excluded
            #
            #     # Compute the neighbor's coordination number Xn (including hydrogens)
            #     neighbor_xn = neighbor.GetDegree()
            #
            #     # Build the neighbor SMARTS fragment
            #     neighbor_smarts = f"[{neighbor_symbol};X{neighbor_xn}]"
            #
            #     # Append to the SMARTS string, including bond symbols if needed
            #     smarts += f"({neighbor_smarts})"
            # Collect SMARTS for all neighboring atoms
            neighbors_smarts = []

            for bond in atom.GetBonds():
                # Identify the neighboring atom for the current bond
                if bond.GetBeginAtomIdx() == atom_idx:
                    neighbor = bond.GetEndAtom()
                else:
                    neighbor = bond.GetBeginAtom()

                neighbor_symbol = neighbor.GetSymbol()

                if neighbor_symbol == 'H' and not include_h:
                    continue  # Skip hydrogens if requested

                # Compute the coordination number Xn for the neighbor (including hydrogens)
                neighbor_xn = neighbor.GetDegree()

                # Build the SMARTS snippet for the neighbor
                neighbor_smarts = f"[{neighbor_symbol};X{neighbor_xn}]"

                neighbors_smarts.append(neighbor_smarts)

            # Sort the neighbor SMARTS strings
            neighbors_smarts_sorted = sorted(neighbors_smarts)

            # Re-initialize the SMARTS string for the central atom
            smarts = f"[{atom_symbol};X{xn}]"

            # Append the sorted neighbor SMARTS
            for neighbor_smarts in neighbors_smarts_sorted:
                smarts += f"({neighbor_smarts})"

            # Add the result to the collection
            data.append({
                "Atom Index": atom_idx,
                "Atom Symbol": atom_symbol,
                "SMARTS": smarts
            })

        # Convert to a DataFrame
        self.smarts_df = pd.DataFrame(data)
        return self.smarts_df

    def degrees_to_radians(self, degrees):
        return degrees * (math.pi / 180)

    def create_class_element_map(self):
        # Define the mapping from class names to elements. Ensure each class name
        # is unique and mapped to the correct element.
        self.class_element_map = {
            'H': 'H',
            'HC': 'H',
            'HO': 'H',
            'HS': 'H',
            'HW': 'H',
            'HA': 'H',
            'H3': 'H',
            'C': 'C',
            'CT': 'C',
            'C2': 'C',
            'C3': 'C',
            'C4': 'C',
            'C7': 'C',
            'C8': 'C',
            'C9': 'C',
            'CD': 'C',
            'CZ': 'C',
            'CH': 'C',
            'CM': 'C',
            'C=': 'C',
            'CA': 'C',
            'CB': 'C',
            'CO': 'C',
            'CN': 'C',
            'CV': 'C',
            'CW': 'C',
            'CR': 'C',
            'CX': 'C',
            'C!': 'C',
            'CU': 'C',
            'CS': 'C',
            'CQ': 'C',
            'CY': 'C',
            'C+': 'C',
            'CF': 'C',
            'C#': 'C',
            'CP': 'C',
            'C:': 'C',
            'C_2': 'C',
            'N': 'N',
            'N2': 'N',
            'N3': 'N',
            'NT': 'N',
            'NA': 'N',
            'NC': 'N',
            'NO': 'N',
            'NZ': 'N',
            'NB': 'N',
            'N$': 'N',
            'N*': 'N',
            'NM': 'N',
            'NX': 'N',
            'O': 'O',
            'OS': 'O',
            'ON': 'O',
            'OW': 'O',
            'OY': 'O',
            'O2': 'O',
            'OU': 'O',
            'O_3': 'O',
            'O_2': 'O',
            'O_1': 'O',
            'O_': 'O',
            'OH': 'O',
            'F': 'F',
            'Si': 'Si',
            'P': 'P',
            'S=': 'S',
            'S': 'S',
            'SH': 'S',
            'SZ': 'S',
            'SA': 'S',
            'Cl': 'Cl',
            'Br': 'Br',
        }

    def generate_xml_blocks(self):
        if self.smarts_df is None:
            print("SMARTS descriptions have not been generated.")
            return

        # Convert ``Atom Index`` in ``smarts_df`` to ``nr`` (one-based indexing)
        self.smarts_df['nr'] = self.smarts_df['Atom Index'] + 1
        merged_df = pd.merge(self.atoms_df, self.smarts_df, on='nr', how='left')

        # Check for atoms that failed to match a SMARTS description
        if merged_df['SMARTS'].isnull().any():
            print("Warning: SMARTS descriptions were not found for some atoms.")

        # Identify unique SMARTS strings
        unique_smarts = merged_df['SMARTS'].unique()

        # Assign sequential type names (e.g., opls_1, opls_2, …) to each unique SMARTS
        smarts_to_unique_type = {smarts: f"opls_{i + 1}" for i, smarts in enumerate(unique_smarts)}

        # Create a ``unique_type`` column based on the SMARTS mapping
        merged_df['unique_type'] = merged_df['SMARTS'].map(smarts_to_unique_type)

        # Build the <AtomTypes> block
        atomtypes_block = "<AtomTypes>\n"
        missing_types = []
        for smarts, unique_type in smarts_to_unique_type.items():
            # Locate the first matching atom type
            matching_rows = merged_df[merged_df['SMARTS'] == smarts]
            if matching_rows.empty:
                print(f"Warning: SMARTS '{smarts}' has no corresponding atom type.")
                missing_types.append(smarts)
                continue  # Skip this SMARTS entry

            corresponding_type = matching_rows['type'].iloc[0]
            # Retrieve the atom type information
            atomtype_info = self.atomtypes_df[self.atomtypes_df['name'] == corresponding_type]
            if not atomtype_info.empty:
                class_ = atomtype_info.iloc[0]['class']
                mass = atomtype_info.iloc[0]['mass']
                # Determine the element symbol (default to carbon)
                element = self.class_element_map.get(class_, 'C')
                atomtypes_block += f'  <Type name="{unique_type}" class="{class_}" element="{element}" mass="{mass}" def="{smarts}"/>\n'
                self.unique_type_to_class[unique_type] = class_
            else:
                print(f"Warning: Type '{corresponding_type}' not found in [ atomtypes ]. Using defaults.")
                class_ = 'C'
                element = 'C'
                mass = 12.011
                atomtypes_block += f'  <Type name="{unique_type}" class="{class_}" element="{element}" mass="{mass}" def="{smarts}"/>\n'
                self.unique_type_to_class[unique_type] = class_
        atomtypes_block += "</AtomTypes>\n"
        self.xml_blocks["AtomTypes"] = atomtypes_block

        # Build the <NonbondedForce> block
        nonbonded_block = '<NonbondedForce coulomb14scale="0.5" lj14scale="0.5">\n'

        # Aggregate charges by ``unique_type``
        unique_nonbonded = merged_df.groupby('unique_type').agg({
            'charge': 'first',
            'type': 'first'  # Original type used to find sigma and epsilon
        }).reset_index()

        # Extract the numeric suffix from ``opls_x`` for numeric sorting
        unique_nonbonded['type_num'] = unique_nonbonded['unique_type'].str.extract(r'opls_(\d+)').astype(int)

        # Sort by ``type_num`` ascending
        unique_nonbonded = unique_nonbonded.sort_values('type_num')

        for _, row in unique_nonbonded.iterrows():
            unique_type = row['unique_type']
            charge = row['charge']
            original_type = row['type']
            # Fetch sigma and epsilon from ``atomtypes_df``
            atomtype_info = self.atomtypes_df[self.atomtypes_df['name'] == original_type]
            if not atomtype_info.empty:
                sigma = atomtype_info.iloc[0]['sigma']
                epsilon = atomtype_info.iloc[0]['epsilon']
            else:
                print(f"Warning: Type '{original_type}' not found in [ atomtypes ]. Using defaults.")
                sigma = 0.35  # Default value
                epsilon = 0.276144  # Default value

            nonbonded_block += f'    <Atom type="{unique_type}" charge="{charge}" sigma="{sigma}" epsilon="{epsilon}"/>\n'
        nonbonded_block += "</NonbondedForce>\n"
        self.xml_blocks["NonbondedForce"] = nonbonded_block

        # Build the <HarmonicBondForce> block
        harmonic_bond_block = "<HarmonicBondForce>\n"
        unique_bonds = set()  # Track processed bonds

        if self.bonds_df is not None:
            for _, bond in self.bonds_df.iterrows():
                ai = bond['ai']
                aj = bond['aj']
                c0 = bond['c0']  # bond length
                c1 = bond['c1']  # force constant

                # Retrieve the ``unique_type`` values
                type_i = merged_df[merged_df['nr'] == ai]['unique_type'].iloc[0]
                type_j = merged_df[merged_df['nr'] == aj]['unique_type'].iloc[0]

                # Look up the corresponding classes
                class_i = self.unique_type_to_class.get(type_i, 'C')
                class_j = self.unique_type_to_class.get(type_j, 'C')

                # Create a unique identifier for the bond to avoid duplicates
                bond_tuple = (class_i, class_j, c0, c1)

                if bond_tuple not in unique_bonds:
                    unique_bonds.add(bond_tuple)
                    harmonic_bond_block += f'  <Bond class1="{class_i}" class2="{class_j}" length="{c0}" k="{c1}"/>\n'
        harmonic_bond_block += "</HarmonicBondForce>\n"
        self.xml_blocks["HarmonicBondForce"] = harmonic_bond_block

        # Build the <HarmonicAngleForce> block
        harmonic_angle_block = "<HarmonicAngleForce>\n"
        unique_angles = set()  # Track processed angles

        if self.angles_df is not None:
            for _, angle in self.angles_df.iterrows():
                ai = angle['ai']
                aj = angle['aj']
                ak = angle['ak']
                c0 = angle['c0']  # equilibrium angle in degrees
                c1 = angle['c1']  # force constant

                # Convert the equilibrium angle from degrees to radians
                c0_rad = self.degrees_to_radians(c0)

                # Retrieve ``unique_type`` values
                type_i = merged_df[merged_df['nr'] == ai]['unique_type'].iloc[0]
                type_j = merged_df[merged_df['nr'] == aj]['unique_type'].iloc[0]
                type_k = merged_df[merged_df['nr'] == ak]['unique_type'].iloc[0]

                # Look up the corresponding classes
                class_i = self.unique_type_to_class.get(type_i, 'C')
                class_j = self.unique_type_to_class.get(type_j, 'C')
                class_k = self.unique_type_to_class.get(type_k, 'C')

                # Format the radian value to eight decimal places
                c0_rad_formatted = f"{c0_rad:.8f}"

                # Create a unique identifier for the angle to avoid duplicates
                angle_tuple = (class_i, class_j, class_k, c0_rad_formatted, c1)

                if angle_tuple not in unique_angles:
                    unique_angles.add(angle_tuple)
                    harmonic_angle_block += f'  <Angle class1="{class_i}" class2="{class_j}" class3="{class_k}" angle="{c0_rad_formatted}" k="{c1}"/>\n'
        harmonic_angle_block += "</HarmonicAngleForce>\n"
        self.xml_blocks["HarmonicAngleForce"] = harmonic_angle_block

        # Build the <RBTorsionForce> block
        rb_torsion_block = "<RBTorsionForce>\n"
        unique_propers = set()  # Track processed torsions

        if self.dihedrals_df is not None:
            for _, dihedral in self.dihedrals_df.iterrows():
                ai = dihedral['ai']
                aj = dihedral['aj']
                ak = dihedral['ak']
                al = dihedral['al']
                c0 = dihedral['c0']
                c1 = dihedral['c1']
                c2 = dihedral['c2']
                c3 = dihedral['c3']
                c4 = dihedral['c4']
                c5 = dihedral['c5']

                # Retrieve ``unique_type`` values
                type_i = merged_df[merged_df['nr'] == ai]['unique_type'].iloc[0]
                type_j = merged_df[merged_df['nr'] == aj]['unique_type'].iloc[0]
                type_k = merged_df[merged_df['nr'] == ak]['unique_type'].iloc[0]
                type_l = merged_df[merged_df['nr'] == al]['unique_type'].iloc[0]

                # Look up the corresponding classes
                class_i = self.unique_type_to_class.get(type_i, 'C')
                class_j = self.unique_type_to_class.get(type_j, 'C')
                class_k = self.unique_type_to_class.get(type_k, 'C')
                class_l = self.unique_type_to_class.get(type_l, 'C')

                # Create a unique identifier for the torsion to avoid duplicates
                proper_tuple = (class_i, class_j, class_k, class_l, c0, c1, c2, c3, c4, c5)

                if proper_tuple not in unique_propers:
                    unique_propers.add(proper_tuple)
                    rb_torsion_block += f'  <Proper class1="{class_i}" class2="{class_j}" class3="{class_k}" class4="{class_l}" c0="{c0}" c1="{c1}" c2="{c2}" c3="{c3}" c4="{c4}" c5="{c5}"/>\n'
        rb_torsion_block += "</RBTorsionForce>\n"
        self.xml_blocks["RBTorsionForce"] = rb_torsion_block

    def write_xml_file(self):
        try:
            with open(self.xml_filename, 'w') as f:
                f.write('<ForceField name="OPLS-AA" version="1.0.1" combining_rule="geometric">\n')
                for block_name in ["AtomTypes", "HarmonicBondForce", "HarmonicAngleForce", "RBTorsionForce",
                                   "NonbondedForce", "Pairs"]:
                    if block_name in self.xml_blocks:
                        f.write(self.xml_blocks[block_name])
                f.write('</ForceField>\n')
            # print(f"Generated XML file have been written to '{self.xml_filename}'.")
        except Exception as e:
            print(f"Error writing to file: {e}")

    def run(self):
        # Read the .itp file
        try:
            with open(self.itp_filename, 'r') as file:
                itp_content = file.read()
        except FileNotFoundError:
            print(f"File not found: {self.itp_filename}")
            return
        except Exception as e:
            print(f"Error reading file: {e}")
            return

        # Parse sections of the file
        self.atoms_df = self.parse_itp_section(itp_content, 'atoms')
        if self.atoms_df is None:
            print("Failed to parse [ atoms ] section.")
            return

        self.atomtypes_df = self.parse_itp_section(itp_content, 'atomtypes')
        if self.atomtypes_df is None:
            print("Failed to parse [ atomtypes ] section.")
            return

        self.bonds_df = self.parse_itp_section(itp_content, 'bonds')
        if self.bonds_df is None:
            print("Warning: Failed to parse [ bonds ] section.")

        self.angles_df = self.parse_itp_section(itp_content, 'angles')
        if self.angles_df is None:
            print("Warning: Failed to parse [ angles ] section.")

        self.dihedrals_df = self.parse_itp_section(itp_content, 'dihedrals')
        if self.dihedrals_df is None:
            print("Warning: Failed to parse [ dihedrals ] section.")

        # Generate SMARTS descriptors
        self.generate_first_coordination_smarts_with_details(include_h=True)
        if self.smarts_df is None:
            print("Failed to generate SMARTS descriptors.")
            return

        # Map class to elements and generate XML blocks
        self.create_class_element_map()
        self.generate_xml_blocks()
        self.write_xml_file()


