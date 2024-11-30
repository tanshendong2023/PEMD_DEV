import pandas as pd
import re
from rdkit import Chem
import math


class XMLGenerator:
    def __init__(self, itp_filename, smiles, xml_filename):
        self.itp_filename = itp_filename
        self.smiles = smiles
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
        # 使用正则表达式提取指定部分
        # 注意：确保在文件末尾部分也能正确提取
        pattern = rf"\[\s*{section_name}\s*\](.*?)\n\["  # 匹配到下一个部分的开始
        match = re.search(pattern, itp_content, re.DOTALL)
        if not match:
            # 如果没有找到下一个部分，尝试提取到文件末尾
            pattern = rf"\[\s*{section_name}\s*\](.*)"
            match = re.search(pattern, itp_content, re.DOTALL)
            if not match:
                print(f"无法找到[{section_name}]部分。")
                return None

        section_content = match.group(1).strip()

        # 分割行，忽略注释和空行
        lines = section_content.split('\n')
        lines = [line for line in lines if line.strip() and not line.strip().startswith(';')]

        # 根据部分名称进行解析
        if section_name.lower() == 'atoms':
            # 定义列名
            columns = ['nr', 'type', 'resnr', 'residue', 'atom', 'cgnr', 'charge', 'mass']
            data = []
            for line in lines:
                parts = line.split()
                if len(parts) < 8:
                    continue  # 跳过不完整的行
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
            # 定义列名
            columns = ['name', 'class', 'mass', 'charge', 'ptype', 'sigma', 'epsilon']
            data = []
            for line in lines:
                parts = line.split()
                if len(parts) < 7:
                    continue  # 跳过不完整的行
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
            # 定义列名
            columns = ['ai', 'aj', 'funct', 'c0', 'c1']
            data = []
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue  # 跳过不完整的行
                ai = int(parts[0])
                aj = int(parts[1])
                funct = int(parts[2])
                c0 = float(parts[3])
                c1 = float(parts[4])
                data.append([ai, aj, funct, c0, c1])
            self.bonds_df = pd.DataFrame(data, columns=columns)
            return self.bonds_df

        elif section_name.lower() == 'angles':
            # 定义列名
            columns = ['ai', 'aj', 'ak', 'funct', 'c0', 'c1']
            data = []
            for line in lines:
                parts = line.split()
                if len(parts) < 6:
                    continue  # 跳过不完整的行
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
            # 定义列名
            columns = ['ai', 'aj', 'ak', 'al', 'funct', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5']
            data = []
            for line in lines:
                parts = line.split()
                if len(parts) < 11:
                    continue  # 跳过不完整的行
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
            print(f"未知的部分名称: {section_name}")
            return None

    def generate_first_coordination_smarts_with_details(self, include_h=True):
        # 解析SMILES
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            print("无效的SMILES字符串。")
            return None
        mol = Chem.AddHs(mol)

        data = []
        # 遍历分子中的每个原子（包括氢原子）
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            atom_symbol = atom.GetSymbol()

            # 计算配位数Xn（包括氢原子）
            xn = atom.GetDegree()

            # 初始化SMARTS字符串，表示中心原子及其配位数
            smarts = f"[{atom_symbol};X{xn}]"

            # # 遍历该原子的所有键，构建SMARTS描述
            # for bond in atom.GetBonds():
            #     # 确定连接的邻接原子
            #     if bond.GetBeginAtomIdx() == atom_idx:
            #         neighbor = bond.GetEndAtom()
            #     else:
            #         neighbor = bond.GetBeginAtom()
            #
            #     neighbor_symbol = neighbor.GetSymbol()
            #
            #     if neighbor_symbol == 'H' and not include_h:
            #         continue  # 如果不包含H，跳过氢原子
            #
            #     # 计算邻接原子的配位数Xn（包括氢原子）
            #     neighbor_xn = neighbor.GetDegree()
            #
            #     # 构建邻接原子的SMARTS
            #     neighbor_smarts = f"[{neighbor_symbol};X{neighbor_xn}]"
            #
            #     # 添加到SMARTS字符串，包含键符号（如果需要）
            #     smarts += f"({neighbor_smarts})"
            # 收集所有邻接原子的SMARTS
            neighbors_smarts = []

            for bond in atom.GetBonds():
                # 确定连接的邻接原子
                if bond.GetBeginAtomIdx() == atom_idx:
                    neighbor = bond.GetEndAtom()
                else:
                    neighbor = bond.GetBeginAtom()

                neighbor_symbol = neighbor.GetSymbol()

                if neighbor_symbol == 'H' and not include_h:
                    continue  # 如果不包含H，跳过氢原子

                # 计算邻接原子的配位数Xn（包括氢原子）
                neighbor_xn = neighbor.GetDegree()

                # 构建邻接原子的SMARTS
                neighbor_smarts = f"[{neighbor_symbol};X{neighbor_xn}]"

                neighbors_smarts.append(neighbor_smarts)

            # 对邻接原子的SMARTS进行排序
            neighbors_smarts_sorted = sorted(neighbors_smarts)

            # 初始化SMARTS字符串，表示中心原子及其配位数
            smarts = f"[{atom_symbol};X{xn}]"

            # 添加排序后的邻接原子SMARTS
            for neighbor_smarts in neighbors_smarts_sorted:
                smarts += f"({neighbor_smarts})"

            # 将结果添加到数据列表
            data.append({
                "Atom Index": atom_idx,
                "Atom Symbol": atom_symbol,
                "SMARTS": smarts
            })

        # 转换为DataFrame
        self.smarts_df = pd.DataFrame(data)
        return self.smarts_df

    def degrees_to_radians(self, degrees):
        return degrees * (math.pi / 180)

    def create_class_element_map(self):
        # 定义类名到元素的映射
        # 确保每个类名唯一，并映射到正确的元素
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
            print("SMARTS 描述未生成。")
            return

        # 将 smarts_df 中的 Atom Index 转换为 nr（从1开始）
        self.smarts_df['nr'] = self.smarts_df['Atom Index'] + 1
        merged_df = pd.merge(self.atoms_df, self.smarts_df, on='nr', how='left')

        # 检查是否有未匹配的原子
        if merged_df['SMARTS'].isnull().any():
            print("警告：某些原子的SMARTS描述未找到。")

        # 识别唯一的SMARTS
        unique_smarts = merged_df['SMARTS'].unique()

        # 为每个唯一的SMARTS分配一个顺序命名的类型名称，如 opls_1, opls_2, ...
        smarts_to_unique_type = {smarts: f"opls_{i + 1}" for i, smarts in enumerate(unique_smarts)}

        # 创建一个新的列 'unique_type'，基于SMARTS映射
        merged_df['unique_type'] = merged_df['SMARTS'].map(smarts_to_unique_type)

        # 生成 <AtomTypes> 块
        atomtypes_block = "<AtomTypes>\n"
        missing_types = []
        for smarts, unique_type in smarts_to_unique_type.items():
            # 找到第一个对应的原子类型
            matching_rows = merged_df[merged_df['SMARTS'] == smarts]
            if matching_rows.empty:
                print(f"警告：SMARTS '{smarts}' 没有对应的原子类型。")
                missing_types.append(smarts)
                continue  # 跳过此SMARTS

            corresponding_type = matching_rows['type'].iloc[0]
            # 获取原子类型信息
            atomtype_info = self.atomtypes_df[self.atomtypes_df['name'] == corresponding_type]
            if not atomtype_info.empty:
                class_ = atomtype_info.iloc[0]['class']
                mass = atomtype_info.iloc[0]['mass']
                # 获取元素符号
                element = self.class_element_map.get(class_, 'C')  # 默认元素为C
                atomtypes_block += f'  <Type name="{unique_type}" class="{class_}" element="{element}" mass="{mass}" def="{smarts}"/>\n'
                self.unique_type_to_class[unique_type] = class_
            else:
                print(f"警告：类型 '{corresponding_type}' 在[ atomtypes ]部分未找到。使用默认值。")
                class_ = 'C'
                element = 'C'
                mass = 12.011
                atomtypes_block += f'  <Type name="{unique_type}" class="{class_}" element="{element}" mass="{mass}" def="{smarts}"/>\n'
                self.unique_type_to_class[unique_type] = class_
        atomtypes_block += "</AtomTypes>\n"
        self.xml_blocks["AtomTypes"] = atomtypes_block

        # 生成 <NonbondedForce> 块
        nonbonded_block = '<NonbondedForce coulomb14scale="0.5" lj14scale="0.5">\n'

        # 创建一个 DataFrame 按 unique_type 聚合 charge
        unique_nonbonded = merged_df.groupby('unique_type').agg({
            'charge': 'first',
            'type': 'first'  # 原始类型，用于查找 sigma and epsilon
        }).reset_index()

        # 提取 opls_x 中的数字部分并转换为整数，以便按数字顺序排序
        unique_nonbonded['type_num'] = unique_nonbonded['unique_type'].str.extract(r'opls_(\d+)').astype(int)

        # 按 type_num 升序排序
        unique_nonbonded = unique_nonbonded.sort_values('type_num')

        for _, row in unique_nonbonded.iterrows():
            unique_type = row['unique_type']
            charge = row['charge']
            original_type = row['type']
            # 从 atomtypes_df 获取 sigma 和 epsilon
            atomtype_info = self.atomtypes_df[self.atomtypes_df['name'] == original_type]
            if not atomtype_info.empty:
                sigma = atomtype_info.iloc[0]['sigma']
                epsilon = atomtype_info.iloc[0]['epsilon']
            else:
                print(f"警告：类型 '{original_type}' 在[ atomtypes ]部分未找到。使用默认值。")
                sigma = 0.35  # 默认值
                epsilon = 0.276144  # 默认值

            nonbonded_block += f'    <Atom type="{unique_type}" charge="{charge}" sigma="{sigma}" epsilon="{epsilon}"/>\n'
        nonbonded_block += "</NonbondedForce>\n"
        self.xml_blocks["NonbondedForce"] = nonbonded_block

        # 生成 <HarmonicBondForce> 块
        harmonic_bond_block = "<HarmonicBondForce>\n"
        unique_bonds = set()  # 初始化集合以存储已处理的键

        if self.bonds_df is not None:
            for _, bond in self.bonds_df.iterrows():
                ai = bond['ai']
                aj = bond['aj']
                c0 = bond['c0']  # bond length
                c1 = bond['c1']  # force constant

                # 获取 unique_type
                type_i = merged_df[merged_df['nr'] == ai]['unique_type'].iloc[0]
                type_j = merged_df[merged_df['nr'] == aj]['unique_type'].iloc[0]

                # 获取 class
                class_i = self.unique_type_to_class.get(type_i, 'C')
                class_j = self.unique_type_to_class.get(type_j, 'C')

                # 创建键的唯一标识符（排序后的类名元组，以避免重复）
                bond_tuple = (class_i, class_j, c0, c1)

                if bond_tuple not in unique_bonds:
                    unique_bonds.add(bond_tuple)
                    harmonic_bond_block += f'  <Bond class1="{class_i}" class2="{class_j}" length="{c0}" k="{c1}"/>\n'
        harmonic_bond_block += "</HarmonicBondForce>\n"
        self.xml_blocks["HarmonicBondForce"] = harmonic_bond_block

        # 生成 <HarmonicAngleForce> 块
        harmonic_angle_block = "<HarmonicAngleForce>\n"
        unique_angles = set()  # 初始化集合以存储已处理的角

        if self.angles_df is not None:
            for _, angle in self.angles_df.iterrows():
                ai = angle['ai']
                aj = angle['aj']
                ak = angle['ak']
                c0 = angle['c0']  # equilibrium angle in degrees
                c1 = angle['c1']  # force constant

                # 将角度从度转换为弧度
                c0_rad = self.degrees_to_radians(c0)

                # 获取 unique_type
                type_i = merged_df[merged_df['nr'] == ai]['unique_type'].iloc[0]
                type_j = merged_df[merged_df['nr'] == aj]['unique_type'].iloc[0]
                type_k = merged_df[merged_df['nr'] == ak]['unique_type'].iloc[0]

                # 获取 class
                class_i = self.unique_type_to_class.get(type_i, 'C')
                class_j = self.unique_type_to_class.get(type_j, 'C')
                class_k = self.unique_type_to_class.get(type_k, 'C')

                # 格式化弧度为小数点后8位
                c0_rad_formatted = f"{c0_rad:.8f}"

                # 创建角的唯一标识符（排序后的类名元组，以避免重复）
                angle_tuple = (class_i, class_j, class_k, c0_rad_formatted, c1)

                if angle_tuple not in unique_angles:
                    unique_angles.add(angle_tuple)
                    harmonic_angle_block += f'  <Angle class1="{class_i}" class2="{class_j}" class3="{class_k}" angle="{c0_rad_formatted}" k="{c1}"/>\n'
        harmonic_angle_block += "</HarmonicAngleForce>\n"
        self.xml_blocks["HarmonicAngleForce"] = harmonic_angle_block

        # 生成 <RBTorsionForce> 块
        rb_torsion_block = "<RBTorsionForce>\n"
        unique_propers = set()  # 初始化集合以存储已处理的二面角

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

                # 获取 unique_type
                type_i = merged_df[merged_df['nr'] == ai]['unique_type'].iloc[0]
                type_j = merged_df[merged_df['nr'] == aj]['unique_type'].iloc[0]
                type_k = merged_df[merged_df['nr'] == ak]['unique_type'].iloc[0]
                type_l = merged_df[merged_df['nr'] == al]['unique_type'].iloc[0]

                # 获取 class
                class_i = self.unique_type_to_class.get(type_i, 'C')
                class_j = self.unique_type_to_class.get(type_j, 'C')
                class_k = self.unique_type_to_class.get(type_k, 'C')
                class_l = self.unique_type_to_class.get(type_l, 'C')

                # 创建二面角的唯一标识符（排序后的类名元组，以避免重复）
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
            print(f"Generated XML file have been written to '{self.xml_filename}'.")
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


