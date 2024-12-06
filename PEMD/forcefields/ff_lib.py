import os
import sys
import io
import shutil
import parmed as pmd
from foyer import Forcefield
from PEMD.simulation import sim_lib
from openbabel import openbabel as ob
from PEMD.model import model_lib, MD_lib
import importlib.resources as pkg_resources
from pysimm import system, lmps, forcefield
from PEMD.forcefields.xml import XMLGenerator
from PEMD.forcefields.ligpargen import PEMDLigpargen
from PEMD.model.build import gen_poly_smiles, gen_poly_3D


def get_gaff2(
        work_dir,
        pdb_file,
        atom_typing
):

    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "cml")
    if os.path.exists(pdb_file):
        obmol = ob.OBMol()
        obConversion.ReadFile(obmol, pdb_file)
    else:
        print(f"{pdb_file} not found in {work_dir}")

    file_prefix, file_extension = os.path.splitext(pdb_file)
    cml_file = os.path.join(work_dir, f"{file_prefix}.cml")
    obConversion.WriteFile(obmol, cml_file)
    data_fname = os.path.join(work_dir, f"{file_prefix}_gaff2.lmp")

    try:
        temp_stdout = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = temp_stdout
        s = system.read_cml(cml_file)
        f = forcefield.Gaff2()

        if atom_typing == 'pysimm':
            for b in s.bonds:
                if b.a.bonds.count == 3 and b.b.bonds.count == 3:
                    b.order = 4
            s.apply_forcefield(f, charges='gasteiger')
        elif atom_typing == 'antechamber':
            obConversion.SetOutFormat("mol2")
            obConversion.WriteFile(obmol, f'{work_dir}/{file_prefix}.mol2')
            print("Antechamber working on {}".format(f'{work_dir}/{file_prefix}.mol2'))
            MD_lib.get_type_from_antechamber(s, f'{file_prefix}.mol2', 'gaff2', f)
            s.pair_style = 'lj'
            s.apply_forcefield(f, charges='gasteiger', skip_ptypes=True)
        else:
            print('Invalid atom typing option, please select pysimm or antechamber.')

        s.write_lammps(data_fname)
        sys.stdout = original_stdout
        print(sys.stdout)

        print("GAFF2 parameter file generated successfully.\n")
    except Exception as e:
        print(f'problem reading {file_prefix + ".cml"} for Pysimm: {str(e)}')

def get_xml_ligpargen(work_dir, name, resname, repeating_unit, chg, chg_model, ):

    smiles = gen_poly_smiles(
        name,
        repeating_unit,
        length=4,
        leftcap='',
        rightcap='',
    )

    ligpargen_dir = os.path.join(work_dir, f'ligpargen_{name}')
    os.makedirs(ligpargen_dir, exist_ok=True)

    xyz_filename = f'{name}.xyz'
    model_lib.smiles_to_xyz(smiles, os.path.join(work_dir,  xyz_filename))

    PEMDLigpargen(
        ligpargen_dir,
        name,
        resname,
        chg,
        chg_model,
        filename = xyz_filename,
    ).run_local()

    gmx_itp_file = os.path.join(ligpargen_dir, f"{name}.gmx.itp")
    xml_filename = os.path.join(work_dir, f"{name}.xml")
    generator = XMLGenerator(
        gmx_itp_file,
        smiles,
        xml_filename
    )
    generator.run()

def get_oplsaa_xml(
        work_dir,
        poly_name,
        poly_resname,
        repeating_unit,
        length_long,
        leftcap,
        rightcap,
        xyz_file,
        xml = 'ligpargen',  # ligpargen or database
):

    xyz_filepath = os.path.join(work_dir, xyz_file)
    MD_dir = os.path.join(work_dir, 'MD_dir')
    os.makedirs(MD_dir, exist_ok=True)

    smiles = gen_poly_smiles(
        poly_name,
        repeating_unit,
        length_long,
        leftcap,
        rightcap,
    )

    pdb_file = gen_poly_3D(
        work_dir,
        poly_name,
        poly_resname,
        length_long,
        smiles,
    )

    pdb_filepath = os.path.join(MD_dir, f"{poly_name}.pdb")
    model_lib.convert_xyz_to_pdb(xyz_filepath, pdb_filepath, poly_name, poly_resname)

    untyped_str = pmd.load_file(pdb_file, structure=True)
    if xml == 'database':
        with pkg_resources.path("PEMD.forcefields", "oplsaa.xml") as oplsaa_path:
            oplsaa = Forcefield(forcefield_files = str(oplsaa_path))
        typed_str = oplsaa.apply(untyped_str, verbose=True, use_residue_map=True)
    else:
        xml_filename = os.path.join(work_dir, f"{poly_name}.xml")
        oplsaa = Forcefield(forcefield_files = xml_filename)
        typed_str = oplsaa.apply(untyped_str, verbose=True, use_residue_map=True)

    top_filename = os.path.join(MD_dir, f"{poly_name}.top")
    gro_filename = os.path.join(MD_dir, f"{poly_name}.gro")
    typed_str.save(top_filename)
    typed_str.save(gro_filename)

    nonbonditp_filename = os.path.join(MD_dir, f'{poly_name}_nonbonded.itp')
    bonditp_filename = os.path.join(MD_dir, f'{poly_name}_bonded.itp')

    model_lib.extract_from_top(top_filename, nonbonditp_filename, nonbonded=True, bonded=False)
    model_lib.extract_from_top(top_filename, bonditp_filename, nonbonded=False, bonded=True)

    os.remove(top_filename)
    os.remove(gro_filename)

    return f'{poly_name}_bonded.itp'


def get_oplsaa_ligpargen(work_dir, name, resname, chg, chg_model, smiles, ):

    ligpargen_dir = os.path.join(work_dir, f'ligpargen_{name}')
    os.makedirs(ligpargen_dir, exist_ok=True)

    MD_dir = os.path.join(work_dir, 'MD_dir')
    os.makedirs(MD_dir, exist_ok=True)

    xyz_filename = f'{name}.xyz'
    model_lib.smiles_to_xyz(smiles, os.path.join(work_dir,  xyz_filename))

    PEMDLigpargen(
        ligpargen_dir,
        name,
        resname,
        chg,
        chg_model,
        filename = xyz_filename,
    ).run_local()

    # PEMDLigpargen(
    #     ligpargen_dir,
    #     name,
    #     resname,
    #     chg,
    #     chg_model,
    #     smiles,
    # ).run_local()

    nonbonditp_filename = os.path.join(MD_dir, f'{name}_nonbonded.itp')
    bonditp_filename = os.path.join(MD_dir, f'{name}_bonded.itp')
    pdb_filename = os.path.join(MD_dir, f'{name}.pdb')

    top_filename = os.path.join(ligpargen_dir, f"{name}.gmx.itp")
    gro_filename = os.path.join(ligpargen_dir, f'{name}.gmx.gro')
    model_lib.extract_from_top(top_filename, nonbonditp_filename, nonbonded=True, bonded=False)
    model_lib.extract_from_top(top_filename, bonditp_filename, nonbonded=False, bonded=True)
    model_lib.convert_gro_to_pdb(gro_filename, pdb_filename,)

    return f'{name}_bonded.itp'

def gen_ff_from_data(work_dir, compound_name, corr_factor, target_sum_chg):

    MD_dir = os.path.join(work_dir, 'MD_dir')
    os.makedirs(MD_dir, exist_ok=True)

    files_to_copy = [
        f"pdb/{compound_name}.pdb",
        f"itp/{compound_name}_bonded.itp",
        f"itp/{compound_name}_nonbonded.itp"
    ]
    for file_path in files_to_copy:
        try:
            resource_dir = pkg_resources.files('PEMD.forcefields')
            resource_path = resource_dir.joinpath(file_path)
            shutil.copy(str(resource_path), MD_dir)
            print(f"Copied {file_path} to {MD_dir} successfully.")
        except Exception as e:
            print(f"Failed to copy {file_path}: {e}")

    filename = f"{compound_name}_bonded.itp"
    sim_lib.scale_chg_itp(MD_dir, filename, corr_factor, target_sum_chg)
    print(f"scale charge successfully.")






