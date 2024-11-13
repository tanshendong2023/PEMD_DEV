#!/bin/bash

# Enable extended globbing to use advanced pattern matching
shopt -s extglob

# Define the directory containing input files and where outputs will be stored
gaussian_dir="${1:-./conformer_search/gaussian_work}"

# Loop over all conf_*.gjf files in the directory where * is any digit
for input_file in "${gaussian_dir}"/conf_*([0-9]).gjf; do
    # Extract the base name of the file (e.g., conf_0)
    base_name=$(basename "$input_file" .gjf)

    # Define the output file for Gaussian calculation
    output_file="${gaussian_dir}/${base_name}.log"

    # Run Gaussian and direct output to log file
    g16 "$input_file" > "$output_file"

    # Assuming out2gjf converts Gaussian output to gjf and gjf2xyz converts gjf to xyz
#    # Generate gjf from Gaussian output
#    gjf_file="${gaussian_dir}/${base_name}.gjf"
#    log2gjf "$output_file" > "$gjf_file"
#
#    # Generate xyz from gjf
#    xyz_file="${gaussian_dir}/${base_name}.xyz"
#    gjf2xyz "$gjf_file" > "$xyz_file"
done

## Define the filename for the merged XYZ file
#merged_filename="merged_gaussian.xyz"
#
## Remove the output file if it already exists to avoid appending to old data
#rm -f "$merged_filename"
#
## Merge all *xtbopt.xyz files into the merged xyz file, ensuring files include digits only
#cat "${gaussian_dir}"/conf_*([0-9]).xyz > "$merged_filename"

# echo "The generated conformers have been merged and saved to ${merged_filename}"
