#!/bin/bash

# Enable extended globbing to use advanced pattern matching
shopt -s extglob

# Define the directory containing input files and where outputs will be stored
xtb_dir="./conformer_search/xtb_work"

# Loop over all conf_*.xyz files in the directory where * is any digit
for input_file in "${xtb_dir}"/conf_*([0-9]).xyz; do
    # Extract the base name of the file (e.g., conf_0)
    base_name=$(basename "$input_file" .xyz)
    namespace="${xtb_dir}/${base_name}"

    # Run xtb with the specified parameters
    xtb "$input_file" --opt --chrg="$1" --uhf="$2" --gfn "$3" --ceasefiles --namespace "$namespace"
done

# Define the output filename for the merged file
output_filename="merged_xtb.xyz"

# Remove the output file if it already exists to avoid appending to old data
rm -f "$output_filename"

# Merge all *xtbopt.xyz files into the merged_xtb.xyz file, ensuring files include digits only
cat "${xtb_dir}"/conf_*([0-9]).xtbopt.xyz > "$output_filename"

echo "The optimization conformers based on xtb saved to ${output_filename}"

