#!/bin/bash

datasets=("11" "12" "21" "22")

for temp_num in "${datasets[@]}"
do
    for targ_num in "${datasets[@]}"
    do
        if [ "$temp_num" != "$targ_num" ]; then
            echo "Running experiment for template dataset city${temp_num} and target dataset city${targ_num}"
            python3 dpmatching.py << EOF
$temp_num
$targ_num
EOF
        fi
    done
done

echo "All experiments completed."
