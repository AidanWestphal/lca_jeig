#!/bin/bash
#
# Run jeig algorithm with universal config for all species
#
# Uses auto-computed parameters based on dataset characteristics.
# Results are saved with _universal suffix to preserve old results.
#

BASE_CONFIG=configs/config_jeig.yaml

for species in beluga forestelephants GZCD giraffe lion plainszebra whaleshark; do
    echo "=== ${species} ==="
    python3 run_clustering_with_save.py \
        --base_config ${BASE_CONFIG} \
        --config configs/${species}/config_${species}_data.yaml
    echo ""
done

echo "Done!"
