for cfg_file in train_configs/*.cfg; do
    echo "Running $cfg_file"
    python3 run_train.py $(cat "$cfg_file")
    echo "Finished $cfg_file"
done