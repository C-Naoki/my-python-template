#!/bin/sh

model=proposal
input_dir=data_name
root_out_dir=out/

COMMAND="poetry run python src/main.py --multirun \
  model=${model} \
  io.input_dir=${input_dir} \
  io.root_out_dir=${root_out_dir} \
  save=True \
  verbose=True"

bash bin/run_wrapper.sh "$@" "$COMMAND"
