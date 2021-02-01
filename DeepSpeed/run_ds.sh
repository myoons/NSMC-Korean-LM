#!/bin/bash
deepspeed --num_gpus=2 \
    ds_KoELECTRA.py -e 5 \
    --deepspeed --deepspeed_config ds_config.json
