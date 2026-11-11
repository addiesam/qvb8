"""Loaders and launchers: everything that actually starts a model. INFERENCE
only -- calibration/pack drivers live under each codec's own calibration/.

    run_inference.py            848  the loader every live QVB8 run goes
                                    through. gen_core.load_model calls its
                                    load_qv_model.
    run_inference_cpu.py        300  CPU path
    placement.py                     pipeline placement over N cards (--pipeline), the multi-GPU path
"""
