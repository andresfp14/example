# Hydra quality of life module.hydraqol.py
import os
import sys
import json
import logging
import datetime
import traceback
from functools import wraps
from pathlib import Path
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
import shutil

def run_decorator(func):
    @wraps(func)
    def wrapper(cfg, *args, **kwargs):
        logger = logging.getLogger("wrapper")
        start_time = datetime.datetime.now()
        script_name = os.path.basename(sys.argv[0])
        save_dir = Path(cfg.save_dir)
        run_info_path = save_dir / "run_info.json"
        config_path = save_dir / "config.yaml"
        hydra_config_path = save_dir / "hydra_config.yaml"
        procid = os.environ.get("SLURM_PROCID", "0")

        # Determine mode from cfg; defaults to 'base' if not specified
        mode = getattr(cfg, "mode", "base")
        valid_modes = ["base", "force", "clean", "check"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}.")

        # --------------------------------------------
        # Step 1: Gather current run status information
        # --------------------------------------------
        save_dir_exists = save_dir.exists()
        run_info_exists = run_info_path.exists()
        config_exists = config_path.exists()

        if not save_dir_exists:
            run_status = "not run"
        else:
            if run_info_exists:
                run_status = "completed"
            elif config_exists:
                run_status = "incomplete"
            else:
                run_status = "not run (only folder)"
            
        # log status, mode, save_dir, and procid
        logger.info(f"state: {run_status}")
        logger.info(f"mode: {mode}")
        logger.info(f"save_dir: {save_dir}")
        logger.info(f"proc ID: {procid}")

        # --------------------------------------------
        # Step 2: Mode Handling
        # --------------------------------------------
        if mode == "check":
            # Check mode: only check status and return
            run_info = {
                "state": run_status,
                "mode": mode,
                "func_name": func.__name__,
                "executed_file": script_name,
                "save_dir": str(cfg.save_dir),
                "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
                "end_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "total_time_seconds": (datetime.datetime.now() - start_time).total_seconds(),
                "result": None
            }
            return run_info

        if mode == "base":
            # If run is completed, skip
            if run_status == "completed":
                logger.info(f"[BASE] Run already completed in {save_dir}. Skipping...")
                run_info = {
                    "state": "skipped",
                    "mode": mode,
                    "func_name": func.__name__,
                    "executed_file": script_name,
                    "save_dir": str(cfg.save_dir),
                    "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "end_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "total_time_seconds": (datetime.datetime.now() - start_time).total_seconds(),
                    "result": None
                }
                logger.info(f"[TASK FINISHED] "+"#"*50)
                for key, value in run_info.items():
                    logger.info(f"{key}: {value}")
                logger.info(f"[TASK FINISHED] "+"#"*50)
                return run_info
            # Otherwise, proceed

        elif mode == "force":
            # Force mode: always remove and re-run
            if save_dir_exists:
                logger.info(f"[FORCE] Removing existing directory {save_dir} and rerunning...")
                shutil.rmtree(save_dir, ignore_errors=True)

        elif mode == "clean":
            # Clean mode: if completed, skip like base; if not completed, remove and re-run
            if run_status == "completed":
                logger.info(f"[CLEAN] Run already completed in {save_dir}. Skipping...")
                run_info = {
                    "state": "skipped",
                    "mode": mode,
                    "func_name": func.__name__,
                    "executed_file": script_name,
                    "save_dir": str(cfg.save_dir),
                    "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "end_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "total_time_seconds": (datetime.datetime.now() - start_time).total_seconds(),
                    "result": None
                }
                logger.info(f"[TASK FINISHED] "+"#"*50)
                for key, value in run_info.items():
                    logger.info(f"{key}: {value}")
                logger.info(f"[TASK FINISHED] "+"#"*50)
                return run_info
            else:
                if save_dir_exists:
                    logger.info(f"[CLEAN] Run not completed in {save_dir}. Removing directory and rerunning...")
                    shutil.rmtree(save_dir, ignore_errors=True)

        # --------------------------------------------
        # Step 3: Setup save_dir and config
        # --------------------------------------------
        if int(procid) == 0:
            save_dir.mkdir(parents=True, exist_ok=True)
            # Save the config
            with open(config_path, "w") as f:
                OmegaConf.save(config=cfg, f=f)
            # Save the Hydra config
            #with open(hydra_config_path, "w") as f:
            #    OmegaConf.save(config=HydraConfig.get(), f=f)
            

        # --------------------------------------------
        # Step 4: Run the function
        # --------------------------------------------
        try:
            # Run the function
            result = func(cfg, *args, **kwargs)
            # On success, record new run_info
            run_info = {
                "status": "success",
                "mode": mode,
                "func_name": func.__name__,
                "executed_file": script_name,
                "save_dir": str(cfg.save_dir),
                "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
                "end_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "total_time_seconds": (datetime.datetime.now() - start_time).total_seconds(),
                "result": result
            }
            if int(procid) == 0:
                with open(run_info_path, "w") as f:
                    json.dump(run_info, f, indent=4)
            logger.info(f"[TASK FINISHED] "+"#"*50)
            for key, value in run_info.items():
                logger.info(f"{key}: {value}")
            logger.info(f"[TASK FINISHED] "+"#"*50)
            return run_info

        except Exception as e:
            # --------------------------------------------
            # Step 5: In case of error, save traceback
            # --------------------------------------------
            error_file = save_dir / f"run_error_{procid}.txt"
            with open(error_file, "w") as f:
                traceback_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
                f.write(traceback_str)
            raise e

    return wrapper
