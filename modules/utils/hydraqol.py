import os
import sys
import json
import logging
import datetime
import traceback
import inspect
from functools import wraps
from pathlib import Path
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
import shutil
import time

################################################################################################
# Custom run decorator
################################################################################################
def run_decorator(func):
    @wraps(func)
    def wrapper(cfg, *args, **kwargs):
        logger = logging.getLogger("wrapper")
        start_time = datetime.datetime.now()
        
        # Get the script name from the function's definition location
        script_path = inspect.getfile(func)
        script_name = os.path.basename(script_path)
        save_dir = Path(cfg.save_dir)
        run_info_path = save_dir / "run_info.json"
        config_path = save_dir / "config.yaml"
        hydra_config_path = save_dir / "hydra_config.yaml"
        procid = os.environ.get("SLURM_PROCID", "0")

        # Get the command and arguments that were executed
        command = f"{sys.executable} {' '.join(sys.argv)}"
        relative_command = f"{sys.executable} runs/{script_name} --config-path=../{save_dir} --config-name=config.yaml"

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
        logger.info("#" * 75)
        logger.info(f"Executed command: {command}")
        logger.info(f"Relative command: {relative_command}")
        logger.info(f"script_name: {script_name}")
        logger.info(f"state: {run_status}")
        logger.info(f"mode: {mode}")
        logger.info(f"save_dir: {save_dir}")
        logger.info(f"proc ID: {procid}")
        logger.info("#" * 30 + " TASK START " + "#" * 33)

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
                "result": None,
                "command": command
            }
            return run_info

        if mode == "base":
            # If run is completed, skip
            if run_status == "completed":
                logger.info(f"[BASE] Run already completed in {save_dir}. Skipping...")
                return None
        
        if mode == "clean":
            # If run is completed, skip
            if run_status == "completed":
                logger.info(f"[CLEAN] Run already completed in {save_dir}. Skipping...")
                return None
            
            # Clean mode: delete existing run and executing
            if save_dir_exists:
                logger.info(f"[CLEAN] Deleting existing run in {save_dir}")
                shutil.rmtree(save_dir)

        if mode == "force":
            # Force mode: delete existing run and start fresh
            if save_dir_exists:
                logger.info(f"[FORCE] Deleting existing run in {save_dir}")
                shutil.rmtree(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)

        

        # --------------------------------------------
        # Step 3: Save Configuration
        # --------------------------------------------
        save_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, config_path)
        with open(config_path, 'r+') as f: 
            content = f.read()
            f.seek(0)
            f.write('# @package _global_\n' + content)
        #OmegaConf.save(HydraConfig.get(), hydra_config_path)

        # --------------------------------------------
        # Step 4: Execute Function with Retry Logic
        # --------------------------------------------
        max_retries = getattr(cfg, "max_retries", 0)
        retry_delay = getattr(cfg, "retry_delay", 5)  # seconds
        retry_count = 0
        last_error = None
        error_files = []

        while retry_count <= max_retries:
            try:
                result = func(cfg, *args, **kwargs)
                break
            except Exception as e:
                last_error = e
                retry_count += 1
                
                # Save error information for this attempt
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                error_file = save_dir / f"error_{timestamp}.txt"
                error_files.append(str(error_file))
                
                with open(error_file, "w") as f:
                    f.write(f"Attempt {retry_count} failed at {timestamp}\n")
                    f.write(f"Error type: {type(e).__name__}\n")
                    f.write(f"Error message: {str(e)}\n")
                    f.write("\nFull traceback:\n")
                    f.write(traceback.format_exc())
                
                if retry_count <= max_retries:
                    logger.warning(f"Attempt {retry_count} failed: {str(e)}")
                    logger.info(f"Error details saved to {error_file}")
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"All {max_retries} attempts failed. Last error: {str(e)}")
                    raise last_error

        # --------------------------------------------
        # Step 5: Save Run Information
        # --------------------------------------------
        run_info = {
            "state": "completed",
            "mode": mode,
            "func_name": func.__name__,
            "executed_file": script_name,
            "save_dir": str(cfg.save_dir),
            "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
            "end_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_time_seconds": (datetime.datetime.now() - start_time).total_seconds(),
            "result": str(result) if result is not None else None,
            "retry_count": retry_count,
            "error_files": error_files if error_files else None,
            "command": command
        }
        with open(run_info_path, "w") as f:
            json.dump(run_info, f, indent=4)

        # Log finished message
        logger.info("#" * 30 + " TASK FINISHED " + "#" * 30)
        logger.info(f"[FINISHED] {func.__name__} completed successfully")
        logger.info(f"Total time: {run_info['total_time_seconds']:.2f} seconds")
        logger.info(f"Save directory: {save_dir}")
        logger.info("#" * 75)

        return result

    return wrapper 

################################################################################################
# Custom OmegaConf resolvers
################################################################################################
def default(val, default=1):
    # Here, you could add more logic to determine if val is "missing"
    return default if val is None else val
OmegaConf.register_new_resolver("default", default)

def math(operator, *args):
    """
    Custom resolver for mathematical operations.
    
    Args:
        operator (str): The mathematical operator to use ('+', '-', '*', '/', '**', etc.)
        *args: The operands for the mathematical operation
        
    Returns:
        The result of applying the operator to the operands
    """
    if not args:
        raise ValueError("At least one operand is required for math operations")
    
    # Convert all arguments to float for consistent handling
    operands = [float(arg) for arg in args]
    
    # Apply the operator
    if operator == '+':
        return sum(operands)
    elif operator == '-':
        result = operands[0]
        for operand in operands[1:]:
            result -= operand
        return result
    elif operator == '*':
        result = 1
        for operand in operands:
            result *= operand
        return result
    elif operator == '/':
        result = operands[0]
        for operand in operands[1:]:
            if operand == 0:
                raise ValueError("Division by zero")
            result /= operand
        return result
    elif operator == '**':
        if len(operands) != 2:
            raise ValueError("Power operation requires exactly 2 operands")
        return operands[0] ** operands[1]
    elif operator == '%':
        if len(operands) != 2:
            raise ValueError("Modulo operation requires exactly 2 operands")
        if operands[1] == 0:
            raise ValueError("Modulo by zero")
        return operands[0] % operands[1]
    elif operator == '//':
        if len(operands) != 2:
            raise ValueError("Floor division operation requires exactly 2 operands")
        if operands[1] == 0:
            raise ValueError("Division by zero")
        return operands[0] // operands[1]
    else:
        raise ValueError(f"Unsupported operator: {operator}")

OmegaConf.register_new_resolver("math", math)


