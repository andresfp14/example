import multiprocessing
import subprocess
import threading
import time
import random
from pathlib import Path
from tqdm import tqdm
from itertools import product
import sys
from .loggers import create_logger

def stream_to_file(pipe, file_object):
    """
    Reads lines continuously from a given pipe (stdout or stderr) and writes them to a file object in real-time. 
    This function is designed to be run in a separate thread, allowing asynchronous logging of subprocess output 
    without blocking the main execution thread.

    Args:
        pipe (io.TextIOWrapper): The pipe object from which to read the output. This is typically obtained 
        from the stdout or stderr of a subprocess.Popen object, configured to be in text mode.
        
        file_object (file): An open file object where the pipe's content will be written. The file must be 
        opened in a mode that supports writing (e.g., 'w', 'a').

    This function does not return a value but writes output directly to the provided file_object until the pipe
    is closed. It's important to ensure that the file_object and pipe are correctly closed by the caller to 
    avoid resource leaks.

    Example usage:
        # Assuming process is a subprocess.Popen object
        with open('log.txt', 'w') as f:
            stdout_thread = threading.Thread(target=stream_to_file, args=(process.stdout, f))
            stdout_thread.start()
            # Other operations can be performed here while the thread captures the output in the background.
            stdout_thread.join()  # Ensure the logging thread has completed before closing the file.
    """
    for line in iter(pipe.readline, ''):
        file_object.write(line)
    pipe.close()

def p_run(func_name, log_name=None, base_file='run.py', **kwargs):
    """
    Executes a specified Python function as a subprocess from a given script, capturing its standard output (stdout) 
    and standard error (stderr) to a log file in real-time. Allows for specifying a custom script file from which 
    the function will be called, instead of the default 'run.py'.

    Args:
        func_name (str): The name of the Python function to execute. This function should be accessible 
        within the script specified by the 'base_file' parameter.
        
        log_name (str, optional): Custom name for the log file. If provided, the log file will be named 
        according to this parameter. If not provided, a unique name is generated based on the function 
        name, current timestamp, and a random number.
        
        base_file (str, optional): The Python script from which the function will be executed. Defaults 
        to 'run.py'. This script must be located in a directory accessible by the Python interpreter 
        executing the `p_run` function.
        
        **kwargs: Arbitrary keyword arguments that are passed to the function being executed. These arguments 
        are converted to command-line arguments in the format '--key=value'.

    The function creates a log file in the './data/logs/' directory, ensuring the directory exists. It then 
    initiates the subprocess with stdout and stderr piped. Two separate threads are spawned to asynchronously 
    capture the output from these pipes to the log file.

    This function does not return any value but logs the execution status. In case of a subprocess failure 
    (non-zero return code), an informative message is logged using a custom logger.

    Example usage:
        p_run('data_processing_function', arg1='value1', arg2='value2', base_file='alternative_script.py')
        # This would execute the 'data_processing_function' from 'alternative_script.py' with the specified arguments and capture its output to a log file.
    """
    # Initialize logger
    logger = create_logger("p_run")

    # Convert keyword arguments to command-line arguments
    args = [f"--{key}={value}" for key, value in kwargs.items()]
    
    # Construct the command to be executed
    cmd = [sys.executable, base_file, func_name] + args

    # Ensure the logs directory exists
    Path("./data/logs/").mkdir(parents=True, exist_ok=True)
    
    # Generate log file name
    logfile = f"./data/logs/{log_name}.log" if log_name else f"./data/logs/log_{func_name}_{int(time.time())}_{random.randint(0,999):03d}.log"

    # Open the log file
    with open(logfile, "w") as output_file:
        # Start the subprocess
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Create threads to capture stdout and stderr
        stdout_thread = threading.Thread(target=stream_to_file, args=(process.stdout, output_file))
        stderr_thread = threading.Thread(target=stream_to_file, args=(process.stderr, output_file))

        # Start the threads
        stdout_thread.start()
        stderr_thread.start()

        # Wait for the output capture to complete
        stdout_thread.join()
        stderr_thread.join()

        # Ensure the subprocess has finished
        process.wait()

        # Log if the process did not complete successfully
        if process.returncode != 0:
            logger.info(f"Failed: {' '.join(cmd)}")

def parallelize_function(func, parallel_args_list, num_processes=None, base_file='run.py'):
    """
    Runs a function in parallel across multiple processes, using a specified script file. This function is designed 
    to parallelize execution of tasks that are encapsulated in Python functions, with arguments specified for each 
    task. It captures the output of these functions to log files in real-time.

    Args:
        func (function): The function to be parallelized. Note that this function should be accessible within 
        the script specified by the 'base_file' parameter.
        
        parallel_args_list (list): A list of tuples, where each tuple contains a tuple of positional arguments 
        and a dictionary of keyword arguments for each invocation of the parallelized function. The keyword 
        arguments can include a 'log_name' key to specify custom log file names for each process.
        
        num_processes (int, optional): The number of processes to use for parallel execution. If None, the 
        function defaults to using the number of available CPU cores on the system.
        
        base_file (str, optional): The Python script from which the function will be executed in each subprocess. 
        Defaults to 'run.py'. This allows for specifying different scripts for different tasks.

    Returns:
        list: A list of return values from the function for each set of arguments. Note that in the context of 
        this function, the primary purpose is to execute parallel tasks rather than collecting return values, 
        as the output is directed to log files.

    Example usage:
        def my_func(arg1, arg2):
            # Function body here

        args_list = [((arg1_value, arg2_value), {'log_name': 'custom_log_for_this_invocation'})]
        results = parallelize_function(my_func, args_list, num_processes=4, base_file='my_script.py')
        # This will run 'my_func' in parallel, using 'my_script.py', with specified arguments and log names.
    """
    results = []

    # Determine the number of processes
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    # Initialize multiprocessing pool and tqdm progress bar
    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(parallel_args_list), desc="Processing", unit="task") as pbar:
        async_results = []

        # Submit tasks to the pool
        for arg_set in parallel_args_list:
            args, kwargs = arg_set
            kwargs['base_file'] = base_file  # Add the base_file to kwargs for each task
            result = pool.apply_async(func, args=args, kwds=kwargs, callback=lambda _: pbar.update(1))
            async_results.append(result)

        # Collect results
        for async_result in async_results:
            results.append(async_result.get())

    return results

def pex(func_name, *args, num_processes=None, base_file=None, **kwargs):
    """
    Executes a specified function in parallel across multiple processes, with the ability to expand and 
    combine list-type keyword arguments into multiple sets of arguments for the function. This allows for 
    comprehensive and efficient experimentation or task execution with varied parameters.

    Args:
        func_name (str): The name of the function to execute in parallel. This function must be accessible 
        within the script specified by the 'base_file' parameter.

        *args: Positional arguments to be passed directly to the function. These are not expanded or varied 
        and are passed as-is to every invocation of the function.

        num_processes (int, optional): The number of processes to use for the parallel execution. Defaults to 
        the number of available CPU cores if None.

        base_file (str, optional): The Python script from which the function will be executed. If None, the 
        script that is currently being executed (where this function is called) is used. This allows for 
        different scripts to be used for different parallel execution tasks.

        **kwargs: Keyword arguments for the function. If a keyword argument is a list, this function will 
        generate combinations of these lists, running the target function for each combination alongside any 
        non-list arguments.

    Returns:
        list: A list of results from each process. Note that in this context, the primary purpose is to 
        execute parallel tasks, and the actual return values might be less relevant if outputs are captured 
        in log files or external systems.

    Example usage:
        pex('process_data', data_id=42, filters=['filter1', 'filter2'], num_processes=4)
        # This would parallelize 'process_data' function calls over the combinations of 'filters' with 
        # 'data_id' as a constant argument, using the current script as the base file for execution.
    """
    # Set base_file to the current script if not specified
    if base_file is None:
        base_file = sys.argv[0]

    # Separate list and non-list kwargs
    list_args = {k: v for k, v in kwargs.items() if isinstance(v, list)}
    non_list_args = {k: v for k, v in kwargs.items() if k not in list_args}

    # Generate combinations of list arguments
    combinations = [dict(zip(list_args, prod)) for prod in product(*list_args.values())]

    # Prepare arguments for parallel execution
    parallel_args_list = []
    for comb in combinations:
        combined_args = {**non_list_args, **comb, 'func_name': func_name, 'base_file': base_file}
        parallel_args_list.append((args, combined_args))

    # Execute in parallel
    return parallelize_function(p_run, parallel_args_list, num_processes=num_processes, base_file=base_file)

