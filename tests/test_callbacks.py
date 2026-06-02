import sys
sys.path.insert(0, "../")
from run_opt import run_optimization_api
from pytopo3d.cli.parser import parse_args
import threading


def main():
    args = parse_args()
    stop_event = threading.Event()

    counter = 0
    def callback(density):
        print(f"max density: {density.max()}, min density: {density.min()}")
        nonlocal counter
        counter += 1
        #test the stop event after 10 iterations
        if counter >= 10: 
            stop_event.set()
    
    results = run_optimization_api(args, callback=callback, stop_event=stop_event)
    print("Optimization completed. Results:", results)

    
if __name__ == "__main__":
    sys.exit(main())
