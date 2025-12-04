#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
from math import ceil

models = {
    "A100": "a100-pcie-40gb",
    "4090": "rtx_4090",
    "3090": "rtx_3090",
    "1080": "rtx_1080",
    "2080Ti": "rtx_2080_ti",
    "TITAN": "titan_rtx",
    "V100": "v100",
}

def main(args):
    for i in range(args.chain):
        if args.interactive:
            cmd = "srun --pty"
            cmd += f" --account=a144"
            if args.container_writable:
                cmd += " --container-writable"
            if args.environment is not None:
                cmd += f" --environment={args.environment}"
            if args.partition is not None:
                cmd += f" -p {args.partition}"
            else:
                cmd += " -p debug"  # default for interactive
        else:
            cmd = "sbatch"
            # Handle time format - if it's just HH:MM, add :00, otherwise use as-is
            time_str = args.time if ":" in args.time and args.time.count(":") == 2 else f"{args.time}:00"
            cmd += f" --time={time_str}"
            cmd += f" --account=a144"
            cmd += f" --nodes=1"
            cmd += f" --ntasks-per-node={max(1, args.gpus)}"
            cmd += f" --cpus-per-task={int(args.cpus/max(1, args.gpus))}"
            if args.partition is not None:
                cmd += f" --partition={args.partition}"
            else:
                cmd += " --partition=normal"
        jobname = args.name + str(i) if (args.chain > 1) else args.name
        cmd += f" -J {jobname}"
        if not args.interactive:
            if i > 0:
                cmd += f" -d afterany:{job_id}"
            cmd += f" --mem={args.mem}G"
            if args.scratch is not None:
                cmd += f" --tmp={args.scratch}"
            if args.environment is not None:
                cmd += f" --environment={args.environment}"
        if args.gmod is not None:
            cmd += f" --gpus={models[args.gmod]}:{args.gpus}"
        elif args.gmem is not None and not args.interactive:
            cmd += f" --gres=gpumem:{args.gmem}m --gpus={args.gpus}"
        if args.mail is not None and not args.interactive:
            cmd += f" --mail-type={args.mail}"

        # script = f"#!/bin/bash\nsource ~/.emimic_bashrc; "
        # script += f"python EgoVerse/egomimic/trainHydra.py "
        script = " ".join(args.command)
        print(script)

        if args.gpus > 1:
            script += " --distributed"
        if i > 0:
            script += " --restore"

        if args.interactive:
            cmd += f" {script}"
        else:
            cmd += f' --wrap="{script}"'
        print("\n" + cmd + "\n")

        OUT = sys.stdout if args.interactive else subprocess.PIPE
        ERR = sys.stderr if args.interactive else subprocess.STDOUT
        try:
            ret = subprocess.run(
                cmd, shell=True, check=True, stdout=OUT, stderr=ERR, text=True
            )
            out_s = ret.stdout
        except subprocess.CalledProcessError as e:
            out_s = e.output
            print(e)
        except Exception as e:
            print(e)
            raise e

        if not args.interactive:
            print(out_s)
            (job_id,) = re.findall(r"Submitted batch job (\d+)", out_s)
            print(job_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpus", "-c", type=int, default=4)
    parser.add_argument(
        "--mem", "-m", type=int, default=128, help="Total memory in GB"
    )
    parser.add_argument("--scratch", type=int, default=None, help="minimum Scratch space in MB")
    parser.add_argument("--time", type=str, default="24:00:00", help="time limit (HH:MM:SS format)")
    parser.add_argument(
        "--warn",
        type=str,
        default="-wt 5 -wa INT",
        help="default: send an interrupt 5 minutes before the end",
    )
    parser.add_argument("--gpus", "-g", type=int, default=1)
    parser.add_argument(
        "--gmod", type=str, choices=list(models), help="GPU model", default=None
    )
    parser.add_argument("--gmem", type=int, default=10_240, help="GPU memory")
    parser.add_argument("--mail", type=str, default=None)
    parser.add_argument("--chain", "-ch", type=int, default=1)
    parser.add_argument("--interactive", "-I", action="store_true")
    parser.add_argument("--environment", type=str, default="/users/jiaqchen/.edf/faive2lerobot.toml", help="Container environment file")
    parser.add_argument("--container-writable", action="store_true", help="Make container writable (for interactive)")
    parser.add_argument("--partition", "-p", type=str, default=None, help="Partition name (default: 'normal' for batch, 'debug' for interactive)")
    parser.add_argument("name", type=str)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    main(parser.parse_args())
