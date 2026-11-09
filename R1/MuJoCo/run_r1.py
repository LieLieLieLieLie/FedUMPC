"""Reproduce the verified R1 MuJoCo benchmark and trajectory figure."""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('stage', choices=['final', 'scene', 'figure'])
args = parser.parse_args()
if args.stage == 'final':
    from r1_mujoco_core import runner
    runner.main()
elif args.stage == 'scene':
    from r1_core_scene import main
    main()
else:
    from make_r1_paper_figure import main
    main()

