# specreboot/cli.py
import argparse

from .run_workflow_matchms import build_parser as build_matchms_parser, run as run_matchms
from .run_workflow_gnps import build_parser as build_gnps_parser, run as run_gnps


def cli():
    parser = argparse.ArgumentParser(prog="specreboot")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_matchms = sub.add_parser("matchms", help="Compute similarities + bootstrap + build SpecReBoot graphs")
    build_matchms_parser(p_matchms)
    p_matchms.set_defaults(func=run_matchms)

    p_gnps = sub.add_parser("gnps", help="Compute bootstrap + rescue edges and merge into an existing GNPS graph")
    build_gnps_parser(p_gnps)
    p_gnps.set_defaults(func=run_gnps)

    args = parser.parse_args()
    args.func(args)